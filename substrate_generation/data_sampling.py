import jax
import jax.numpy as jnp
import numpy as np
from config import config
from tensorneat import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEATFeedForward, MLPSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.genome.operations import DefaultMutation
from tensorneat.genome.gene import DefaultConn
from tensorneat.common import ACT


def collect_random_policy_data(env_problem, key, num_steps) -> np.ndarray:
    """
    Collects data by running a random policy in the environment.
    """
    print(f"Starting data collection for {num_steps} steps using a random policy...")
    
    # splitting key ensures the randomness of the reset is separate from the randomness of the actions
    key, reset_key = jax.random.split(key)

    # resets the environment to get the starting observation and env_state
    initial_obs, initial_env_state = env_problem.env_reset(reset_key)
    
    # a zero-vector is used as initial action
    initial_action = jnp.zeros(env_problem.output_shape)
    
    # the "carry" holds the action from the last step
    initial_carry = (initial_env_state, initial_action)
    
    # defines the single-step function for the JAX loop
    def policy_step(carry, key):
        # unpacks the state from the previous iteration
        env_state, last_action = carry
        
        # generate random action
        current_action = jax.random.uniform(
            key, shape=(env_problem.output_shape[0],), minval=-1.0, maxval=1.0
        )
        
        # steps the environment using the current action
        next_obs, next_env_state, _, _, _ = env_problem.env_step(
            jax.random.PRNGKey(0), env_state, current_action
        )
        
        # The snapshot records the action from the previous step (last_action) and the observation that RESULTED from it (next_obs).
        snapshot = jnp.concatenate([last_action, next_obs])
        
        # The new carry for the next iteration must include the action we just took.
        new_carry = (next_env_state, current_action)

        return new_carry, snapshot

    # Compiles and runs the simulation loop
    jitted_policy_step = jax.jit(policy_step) # jit traces the function's operations and converts it to XLA
    step_keys = jax.random.split(key, num_steps) # takes the main simulation key and split it into num_steps unique sub-keys
    
    # final carry is irrelevant, collected data holds array of shape (num_steps, obs_size + act_size)
    (_, _), collected_data = jax.lax.scan(jitted_policy_step, initial_carry, step_keys)

    print("Causal data collection finished.")
    return jax.device_get(collected_data)


def collect_trained_agent_policy_data(
    env_problem,
    key,
    num_steps: int,
    training_config: dict
) -> np.ndarray:
    """
    Fully encapsulates the process of training a temporary agent with
    a simple MLP substrate and then collects data by running it in the environment.
    """
    print("\nStarting Agent Training and Data Collection")

    # Train a baseline agent with a simple MLP substrate
    print("Configuring and training the agent...")

    # Unpack config and setup environment parameters
    obs_size = env_problem.input_shape[0]
    act_size = env_problem.output_shape[0]
    
    # Create the MLP Substrate
    hidden_layer = [obs_size] * training_config["hidden_depth"]
    input_layer = [obs_size+1]
    output_layer = [act_size]
    mlp_layers = input_layer + hidden_layer + output_layer
    mlp_depth = int(training_config["depth_factor"])
    half_mlp_width = int(training_config["width_factor"]/2)
    mlp_substrate = MLPSubstrate(layers=mlp_layers, coor_range=(-half_mlp_width, half_mlp_width, 0, mlp_depth))
    query_dim = int(mlp_substrate.query_coors.shape[1])
    print("Query dimension for sampling: ", query_dim)

    algo_params = config["algorithm"] # for convenience

    # Create the NEAT components from the config
    conn_gene = DefaultConn(
        weight_mutate_power=algo_params["conn_weight_mutate_power"],
        weight_mutate_rate=algo_params["conn_weight_mutate_rate"],
        weight_lower_bound=algo_params["conn_weight_lower_bound"],
        weight_upper_bound=algo_params["conn_weight_upper_bound"],
    )

    genome=DefaultGenome(
        num_inputs=query_dim, 
        num_outputs=1,
        output_transform=algo_params["cppn_output_activation"],
        max_nodes=algo_params["cppn_max_nodes"],
        max_conns=algo_params["cppn_max_conns"],
        init_hidden_layers=algo_params["cppn_init_hidden_layers"](query_dim),
        mutation=DefaultMutation(
            node_add=algo_params["node_add_prob"], conn_add=algo_params["conn_add_prob"], 
            node_delete=algo_params["node_delete_prob"], conn_delete=algo_params["conn_delete_prob"],
        ),
        conn_gene=conn_gene,
    )

    neat_algorithm = NEAT(
        pop_size=training_config["pop_size"], 
        species_size=training_config["species_size"],
        survival_threshold=algo_params["survival_threshold"],
        compatibility_threshold=algo_params["compatibility_threshold"],
        species_fitness_func=algo_params["species_fitness_func"],
        genome_elitism=algo_params["genome_elitism"],
        species_elitism=algo_params["species_elitism"],
        genome=genome,
    )

    evol_algorithm_mlp = HyperNEATFeedForward(
        substrate=mlp_substrate, 
        neat=neat_algorithm, 
        activation=algo_params["activation_function"],
        output_transform=algo_params["output_activation"],
        weight_threshold=config["substrate"]["weight_threshold"],
    )

    # Setup and run the training pipeline
    key, pipeline_key = jax.random.split(key)
    pipeline = Pipeline(
        algorithm=evol_algorithm_mlp,
        problem=env_problem,
        seed=int(pipeline_key[1]), # Uses part of the key for a deterministic seed
        generation_limit=training_config["generation_limit"],
        fitness_target=config["evolution"]["fitness_target"],
    )
    
    init_state = pipeline.setup()
    trained_state, best_genome = pipeline.auto_run(state=init_state)

    sampled_data = sample_from_pretrained_agent(key, trained_state, best_genome, pipeline, env_problem, num_steps)

    return sampled_data


def sample_from_pretrained_agent(key, trained_state, best_genome, pipeline, env_problem, num_steps):

    # Collect data using the trained agent
    print(f"Collecting {num_steps} data points using the trained agent policy...")
    
    params = pipeline.algorithm.transform(trained_state, best_genome)
    act_func = pipeline.algorithm.forward

    key, reset_key = jax.random.split(key)
    initial_obs, initial_env_state = env_problem.env_reset(reset_key)

    # a zero-vector is used as initial action to be recorded
    initial_action = jnp.zeros(env_problem.output_shape)

    # the initial values are stored in an initial carry
    initial_carry = (initial_env_state, initial_obs, initial_action)
    
    # defines the single-step function for the JAX loop
    def policy_step(carry, _): # The second argument is unused for a deterministic policy
        # unpacks the carried over state from the previous iteration
        env_state, last_obs, last_action = carry
        
        # The policy network computes the action for the current step (action_t) based on the observation from the last step (obs_t).
        current_action = act_func(trained_state, params, last_obs)
        
        # Take the step in the environment using the current action
        next_obs, next_env_state, _, _, _ = env_problem.env_step(
            jax.random.PRNGKey(0), env_state, current_action
        )
        
        # snapshot records action from the previous step (last_action) and observation that resulted from it (next_obs).
        snapshot = jnp.concatenate([last_action, next_obs])
        
        # defines the values to be carries over into the next loop
        new_carry = (next_env_state, next_obs, current_action)
        
        return new_carry, snapshot

    # Compile and run the simulation loop
    jitted_policy_step = jax.jit(policy_step)
    
    # collected data is the only return value of interest
    (_, _, _), collected_data = jax.lax.scan(jitted_policy_step, initial_carry, None, length=num_steps)
    
    print("Causal expert data collection finished.\n")
    return jax.device_get(collected_data)