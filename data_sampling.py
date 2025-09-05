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
    (This function remains unchanged).
    """
    print(f"Starting data collection for {num_steps} steps using a random policy...")
    
    key, reset_key = jax.random.split(key)
    initial_obs, initial_env_state = env_problem.env_reset(reset_key)
    
    def policy_step(carry, key):
        env_state, last_obs = carry
        action = jax.random.uniform(
            key, shape=(env_problem.output_shape[0],), minval=-1.0, maxval=1.0
        )
        next_obs, next_env_state, _, _, _ = env_problem.env_step(
            jax.random.PRNGKey(0), env_state, action
        )
        snapshot = jnp.concatenate([last_obs, action])
        return (next_env_state, next_obs), snapshot

    jitted_policy_step = jax.jit(policy_step)
    step_keys = jax.random.split(key, num_steps)
    initial_carry = (initial_env_state, initial_obs)
    (_, _), collected_data = jax.lax.scan(jitted_policy_step, initial_carry, step_keys)

    print("Random data collection finished.")
    return jax.device_get(collected_data)


def collect_trained_agent_policy_data(
    env_problem,
    key,
    num_steps: int,
    training_config: dict
) -> np.ndarray:
    """
    Fully encapsulates the process of training a temporary "expert" agent with
    an MLP substrate and then collects data by running it in the environment.
    """
    print("\n--- Starting Expert Training and Data Collection ---")

    # --- Phase 1: Train a baseline agent with a simple MLP substrate ---
    print("--> Step 1: Configuring and training the expert agent...")

    # Unpack config and setup environment parameters
    obs_size = env_problem.input_shape[0]
    act_size = env_problem.output_shape[0]
    
    # Create the MLP Substrate
    hidden_width_mlp = int(obs_size * 1.5)
    mlp_layers = [obs_size+1] + [hidden_width_mlp] * training_config["hidden_depth"] + [act_size]
    mlp_substrate = MLPSubstrate(layers=mlp_layers)
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
        seed=int(pipeline_key[1]), # Use part of the key for a deterministic seed
        generation_limit=training_config["generation_limit"],
        fitness_target=config["evolution"]["fitness_target"],
    )
    
    init_state = pipeline.setup()
    trained_state, best_genome = pipeline.auto_run(state=init_state)

    print("--> Step 1 Finished: Expert agent has been trained.")

    # --- Phase 2: Collect data using the trained agent ---
    print(f"--> Step 2: Collecting {num_steps} data points using the expert policy...")
    
    params = pipeline.algorithm.transform(trained_state, best_genome)
    act_func = pipeline.algorithm.forward

    key, reset_key = jax.random.split(key)
    initial_obs, initial_env_state = env_problem.env_reset(reset_key)

    def policy_step(carry, _):
        env_state, last_obs = carry
        action = act_func(trained_state, params, last_obs)
        next_obs, next_env_state, _, _, _ = env_problem.env_step(
            jax.random.PRNGKey(0), env_state, action
        )
        snapshot = jnp.concatenate([last_obs, action])
        return (next_env_state, next_obs), snapshot

    jitted_policy_step = jax.jit(policy_step)
    initial_carry = (initial_env_state, initial_obs)
    (_, _), collected_data = jax.lax.scan(jitted_policy_step, initial_carry, None, length=num_steps)
    
    print("--- Expert Training and Data Collection Finished ---\n")
    return jax.device_get(collected_data)