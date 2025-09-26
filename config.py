import jax.numpy as jnp
from tensorneat.common import ACT, AGG

def initial_cppn_layers2(query_dim):
    if int(query_dim) >= 16:
        return [int(query_dim / 2),int(query_dim / 6)]
    elif int(query_dim) >= 8:
        return [int(query_dim / 3)]
    else:
        return [2]
    
def initial_cppn_layers(query_dim):
    return [0]

config = {
    # TOP-LEVEL EXPERIMENT SETTINGS
    "experiment": {
        "env_name": "ant",
        "seed": 42,
    },
    # ENVIRONMENT CONFIGURATION
    "environment": {
        "backend": "spring", # mjx | generalized | positional | spring
        "max_step": 1000,
        "repeat_times": 35, # 35 ref - stabilizes convergence, but impacts run time very significantly (repetitions are not parallelized)
        "args_sets": {  # Specific reward/cost weights for each environment
            "ant": {
                "healthy_reward": 0.05, # default 1.0, 0.05 ref for mjx and spring
                "ctrl_cost_weight": 0.5, # default 0.5, 0.02 | 0.05 | 0.5 ref for mjx, 0.5 | 0.8 ref for spring
                "contact_cost_weight": 0.0005, # default 0.0005, use_contact_forces is not implemented
            },
            "halfcheetah": { 
                "forward_reward_weight": 2.0, 
                "ctrl_cost_weight": 0.05 
                },
            "swimmer": { 
                "ctrl_cost_weight": 0.0001 
                }
        }
    },
    # DATA SAMPLING
    "data_sampling": {
        "sampling_steps": 1000,
        "num_agents_to_sample": 1,
        # Config for the temporary sampling agent
        "trained_agent_sampling": {
            "generation_limit": 100,
            "pop_size": 1000,
            "species_size": 20,
            "hidden_depth": 1,
        }
    },
    # DATA ANALYSIS
    "data_analysis": {
        "variance_threshold": 0.65,
        "feature_dims": 8,
        "sdl_alpha": 1.0,
        "sdl_max_iter": 2000,
        "normalize_coors": True,
    },
    # EVOLUTION PIPELINE CONFIGURATION
    "pipeline": {
        "generation_limit": 500,
        "fitness_target": 10000.0,
    },
    # SUBSTRATE & HYPERNEAT CONFIGURATION
    "substrate": {
        "hidden_layer_type": "shift", # one_hot | one_double_hot | two_hot | shift | shift_two | shift_three
        "hidden_depth": 2, # number of hidden layers
        "depth_factor": 1, # factor which "stretches" the coordinates into depth direction
        "width_factor": 1, # factor which "stretches" the coordinates in all directions, except depth
    },
    # CORE NEAT / CPPN GENOME CONFIGURATION
    "algorithm": {
        "conn_gene": {
            "conn_weight_mutate_power": 0.25, # 0.2 | 0.25 ref
            "conn_weight_mutate_rate": 0.25, # 0.25 ref
            "conn_weight_lower_bound": -5.0, # -5 ref
            "conn_weight_upper_bound": 5.0, # 5 ref
            "weight_replace_rate": 0.015, # 0.015 ref
            "weight_init_mean": 0, # 0 ref
            "weight_init_std": 1, # 1 ref
        },
        "node_gene": {
            "activation_function_options": [ACT.tanh, ACT.sigmoid, ACT.sin, ACT.relu, ACT.identity, ACT.scaled_tanh],
            "activation_default": ACT.tanh,
            "activation_replace_rate": 0.1,
            "aggregation_options": [AGG.sum, AGG.product],
            "aggregation_default": AGG.sum,
            "aggregation_replace_rate": 0.1,
            "bias_init_mean": 0,
            "bias_init_std": 0.1,
            "bias_mutate_power": 0.05,
            "bias_mutate_rate": 0.01,
            "bias_replace_rate": 0.015,
            "bias_lower_bound": -5,
            "bias_upper_bound": 5,
            "response_init_mean": 1,
            "response_init_std": 0,
            "response_lower_bound": -5,
            "response_upper_bound": 5,
            "response_replace_rate": 0.015,
            "response_mutate_power": 0.15,
            "response_mutate_rate": 0.2,
        },
        "mutation": {
            "node_add_prob": 0.1, # 0.2 ref
            "conn_add_prob": 0.2, # 0.2 | 0.3 ref
            "node_delete_prob": 0.05, # 0.05 | 0.1 | 0.15 ref
            "conn_delete_prob": 0.05, # 0.05 | 0.1 | 0.15 | 0.2 ref
        },
        "genome": {
            "cppn_output_activation": ACT.tanh, # ACT.scaled_tanh is scaled by a factor of 3
            "cppn_max_nodes": 128,
            "cppn_max_conns": 512,
            "cppn_init_hidden_layers": lambda query_dim: initial_cppn_layers(query_dim=query_dim),
        },
        "neat": {
            "pop_size": 700, # note at 16GB VRAM for recurrent network with "shift": 1000 for hidden_depth = 1, 700 for hidden_depth = 2, 300 for hidden_depth = 3
            "genome_elitism": 5, # 3 | 5 ref
            "species_elitism": 5, # 3 | 5 ref
            "species_size": 50, # 50 ref
            "min_species_size": 5,
            "survival_threshold": 0.05, # 0.05 | 0.1 ref
            "compatibility_threshold": 0.75, # 0.8 | 1.0 ref
            "species_fitness_func": jnp.max, # jnp.max ref
            "species_number_calculate_by": "rank", # fitness | rank
            "max_stagnation": 15, # 15 ref
        },
        "hyperneat": {
            "activation_function": ACT.tanh,
            "output_activation": ACT.tanh,
            "weight_threshold": 0.05, # 0.1 ref
            "max_weight": 1, # 1.5 | 2.5 ref
            "recurrent_activations": 10, # 10 ref
        },
    }
}

# dynamically set params

# directory
config['experiment']['output_dir'] = f"output/{config['experiment']['env_name']}"

# Brax environment arguments
env_name = config['experiment']['env_name']
config['environment']['brax_args'] = config['environment']['args_sets'].get(env_name, {})

