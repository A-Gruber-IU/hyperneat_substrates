import jax.numpy as jnp
from tensorneat.common import ACT, AGG

def initial_cppn_layers_dynamic(query_dim):
    if int(query_dim) >= 16:
        return [int(query_dim/5),int(query_dim/8)]
    elif int(query_dim) >= 8:
        return [int(query_dim/4)]
    else:
        return [2]
    
def initial_cppn_layers_flat(query_dim):
    return [2]
    
def initial_cppn_layers_none(query_dim):
    return [0]

config = {
    # TOP-LEVEL EXPERIMENT SETTINGS
    "experiment": {
        "env_name": "ant",
        "seed": 42,
    },
    # ENVIRONMENT CONFIGURATION
    "environment": {
        "backend": "mjx", # mjx | generalized | positional | spring --- mjx creates most physically plausible movements
        "max_step": 1000,
        "repeat_times": 8, # stabilizes convergence, but impacts run time very significantly
        "obs_normalization": False,
        "sample_episodes": 16, # related to obs_normalization
        "args_sets": {  # Specific reward/cost weights for each environment, use_contact_forces is not implemented
            "ant": {
                "healthy_reward": 0.05, # 0.05 (default 1.0)
                "ctrl_cost_weight": 0.5, # 0.05 for mjx and positional, 0.5-0.8 for spring (default 0.5)
                "contact_cost_weight": 0.0005, # (default 0.0005)
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
        "num_agents_to_sample": 2,
        # Config for the temporary sampling agent
        "trained_agent_sampling": {
            "generation_limit": 50,
            "pop_size": 1000,
            "species_size": 10,
            "hidden_depth": 1,
        }
    },
    # DATA ANALYSIS
    "data_analysis": {
        "variance_threshold": 1.0,
        "feature_dims": [1,2,5,9,14,19], # [1,2,5,9,14,19]
        "dl_alpha": 0.5,
        "dl_max_iter": 2000,
        "normalize_coors": True,
    },
    # EVOLUTION PIPELINE CONFIGURATION
    "pipeline": {
        "generation_limit": 200,
        "fitness_target": 10000.0,
    },
    # SUBSTRATE & HYPERNEAT CONFIGURATION
    "substrate": {
        "hidden_layer_type": "shift", # one_hot | one_double_hot | two_hot | shift | shift_two_in | shift_three_in | shift_two_out | shift_three_out 
        "hidden_depth": 1, # number of hidden layers
        "depth_factor": 1, # factor which "stretches" the coordinates into depth direction
        "width_factor": 1, # factor which "stretches" the coordinates in all directions, except depth
    },
    # CORE NEAT / CPPN GENOME CONFIGURATION
    "algorithm": {
        "conn_gene": {
            "conn_weight_mutate_power": 0.25, # 0.25
            "conn_weight_mutate_rate": 0.25, # 0.25
            "conn_weight_lower_bound": -5.0, # -5
            "conn_weight_upper_bound": 5.0, # 5
            "weight_replace_rate": 0.001, # 0.001 | 0.015
            "weight_init_mean": 0, # 0
            "weight_init_std": 1, # 1
        },
        "node_gene": { # node_value = act(response * agg(inputs) + bias)
            "activation_function_options": [ACT.tanh, ACT.sigmoid, ACT.sin, ACT.relu, ACT.identity],
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
            "node_add_prob": 0.3, # 0.2 for initial_cppn_layers_dynamic, 0.3 | 0.4 for initial_cppn_layers_none | 0.3 initial_cppn_layers_flat
            "conn_add_prob": 0.6, # 0.2 for initial_cppn_layers_dynamic, 0.4 | 0.6 for initial_cppn_layers_none | 0.6 initial_cppn_layers_flat
            "node_delete_prob": 0.03, # 0.001-0.05 
            "conn_delete_prob": 0.03, # 0.001-0.05
        },
        "genome": {
            "cppn_output_activation": ACT.tanh, # ACT.tanh, ACT.scaled_tanh is scaled by a factor of 3
            "cppn_max_nodes": 128,
            "cppn_max_conns": 256,
            "cppn_init_hidden_layers": lambda query_dim: initial_cppn_layers_flat(query_dim=query_dim),
        },
        "neat": {
            "pop_size": 3000, # 3500
            "genome_elitism": 2, # 3|5
            "species_elitism": 3, # 5
            "species_size": 20, # 25|35|50 
            "min_species_size": 10, # 5|10
            "survival_threshold": 0.025, # 0.05 | 0.1
            "compatibility_threshold": 0.75, # 0.75
            "species_fitness_func": jnp.max, # jnp.max
            "species_number_calculate_by": "rank", # fitness | rank
            "max_stagnation": 15, # 15|20
        },
        "hyperneat": {
            "activation_function": ACT.tanh,
            "output_activation": ACT.tanh,
            "weight_threshold": 0.005, # 0.005 | 0.05
            "max_weight": 1, # 1.5 | 2.5
            "recurrent_activations": 5, # 5|10
        },
    }
}

# dynamically set params

# directory
config['experiment']['output_dir'] = f"output/{config['experiment']['env_name']}"

# Brax environment arguments
env_name = config['experiment']['env_name']
config['environment']['brax_args'] = config['environment']['args_sets'].get(env_name, {})

