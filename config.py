import jax.numpy as jnp
from tensorneat.common import ACT

from substrate_generation.pca_coor_generator import PCAanalyzer
from substrate_generation.fa_coor_generator import FactorAnalyzer
from substrate_generation.sdl_coor_generator import SparseDictionaryAnalyzer

config = {
    # TOP-LEVEL EXPERIMENT SETTINGS
    "experiment": {
        "env_name": "ant",
        "seed": 42,
    },
    # ENVIRONMENT CONFIGURATION
    "environment": {
        "backend": "generalized",
        "max_step": 1000,
        "repeat_times": 5,
        "args_sets": {  # Specific reward/cost weights for each environment
            "ant": {
                "healthy_reward": 0.1,
                "ctrl_cost_weight": 1e-4,
                "contact_cost_weight": 5e-5,
            },
            "halfcheetah": { "forward_reward_weight": 2.0, "ctrl_cost_weight": 0.05 },
            "swimmer": { "ctrl_cost_weight": 0.0001 }
        }
    },
    # EVOLUTION PIPELINE CONFIGURATION
    "evolution": {
        "generation_limit": 350,
        "fitness_target": 10000.0,
        "pop_size": 1000,
        "species_size": 10,
    },
    # SUBSTRATE & HYPERNEAT CONFIGURATION
    "substrate": {
        "hidden_layer_type": "shift",
        "hidden_depth": 2,
        "weight_threshold": 0.1,
        "recurrent_activations": 25,
    },
    # DATA SAMPLING
    "data_sampling": {
        "sampling_steps": 1000,
        "num_agents_to_sample": 2,
        # Config for the temporary sampling agent
        "trained_agent_sampling": {
            "generation_limit": 35,
            "pop_size": 1000,
            "species_size": 10,
            "hidden_depth": 1,
        }
    },
    # DATA ANALYSIS
    "data_analysis": {
        "variance_threshold": 0.75,
        "max_dims": lambda obs_size, act_size: int((obs_size+act_size)/4),
        "sdl_alpha": 1.0,
        "sdl_max_iter": 2000,
    },
    # CORE NEAT / CPPN GENOME CONFIGURATION
    "algorithm": {
        "conn_weight_mutate_power": 0.25,
        "conn_weight_mutate_rate": 0.3,
        "conn_weight_lower_bound": -1.0,
        "conn_weight_upper_bound": 1.0,
        "cppn_output_activation": ACT.tanh,
        "cppn_max_nodes": 256,
        "cppn_max_conns": 1024,
        "cppn_init_hidden_layers": lambda query_dim: [int(query_dim / 4)],
        "node_add_prob": 0.4,
        "conn_add_prob": 0.5,
        "node_delete_prob": 0.15,
        "conn_delete_prob": 0.2,
        "survival_threshold": 0.10,
        "compatibility_threshold": 1.0,
        "species_fitness_func": jnp.max,
        "genome_elitism": 5,
        "species_elitism": 3,
        "activation_function": ACT.tanh,
        "output_activation": ACT.tanh,
    }
}

# dynamically set params

# directory
config['experiment']['output_dir'] = f"output/{config['experiment']['env_name']}"

# Brax environment arguments
env_name = config['experiment']['env_name']
config['environment']['brax_args'] = config['environment']['args_sets'].get(env_name, {})

# expert_sample_config
config['data_sampling']['trained_agent_sampling']['env_args'] = config['environment']['brax_args']
config['data_sampling']['trained_agent_sampling']['weight_threshold'] = config['substrate']['weight_threshold']
config['data_sampling']['trained_agent_sampling']['fitness_target'] = config['evolution']['fitness_target']
