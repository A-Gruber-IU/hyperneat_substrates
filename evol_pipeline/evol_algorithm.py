from config import config

from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT
from tensorneat.genome import DefaultGenome, RecurrentGenome
from tensorneat.genome.operations import DefaultMutation
from tensorneat.genome.gene import DefaultConn, BiasNode, DefaultNode
from tensorneat.common import ACT, AGG

def create_evol_algorithm(substrate, sampling = False):
    query_dim = int(substrate.query_coors.shape[1])

    algo_params = config["algorithm"]
    
    conn_gene = DefaultConn(
        weight_mutate_power=algo_params["conn_gene"]["conn_weight_mutate_power"],
        weight_mutate_rate=algo_params["conn_gene"]["conn_weight_mutate_rate"],
        weight_replace_rate=algo_params["conn_gene"]["weight_replace_rate"],
        weight_lower_bound=algo_params["conn_gene"]["conn_weight_lower_bound"],
        weight_upper_bound=algo_params["conn_gene"]["conn_weight_upper_bound"],
        weight_init_mean=algo_params["conn_gene"]["weight_init_mean"],
        weight_init_std=algo_params["conn_gene"]["weight_init_std"],
    )

    bias_node_gene=BiasNode(
        bias_init_mean=algo_params["node_gene"]["bias_init_mean"],
        bias_init_std=algo_params["node_gene"]["bias_init_std"],
        bias_mutate_power=algo_params["node_gene"]["bias_mutate_power"],
        bias_mutate_rate=algo_params["node_gene"]["bias_mutate_rate"],
        bias_replace_rate=algo_params["node_gene"]["bias_replace_rate"],
        bias_lower_bound=algo_params["node_gene"]["bias_lower_bound"],
        bias_upper_bound=algo_params["node_gene"]["bias_upper_bound"],
        activation_options=algo_params["node_gene"]["activation_function_options"],
        activation_default=algo_params["node_gene"]["activation_default"],
        activation_replace_rate=algo_params["node_gene"]["activation_replace_rate"],
        aggregation_options=algo_params["node_gene"]["aggregation_options"],
        aggregation_default=algo_params["node_gene"]["aggregation_default"],
        aggregation_replace_rate=algo_params["node_gene"]["aggregation_replace_rate"],
    )

    default_node_gene=DefaultNode(
        bias_init_mean=algo_params["node_gene"]["bias_init_mean"],
        bias_init_std=algo_params["node_gene"]["bias_init_std"],
        bias_mutate_power=algo_params["node_gene"]["bias_mutate_power"],
        bias_mutate_rate=algo_params["node_gene"]["bias_mutate_rate"],
        bias_replace_rate=algo_params["node_gene"]["bias_replace_rate"],
        bias_lower_bound=algo_params["node_gene"]["bias_lower_bound"],
        bias_upper_bound=algo_params["node_gene"]["bias_upper_bound"],
        activation_options=algo_params["node_gene"]["activation_function_options"],
        activation_default=algo_params["node_gene"]["activation_default"],
        activation_replace_rate=algo_params["node_gene"]["activation_replace_rate"],
        aggregation_options=algo_params["node_gene"]["aggregation_options"],
        aggregation_default=algo_params["node_gene"]["aggregation_default"],
        aggregation_replace_rate=algo_params["node_gene"]["aggregation_replace_rate"],
        response_init_mean=algo_params["node_gene"]["response_init_mean"],
        response_init_std=algo_params["node_gene"]["response_init_std"],
        response_lower_bound=algo_params["node_gene"]["response_lower_bound"],
        response_upper_bound=algo_params["node_gene"]["response_upper_bound"],
        response_replace_rate=algo_params["node_gene"]["response_replace_rate"],
        response_mutate_power=algo_params["node_gene"]["response_mutate_power"],
        response_mutate_rate=algo_params["node_gene"]["response_mutate_rate"],
    )

    mutation=DefaultMutation(
        node_add=algo_params["mutation"]["node_add_prob"], 
        conn_add=algo_params["mutation"]["conn_add_prob"], 
        node_delete=algo_params["mutation"]["node_delete_prob"], 
        conn_delete=algo_params["mutation"]["conn_delete_prob"],
    )

    genome=DefaultGenome(
        num_inputs=query_dim, 
        num_outputs=1,
        input_transform=None,
        output_transform=algo_params["genome"]["cppn_output_activation"],
        max_nodes=algo_params["genome"]["cppn_max_nodes"],
        max_conns=algo_params["genome"]["cppn_max_conns"],
        init_hidden_layers=algo_params["genome"]["cppn_init_hidden_layers"](query_dim),
        mutation=mutation,
        node_gene=default_node_gene,
        conn_gene=conn_gene,
    )

    neat_algorithm = NEAT(
        pop_size=config["data_sampling"]["trained_agent_sampling"]["pop_size"] if sampling else algo_params["neat"]["pop_size"],
        species_size=config["data_sampling"]["trained_agent_sampling"]["species_size"] if sampling else algo_params["neat"]["species_size"],
        min_species_size=1 if sampling else algo_params["neat"]["min_species_size"],
        survival_threshold=algo_params["neat"]["survival_threshold"],
        compatibility_threshold=algo_params["neat"]["compatibility_threshold"],
        species_fitness_func=algo_params["neat"]["species_fitness_func"],
        genome_elitism=algo_params["neat"]["genome_elitism"],
        species_elitism=algo_params["neat"]["species_elitism"],
        genome=genome,
        species_number_calculate_by=algo_params["neat"]["species_number_calculate_by"],
        max_stagnation=algo_params["neat"]["max_stagnation"]
    )

    evol_algorithm = HyperNEAT(
        substrate=substrate,
        neat=neat_algorithm,
        activation=algo_params["hyperneat"]["activation_function"],
        activate_time=algo_params["hyperneat"]["recurrent_activations"],
        output_transform=algo_params["hyperneat"]["output_activation"],
        weight_threshold=algo_params["hyperneat"]["weight_threshold"],
        max_weight=algo_params["hyperneat"]["max_weight"]
    )

    return evol_algorithm