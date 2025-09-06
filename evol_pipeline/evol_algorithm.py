from config import config

from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT
from tensorneat.genome import DefaultGenome
from tensorneat.genome.operations import DefaultMutation
from tensorneat.genome.gene import DefaultConn
from tensorneat.common import ACT

def create_evol_algorithm(substrate):
    query_dim = int(substrate.query_coors.shape[1])

    algo_params = config["algorithm"] # for convenience

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
        pop_size=config["evolution"]["pop_size"],
        species_size=config["evolution"]["species_size"],
        survival_threshold=algo_params["survival_threshold"],
        compatibility_threshold=algo_params["compatibility_threshold"],
        species_fitness_func=algo_params["species_fitness_func"],
        genome_elitism=algo_params["genome_elitism"],
        species_elitism=algo_params["species_elitism"],
        genome=genome,
    )

    evol_algorithm = HyperNEAT(
        substrate=substrate,
        neat=neat_algorithm,
        activation=algo_params["activation_function"],
        activate_time=config["substrate"]["recurrent_activations"],
        output_transform=algo_params["output_activation"],
        weight_threshold=config["substrate"]["weight_threshold"],
    )

    return evol_algorithm