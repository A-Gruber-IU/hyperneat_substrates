import numpy as np
import networkx as nx
import jax
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.image as mpimg
from IPython.display import SVG
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def visualize_cppn(pipeline, state, save_path):
    # visualize cppn
    best_genome = pipeline.best_genome
    cppn_genome = pipeline.algorithm.neat.genome
    cppn_network = cppn_genome.network_dict(state, *best_genome)
    print(f"Visualizing CPPN. Saving to {save_path}.")
    cppn_genome.visualize(cppn_network, save_path=save_path)
    SVG(filename=save_path)

def visualize_nn(pipeline, state, save_path, substrate, input_coors, hidden_coors, output_coors, hidden_depth, max_weight):

    weight_lower_limit = -max_weight
    weight_upper_limit = max_weight

    best_genome = pipeline.best_genome
    print("Manually reconstructing the phenotype. A visual layout will be generated.")

    # 1. Get Weights from CPPN ---
    neat_algorithm = pipeline.algorithm.neat
    cppn_params = neat_algorithm.transform(state, best_genome)
    query_coors = substrate.query_coors
    cppn_forward_func = neat_algorithm.forward

    all_substrate_weights = jax.vmap(
        cppn_forward_func, in_axes=(None, None, 0)
    )(state, cppn_params, query_coors)

    all_substrate_connections = np.array(substrate.conns)
    all_substrate_weights_np = np.array(all_substrate_weights).squeeze()

    # 2. Filter Edges by Weight Threshold ---
    internal_weight_threshold = pipeline.algorithm.weight_threshold
    active_mask = np.abs(all_substrate_weights_np) > internal_weight_threshold
    active_conns = all_substrate_connections[active_mask]
    active_weights = all_substrate_weights_np[active_mask]

    print(f"Substrate has {len(all_substrate_connections)} potential connections.")

    # 3. Build Graph and Assign Layers (Simplified) ---
    G_to_draw = nx.DiGraph()
    all_node_keys = [int(n[0]) for n in substrate.nodes]

    num_inputs  = len(input_coors)
    num_outputs = len(output_coors)
    num_hiddens = len(hidden_coors)

    input_keys  = all_node_keys[:num_inputs]
    output_keys = all_node_keys[num_inputs : num_inputs + num_outputs]
    hidden_keys = all_node_keys[num_inputs + num_outputs:]

    # Inputs are at layer 0
    for key in input_keys:
        G_to_draw.add_node(key, subset=0)

    # Simplified hidden layer logic (assumes all layers have the same width)
    if hidden_depth > 0 and num_hiddens > 0:
        hidden_width = num_hiddens // hidden_depth
        if num_hiddens % hidden_depth != 0:
            print(f"Warning: Number of hidden nodes ({num_hiddens}) is not evenly divisible by hidden_depth ({hidden_depth}). Visualization may be inaccurate.")

        for i in range(hidden_depth):
            layer_subset_id = i + 1
            start_index = i * hidden_width
            end_index = start_index + hidden_width
            layer_keys = hidden_keys[start_index:end_index]
            for key in layer_keys:
                G_to_draw.add_node(key, subset=layer_subset_id)

    # Outputs are at the final layer
    output_layer_id = hidden_depth + 1
    for key in output_keys:
        G_to_draw.add_node(key, subset=output_layer_id)

    # Generate the layout based on the 'subset' attribute
    pos = nx.multipartite_layout(G_to_draw, subset_key='subset')

    # 4. Process and Filter Edges for Drawing ---
    ac = np.asarray(active_conns)
    if ac.size == 0:
        edges_to_add = []
        active_weights = np.array([])
    else:
        all_edges = [(int(row[0]), int(row[1])) for row in ac]
        all_weights = np.asarray(active_weights)
        edges_to_add = []
        active_weights_filtered = []
        for edge, weight in zip(all_edges, all_weights):
            if edge[0] != edge[1]: # Filter out self-loops
                edges_to_add.append(edge)
                active_weights_filtered.append(weight)
        active_weights = np.array(active_weights_filtered)

    print(f"Visualizing {len(active_weights)} connections. Excluded loops. Weight threshold: {internal_weight_threshold}")
    G_to_draw.add_edges_from(edges_to_add)

    # 5. Prepare for Drawing (Colors, Weights, etc.) ---
    node_colors = []
    for node_key in G_to_draw.nodes():
        if node_key in input_keys: color = 'blue'
        elif node_key in output_keys: color = 'red'
        else: color = 'green'
        node_colors.append(color)

    fig, ax = plt.subplots(figsize=(12, 12))
    
    weights = np.asarray(active_weights)
    if weights.size > 0:
        idx_all = np.arange(len(edges_to_add))
        pos_idx = idx_all[weights > 0]
        neg_idx = idx_all[weights < 0]

        edges_pos = [edges_to_add[i] for i in pos_idx]
        edges_neg = [edges_to_add[i] for i in neg_idx]
        w_pos = weights[pos_idx]
        w_neg_mag = -weights[neg_idx]

        eps = np.finfo(float).eps
        widths_pos = 0.5 + 1.5 * np.clip(w_pos / max(weight_upper_limit, eps), 0.0, 1.0)
        widths_neg = 0.5 + 1.5 * np.clip(w_neg_mag / max(-weight_lower_limit, eps), 0.0, 1.0)
    else:
        edges_pos, edges_neg, w_pos, w_neg_mag, widths_pos, widths_neg = [], [], [], [], [], []

    # 6. Draw the Network ---
    nx.draw_networkx_nodes(G_to_draw, pos=pos, node_color=node_colors, node_size=20, ax=ax)

    if len(edges_pos):
        nx.draw_networkx_edges(
            G_to_draw, pos=pos, edgelist=edges_pos, edge_color=w_pos,
            edge_cmap=plt.cm.Greys, edge_vmin=0.0, edge_vmax=float(weight_upper_limit),
            width=widths_pos, arrows=True, arrowstyle='-|>', arrowsize=4, ax=ax
        )
    if len(edges_neg):
        nx.draw_networkx_edges(
            G_to_draw, pos=pos, edgelist=edges_neg, edge_color=w_neg_mag,
            edge_cmap=plt.cm.Reds, edge_vmin=0.0, edge_vmax=float(-weight_lower_limit),
            width=widths_neg, arrows=True, arrowstyle='-|>', arrowsize=4, ax=ax
        )

    # Add Titles and Colorbars
    ax.set_title(f"Substrate Network — Positives Greys, Negatives Reds (Bounds [{weight_lower_limit}, {weight_upper_limit}])")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    # Left: negative (Reds)
    if len(w_neg_mag):
        sm_neg = ScalarMappable(cmap=plt.cm.Reds,
                                norm=Normalize(vmin=0.0, vmax=float(-weight_lower_limit)))
        sm_neg.set_array([])
        cax_neg = inset_axes(
            ax, width="32%", height="3%", loc="lower left",
            bbox_to_anchor=(0.06, -0.12, 1.0, 1.0),  # left aligned
            bbox_transform=ax.transAxes, borderpad=0
        )
        cbar_neg = fig.colorbar(sm_neg, cax=cax_neg, orientation='horizontal')
        cbar_neg.set_label('Negative |weight|', labelpad=2)
        cbar_neg.ax.xaxis.set_label_position('bottom')
        cbar_neg.ax.xaxis.set_ticks_position('bottom')
        cbar_neg.set_ticks([0, (-weight_lower_limit)/2, -weight_lower_limit])

    # Right: positive (Greys)
    if len(w_pos):
        sm_pos = ScalarMappable(cmap=plt.cm.Greys,
                                norm=Normalize(vmin=0.0, vmax=float(weight_upper_limit)))
        sm_pos.set_array([])
        cax_pos = inset_axes(
            ax, width="32%", height="3%", loc="lower right",
            bbox_to_anchor=(-0.06, -0.12, 1.0, 1.0),  # right aligned
            bbox_transform=ax.transAxes, borderpad=0
        )
        cbar_pos = fig.colorbar(sm_pos, cax=cax_pos, orientation='horizontal')
        cbar_pos.set_label('Positive weight', labelpad=2)
        cbar_pos.ax.xaxis.set_label_position('bottom')
        cbar_pos.ax.xaxis.set_ticks_position('bottom')
        cbar_pos.set_ticks([0, weight_upper_limit/2, weight_upper_limit])

    out_path = f"{save_path}"
    fig.savefig(out_path, dpi=800)
    # plt.show()
    plt.close(fig)

    print(f"Visualization saved to: {out_path}")



def display_plots_side_by_side(plot_paths: list, plot_titles: list, main_title: str, figsize_per_plot: tuple = (10, 10), save_path: str = "output"):
    """
    Loads a list of saved image files and displays them side-by-side in a single figure.
    """
    if not plot_paths:
        print("Warning: No plot paths provided to display.")
        return

    assert len(plot_paths) == len(plot_titles), \
        f"Mismatch between number of plots ({len(plot_paths)}) and titles ({len(plot_titles)})."

    num_plots = len(plot_paths)
    
    # Calculate the total figure size
    total_width = figsize_per_plot[0] * num_plots
    height = figsize_per_plot[1]
    
    # Create the figure and the subplots (axes)
    fig, axes = plt.subplots(1, num_plots, figsize=(total_width, height))
    
    # If there's only one plot, axes is not an array, so we make it one
    if num_plots == 1:
        axes = [axes]

    # Loop through the paths and titles to display each image
    for i, (path, title) in enumerate(zip(plot_paths, plot_titles)):
        try:
            # Load the saved image from disk
            img = mpimg.imread(path)
            
            # Display the image on the corresponding subplot axis
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=16)
            
            # Hide the axis ticks for a cleaner look
            axes[i].axis('off')
        except FileNotFoundError:
            print(f"Error: Could not find image at path: {path}")
            axes[i].set_title(f"Image not found:\n{path}", color='red')
            axes[i].axis('off')

    fig.suptitle(main_title, fontsize=20)
    
    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(save_path, dpi=800)
    # Render the figure in the notebook output
    plt.show()