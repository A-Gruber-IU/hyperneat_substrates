import numpy as np
import networkx as nx
import jax
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.image as mpimg
from IPython.display import SVG

def visualize_cppn(pipeline, state, save_path):
    # visualize cppn
    best_genome = pipeline.best_genome
    cppn_genome = pipeline.algorithm.neat.genome
    cppn_network = cppn_genome.network_dict(state, *best_genome)
    print(f"Visualizing CPPN. Saving to {save_path}.")
    cppn_genome.visualize(cppn_network, save_path=save_path)
    SVG(filename=save_path)

def visualize_nn(pipeline, state, output_dir, substrate, input_coors, hidden_coors, output_coors, hidden_depth):

    best_genome = pipeline.best_genome
    print("Manually reconstructing the phenotype. A visual layout will be generated.")

    # 1) Weights from CPPN (your existing logic)
    neat_algorithm = pipeline.algorithm.neat
    cppn_params = neat_algorithm.transform(state, best_genome)
    query_coors = substrate.query_coors
    cppn_forward_func = neat_algorithm.forward

    all_substrate_weights = jax.vmap(
        cppn_forward_func, in_axes=(None, None, 0)
    )(state, cppn_params, query_coors)

    all_substrate_connections = np.array(substrate.conns)
    all_substrate_weights_np = np.array(all_substrate_weights).squeeze()

    # 2) Select edges: no percentile pruning; keep internal threshold (toggleable)
    internal_weight_threshold = pipeline.algorithm.weight_threshold
    active_mask = np.abs(all_substrate_weights_np) > internal_weight_threshold
    active_conns = all_substrate_connections[active_mask]
    active_weights = all_substrate_weights_np[active_mask]

    # If you want literally every potential connection regardless of threshold:
    # active_conns = all_substrate_connections
    # active_weights = all_substrate_weights_np

    print(f"Substrate has {len(all_substrate_connections)} potential connections.")

    # Build graph, assign layers, generate layout
    G_to_draw = nx.DiGraph()
    all_node_keys = [int(n[0]) for n in substrate.nodes]


    # Which coordinate dimension encodes "layer"? In your code it's the last one.
    LAYER_AXIS = -1  # last coordinate

    def compute_hidden_layer_groups(hidden_coors, layer_axis=LAYER_AXIS):
        """
        Returns:
        order_vals: sorted unique layer values (e.g., 3, 6, 9, ...)
        idx_groups: list of lists; each inner list has indices of hidden nodes that belong to that layer
        widths:     number of hidden nodes per layer (len of each group)
        """
        hc = np.asarray(hidden_coors)
        if hc.ndim != 2:
            raise ValueError(f"hidden_coors must be 2D (num_hidden, coord_dims); got shape {hc.shape}")

        layer_vals = hc[:, layer_axis]
        order_vals = np.unique(layer_vals)
        idx_groups = [np.where(layer_vals == v)[0].tolist() for v in order_vals]
        widths = [len(g) for g in idx_groups]
        return widths

    # Example usage:
    hidden_widths = compute_hidden_layer_groups(hidden_coors, layer_axis=LAYER_AXIS)
    # All node keys in substrate order (N,1) -> flatten to ints
    all_node_keys = [int(n[0]) for n in substrate.nodes]

    num_inputs  = len(input_coors)
    num_outputs = len(output_coors)
    num_hiddens = len(hidden_coors)

    # Correct slicing for FullSubstrate:
    input_keys  = all_node_keys[:num_inputs]
    output_keys = all_node_keys[num_inputs : num_inputs + num_outputs]
    hidden_keys = all_node_keys[num_inputs + num_outputs : num_inputs + num_outputs + num_hiddens]

    # Add nodes to the graph with subsets (partitions) for visualization
    G_to_draw = nx.DiGraph()

    # Inputs at layer 0
    for k in input_keys:
        G_to_draw.add_node(k, subset=0)

    # Hidden layers (1..hidden_depth) — we map the *contiguous* hidden range to layers
    start_hidden = num_inputs + num_outputs

    # If each hidden layer has the same width (classic case):
    # hidden_width_full = len(input_coors)  # or len(hidden_coors)//hidden_depth
    # But we will use the robust per-layer widths we computed above:
    cum = 0
    for j, w in enumerate(hidden_widths):
        layer_id = j + 1
        start = start_hidden + cum
        end   = start + w
        for i in range(start, min(end, len(all_node_keys))):
            G_to_draw.add_node(all_node_keys[i], subset=layer_id)
        cum += w

    # Outputs at the final layer (after all hidden layers)
    output_layer_id = hidden_depth + 1
    for k in output_keys:
        G_to_draw.add_node(k, subset=output_layer_id)


    # Layout from the detailed layer assignment
    pos = nx.multipartite_layout(G_to_draw, subset_key='subset')

    # 4) Fixed-bounds grayscale mapping & robust edge extraction

    # Helper: coerce bounds to floats (in case 0,0 was typed instead of 0.0)
    def _to_float_bound(x, name):
        if isinstance(x, (tuple, list, np.ndarray)):
            if len(x) == 0:
                raise ValueError(f"{name} is empty; set a valid float (e.g., 0.0).")
            x = x[0]
        try:
            return float(x)
        except Exception as e:
            raise ValueError(
                f"Could not convert {name}={x!r} to float. "
                f"Use a scalar like 0.0 or 1.0. Original error: {e}"
            )

    LOWER = _to_float_bound(-1, "WEIGHT_LOWER_BOUND")
    UPPER = _to_float_bound(1, "WEIGHT_UPPER_BOUND")

    # Your active_conns rows look like [src, dst, extra]; take first two columns
    ac = np.asarray(active_conns)
    if ac.ndim != 2 or ac.shape[1] < 2:
        raise ValueError(f"Expected active_conns to have at least 2 columns; got shape {ac.shape}")

    all_edges = [(int(row[0]), int(row[1])) for row in ac]
    all_weights = np.asarray(active_weights)

    # Create lists to hold the edges and weights that are not self-loops (those are filtered out, because they "bloat" the plot)
    edges_to_add = []
    active_weights_filtered = []
    for edge, weight in zip(all_edges, all_weights):
        if edge[0] != edge[1]:  # This condition checks if the edge is NOT a self-loop
            edges_to_add.append(edge)
            active_weights_filtered.append(weight)

    # Convert back to a NumPy array for consistency
    active_weights = np.array(active_weights_filtered)

    print(f"Visualizing {len(active_weights)} connections. Excluded loops. Weight threshold: {internal_weight_threshold}")

    # Add edges to graph
    G_to_draw.add_edges_from(edges_to_add)

    # Magnitudes for color mapping (must align 1:1 with edges_to_add)
    abs_w = np.abs(active_weights)
    if len(abs_w) != len(edges_to_add):
        raise ValueError(
            f"Edge/weight mismatch: {len(edges_to_add)} edges vs {len(abs_w)} weights. "
            "Ensure any filtering is applied identically to connections and weights."
        )

    # Node colors: inputs=blue, outputs=red, hidden=green
    node_colors = []
    for node_key in G_to_draw.nodes():
        if node_key in input_keys:
            color = 'blue'
        elif node_key in output_keys:
            color = 'red'
        else:
            color = 'green'
        node_colors.append(color)

    # 5) Draw with separate colormaps for positive (Greys) and negative (Reds)

    fig, ax = plt.subplots(figsize=(12, 12))

    weights = np.asarray(active_weights)
    idx_all = np.arange(len(edges_to_add))

    pos_idx = idx_all[weights > 0]
    neg_idx = idx_all[weights < 0]
    zero_idx = idx_all[weights == 0]  # optional

    edges_pos = [edges_to_add[i] for i in pos_idx]
    edges_neg = [edges_to_add[i] for i in neg_idx]
    w_pos = weights[pos_idx]                # > 0
    w_neg_mag = -weights[neg_idx]           # positive magnitudes for negative edges

    # Edge widths scaled per side using fixed bounds
    eps = np.finfo(float).eps  # protect against division by zero

    widths_pos = 0.5 + 1.5 * np.clip(w_pos / max(UPPER, eps), 0.0, 1.0) if len(w_pos) else []
    widths_neg = 0.5 + 1.5 * np.clip(w_neg_mag / max(-LOWER, eps), 0.0, 1.0) if len(w_neg_mag) else []

    # Draw nodes once
    nx.draw_networkx_nodes(
        G_to_draw,
        pos=pos,
        node_color=node_colors,
        node_size=20,
        ax=ax
    )

    # Draw POSITIVE edges: Greys (white → black), mapped over [0, UPPER]
    if len(edges_pos):
        nx.draw_networkx_edges(
            G_to_draw,
            pos=pos,
            edgelist=edges_pos,
            edge_color=w_pos,             # raw positive weights
            edge_cmap=plt.cm.Greys,
            edge_vmin=0.0,
            edge_vmax=float(UPPER),
            width=widths_pos,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=4,
            ax=ax
        )

    # Draw NEGATIVE edges: Reds (white → red), mapped over [0, |LOWER|] using magnitudes
    if len(edges_neg):
        nx.draw_networkx_edges(
            G_to_draw,
            pos=pos,
            edgelist=edges_neg,
            edge_color=w_neg_mag,         # magnitudes of negative weights
            edge_cmap=plt.cm.Reds,
            edge_vmin=0.0,
            edge_vmax=float(-LOWER),
            width=widths_neg,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=4,
            ax=ax
        )

    # Colorbars
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax.set_title(f"Substrate Network — Positives Greys, Negatives Reds (Bounds [{LOWER}, {UPPER}])")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18) # Manually make space at the bottom for colorbars

    # Left: negative (Reds)
    if len(w_neg_mag):
        sm_neg = ScalarMappable(cmap=plt.cm.Reds,
                                norm=Normalize(vmin=0.0, vmax=float(-LOWER)))
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
        cbar_neg.set_ticks([0, (-LOWER)/2, -LOWER])

    # Right: positive (Greys)
    if len(w_pos):
        sm_pos = ScalarMappable(cmap=plt.cm.Greys,
                                norm=Normalize(vmin=0.0, vmax=float(UPPER)))
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
        cbar_pos.set_ticks([0, UPPER/2, UPPER])

    out_path = f"{output_dir}/ANN.svg"
    fig.savefig(out_path, dpi=800)
    plt.show()
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