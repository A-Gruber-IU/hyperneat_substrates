from typing import List, Sequence, Tuple, Optional, Literal
import numpy as np

from tensorneat.algorithm.hyperneat.substrate import DefaultSubstrate


class AutoLayeredCoordMLPSubstrate(DefaultSubstrate):
    """
    Feedforward HyperNEAT substrate built from:
      - input_coors: list of coordinates
      - hidden_coors: list of coordinates
      - output_coors: list of coordinates

    Hidden coordinates are grouped into layers by their last dimension. All
    coordinates with (approximately) the same last component belong to the same layer.

    The network is wired feedforward-only:
      - input -> first hidden layer (or directly -> output if no hidden)
      - hidden[k] -> hidden[k+1]
      - last hidden -> output
    Optionally, set allow_skip=True to connect every layer to all *later* layers.

    TensorNEAT's node ordering requirement is respected:
      - nodes array is [inputs..., outputs..., hidden...]
    """
    connection_type = "feedforward"

    def __init__(
        self,
        input_coors: Sequence[Sequence[float]],
        hidden_coors: Sequence[Sequence[float]],
        output_coors: Sequence[Sequence[float]],
        *,
        layer_tol: float = 1e-8,
        allow_skip: bool = False,
    ):
        """
        Args:
            input_coors:   list/array of input coordinates, shape (N_in, D)
            hidden_coors:  flat list/array of hidden coordinates, shape (N_h, D)
                           LAST DIMENSION encodes the layer id/value.
            output_coors:  list/array of output coordinates, shape (N_out, D)
            layer_tol:     tolerance to group last-dimension values into the same layer
                           (helps with floating-point noise).
            allow_skip:    if True, connect each layer to all subsequent layers
                           (input->all hidden & output; hidden_k->hidden_{>k} & output)
        """
        (query_coors, nodes, conns,
         num_inputs, num_outputs) = _analysis_from_flat_hidden(
            input_coors, hidden_coors, output_coors,
            layer_tol=layer_tol,
            allow_skip=allow_skip,
        )
        super().__init__(num_inputs, num_outputs, query_coors, nodes, conns)


def _analysis_from_flat_hidden(
    input_coors: Sequence[Sequence[float]],
    hidden_coors: Sequence[Sequence[float]],
    output_coors: Sequence[Sequence[float]],
    *,
    layer_tol: float,
    allow_skip: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    # Validate & normalize
    in_arr = np.asarray(input_coors, dtype=np.float32)
    out_arr = np.asarray(output_coors, dtype=np.float32)
    hid_arr = np.asarray(hidden_coors, dtype=np.float32)

    assert in_arr.ndim == 2 and out_arr.ndim == 2, "input_coors and output_coors must be 2D arrays"
    coord_dim = in_arr.shape[1]
    assert out_arr.shape[1] == coord_dim, "All coordinates must share the same dimensionality"
    if hid_arr.size > 0:
        assert hid_arr.ndim == 2 and hid_arr.shape[1] == coord_dim, "hidden_coors must match coordinate dimensionality"

    n_in = in_arr.shape[0]
    n_out = out_arr.shape[0]
    assert n_in >= 1 and n_out >= 1, "At least one input and one output required"

    # Group hidden coords by last dimension (within tolerance)
    hidden_layers_coors: List[np.ndarray] = []
    if hid_arr.size > 0:
        # sort by last dimension
        order = np.argsort(hid_arr[:, -1])
        hid_sorted = hid_arr[order]
        last_vals = hid_sorted[:, -1]

        # cluster consecutive values whose gaps <= layer_tol
        boundaries = [0]
        for i in range(1, len(last_vals)):
            if abs(float(last_vals[i] - last_vals[i-1])) > layer_tol:
                boundaries.append(i)
        boundaries.append(len(last_vals))

        # extract layers
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            layer = hid_sorted[s:e]
            hidden_layers_coors.append(layer)

    # Assign global indices in TensorNEAT order: inputs -> outputs -> hidden
    input_indices = list(range(n_in))
    output_indices = list(range(n_in, n_in + n_out))

    hidden_layers_indices: List[List[int]] = []
    start = n_in + n_out
    for layer in hidden_layers_coors:
        cnt = layer.shape[0]
        idxs = list(range(start, start + cnt))
        hidden_layers_indices.append(idxs)
        start += cnt

    # Build an index-addressable node_coors aligned with global indices
    total_nodes = n_in + n_out + sum(len(x) for x in hidden_layers_indices)
    node_coors: List[Tuple[float, ...]] = [None] * total_nodes  # type: ignore

    # inputs
    for local_i, gid in enumerate(input_indices):
        node_coors[gid] = tuple(in_arr[local_i])
    # outputs
    for local_i, gid in enumerate(output_indices):
        node_coors[gid] = tuple(out_arr[local_i])
    # hidden
    for layer_idxs, layer_coors in zip(hidden_layers_indices, hidden_layers_coors):
        for local_i, gid in enumerate(layer_idxs):
            node_coors[gid] = tuple(layer_coors[local_i])

    # Build feedforward connections
    # topological view (for wiring): [inputs], hidden_layers..., [outputs]
    topo_indices = [input_indices, *hidden_layers_indices, output_indices]
    topo_coords  = [in_arr, *hidden_layers_coors, out_arr]

    query_blocks = []
    key_blocks = []

    def connect_layers(src_ids, src_xy, tgt_ids, tgt_xy):
        src_ids = np.asarray(src_ids, dtype=np.int32)
        tgt_ids = np.asarray(tgt_ids, dtype=np.int32)
        src_xy  = np.asarray(src_xy,  dtype=np.float32)
        tgt_xy  = np.asarray(tgt_xy,  dtype=np.float32)

        if len(src_ids) == 0 or len(tgt_ids) == 0:
            return None, None
        s_rep = np.repeat(src_xy, repeats=len(tgt_ids), axis=0)
        t_tile = np.tile(tgt_xy, (len(src_ids), 1))
        q = np.concatenate([s_rep, t_tile], axis=1)  # (|S|*|T|, 2*D)

        s_ids = np.repeat(src_ids, repeats=len(tgt_ids))
        t_ids = np.tile(tgt_ids, len(src_ids))
        k = np.column_stack([s_ids, t_ids]).astype(np.int32)
        return q, k
    
    L = len(topo_indices)
    if L == 2:
        # No hidden layers: input -> output only
        q, k = connect_layers(topo_indices[0], topo_coords[0], topo_indices[1], topo_coords[1])
        if q is not None:
            query_blocks.append(q)
            key_blocks.append(k)
    else:
        # With hidden layers
        for k_idx in range(L - 1):
            tgt_range = range(k_idx + 1, L) if allow_skip else range(k_idx + 1, k_idx + 2)
            for t_idx in tgt_range:
                q, k = connect_layers(topo_indices[k_idx], topo_coords[k_idx],
                                      topo_indices[t_idx], topo_coords[t_idx])
                if q is not None:
                    query_blocks.append(q)
                    key_blocks.append(k)

    query_coors = (np.vstack(query_blocks).astype(np.float32)
                   if query_blocks else np.zeros((0, 2 * coord_dim), dtype=np.float32))
    correspond_keys = (np.vstack(key_blocks).astype(np.int32)
                       if key_blocks else np.zeros((0, 2), dtype=np.int32))

    # Pack outputs for TensorNEAT
    ordered_nodes = [*input_indices, *output_indices]
    for hl in hidden_layers_indices:
        ordered_nodes.extend(hl)
    nodes = np.array(ordered_nodes, dtype=np.int32)[:, np.newaxis]

    conns = np.zeros((len(correspond_keys), 3), dtype=np.float32)  # [src, dst, weight]
    if len(correspond_keys) > 0:
        conns[:, :2] = correspond_keys

    return query_coors, nodes, conns, n_in, n_out
