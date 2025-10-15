import numpy as np

def process_coordinates(
        all_feature_coors, 
        normalize_coors, 
        width_factor, 
        obs_size, 
        depth_factor, 
        feature_dims
    ):
    if normalize_coors:
        normalized_coors = np.copy(all_feature_coors)
        print("Normalizing coordinates...")
        # Small epsilon to prevent division by zero
        eps = 1e-9

        # Iterate over each dimension (each column)
        for i in range(normalized_coors.shape[1]):
            column = normalized_coors[:, i]
            
            # Find the maximum positive and absolute minimum negative values
            max_pos = np.max(column[column > 0], initial=eps)
            min_neg = np.min(column[column < 0], initial=-eps)
            
            # Normalize positive and negative values independently (to preserve the zero-point)
            column[column > 0] /= max_pos
            column[column < 0] /= np.abs(min_neg) # Divide by absolute value to scale into [-1, 0]
            
            normalized_coors[:, i] = column

        all_feature_coors = normalized_coors

    input_feature_coors = all_feature_coors[:obs_size]
    output_feature_coors = all_feature_coors[obs_size:]

    # Augment coordinates with the layering dimension
    # Create a column of -1 for the input layer
    # input_layer_dim = np.full((input_feature_coors.shape[0], 1), -depth_factor) # normalization -1 to 1
    input_layer_dim = np.zeros((input_feature_coors.shape[0], 1)) # normalization 0 to 1
    # Create a column with depth value for output layer
    output_layer_dim = np.full((output_feature_coors.shape[0], 1), depth_factor)

    # Horizontally stack the feature coordinates with the new layer dimension
    input_coors_full = np.hstack([input_feature_coors, input_layer_dim])
    output_coors_full = np.hstack([output_feature_coors, output_layer_dim])

    if width_factor != 1.0:
        print(f"Applying width factor: {width_factor}")
        input_coors_full[:, :-1] *= width_factor
        output_coors_full[:, :-1] *= width_factor

    # The total coordinate size is now feature dimensions + 1 for layering direction
    final_coord_size = feature_dims + 1
    print(f"Added layering dimension. Final coordinate size: {final_coord_size}")

    # Convert to list of tuples for compatibility
    input_coors_list = [tuple(row) for row in input_coors_full]
    # input_coors_list.append(tuple([0.0] * feature_dims + [-depth_factor])) # bias node for normalization -1 to 1
    input_coors_list.append(tuple([0.0] * final_coord_size)) # bias node for normalization 0 to 1

    output_coors_list = [tuple(row) for row in output_coors_full]

    return input_coors_list, output_coors_list