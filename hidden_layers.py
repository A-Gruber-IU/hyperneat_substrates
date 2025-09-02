
class SubstrateGenerator:
    """
    Handles the generation of substrate coordinates, particularly for hidden layers and output nodes.
    """
    def __init__(self, env_name, obs_size, act_size, hidden_layer_type, hidden_depth):
        self.env_name = env_name
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_layer_type = hidden_layer_type
        self.hidden_depth = hidden_depth
    
    def get_hidden_coors(self, input_coors, coord_size):
        hidden_coors = []
        num_features = coord_size - 2
        for i in range(self.hidden_depth):
            shift = i + 1
            if self.hidden_layer_type == "shift":
                base_coords = input_coors[:-1] # Exclude bias
                for c in base_coords:
                    new_c = list(c); new_c[-1] = shift; hidden_coors.append(tuple(new_c))
            elif self.hidden_layer_type == "one_double_hot":
                for j in range(num_features):
                    row = [0.0] * coord_size; row[j] = 1.0; row[-1] = shift; hidden_coors.append(tuple(row))
                    row = [0.0] * coord_size; row[j] = 2.0; row[-1] = shift; hidden_coors.append(tuple(row))
            if self.hidden_layer_type == "two_hot":
                for j in range(num_features):
                    row = [0.0] * coord_size; row[j] = 1.0; row[-1] = shift; hidden_coors.append(tuple(row))
                for j in range(num_features):
                    for k in range(j + 1, num_features):
                        row = [0.0] * coord_size; row[j] = 1.0; row[k] = 1.0; row[-1] = shift; hidden_coors.append(tuple(row))
            else: # self.hidden_layer_type == "one_hot" as default
                for j in range(num_features):
                    row = [0.0] * coord_size; row[j] = 1.0; row[-1] = shift; hidden_coors.append(tuple(row))
        return hidden_coors

    def get_output_coors(self, coord_size):
        # Try to get the specific output mapping first
        output_coords = self.mapping.get("output")
        if output_coords:
            # Validate that the user-defined coordinates have the correct width
            for i, c in enumerate(output_coords):
                if len(c) != coord_size:
                    raise ValueError(f"Output coordinate at index {i} for '{self.env_name}' has width {len(c)}, but calculated coord_size is {coord_size}. Please check the mapping.")
            return output_coords
        
        # Fallback to generic output coordinates if none are defined
        print(f"Warning: No output mapping for '{self.env_name}'. Using a generic one.")
        output_coors = []
        feature_dims = coord_size - 2
        for i in range(self.act_size):
            coord = tuple([0] * feature_dims + [i + 1] + [self.hidden_depth + 1])
            output_coors.append(coord)
        return output_coors
