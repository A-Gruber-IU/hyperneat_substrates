
class HiddenLayerGenerator:
    """
    Handles the generation of substrate coordinates for hidden layers.
    """
    def __init__(self, env_name, obs_size, act_size, hidden_layer_type, hidden_depth):
        self.env_name = env_name
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_layer_type = hidden_layer_type
        self.hidden_depth = hidden_depth
    
    def get_hidden_coors(self, input_coors):
        hidden_coors = []
        coord_size = len(input_coors[0])
        if coord_size <= 1:
            return []
        num_features = coord_size - 1

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
