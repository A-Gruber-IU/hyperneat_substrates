class HiddenLayerGenerator:
    """
    Handles the generation of substrate coordinates for hidden layers.
    """
    def __init__(self, env_name, obs_size, act_size, hidden_layer_type, hidden_depth, depth_factor, width_factor):
        self.env_name = env_name
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_layer_type = hidden_layer_type
        self.hidden_depth = hidden_depth
        self.depth_factor = depth_factor
        self.width_factor = width_factor
    
    def get_hidden_coors(self, input_coors):
        hidden_coors = []
        coord_size = len(input_coors[0])
        if coord_size <= 1:
            return []
        num_features = coord_size - 1
        non_input_layers = self.hidden_depth + 1

        for i in range(self.hidden_depth):
            
            shift = (i + 1)*self.depth_factor
            if self.hidden_layer_type == "shift":
                for coor in input_coors:
                    new_coor = list(coor)
                    new_coor[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(new_coor))

            elif self.hidden_layer_type == "shift_two":
                for coor in input_coors:
                    new_coor = list(coor)
                    new_coor[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(new_coor))
                for coor in input_coors:
                    new_coor = list(coor)
                    for i in range(len(new_coor) - 1):
                        new_coor[i] += self.width_factor
                    new_coor[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(new_coor))

            elif self.hidden_layer_type == "shift_three":
                for coor in input_coors:
                    new_coor = list(coor)
                    new_coor[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(new_coor))
                for coor in input_coors:
                    new_coor = list(coor)
                    for i in range(len(new_coor) - 1):
                        new_coor[i] += self.width_factor
                    new_coor[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(new_coor))
                for coor in input_coors:
                    new_coor = list(coor)
                    for i in range(len(new_coor) - 1):
                        new_coor[i] += self.width_factor*2
                    new_coor[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(new_coor))

            elif self.hidden_layer_type == "one_double_hot":
                for j in range(num_features):
                    row_1 = [0.0] * coord_size
                    row_1[j] = self.width_factor
                    row_1[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(row_1))
                    row_2 = [0.0] * coord_size
                    row_2[j] = 2.0 * self.width_factor
                    row_2[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(row_2))

            elif self.hidden_layer_type == "two_hot":
                for j in range(num_features):
                    row = [0.0] * coord_size
                    row[j] = self.width_factor
                    row[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(row))
                for j in range(num_features):
                    for k in range(j + 1, num_features):
                        row = [0.0] * coord_size
                        row[j] = self.width_factor
                        row[k] = self.width_factor
                        row[-1] = shift/non_input_layers
                        hidden_coors.append(tuple(row))
                        
            elif self.hidden_layer_type == "one_hot":
                for j in range(num_features):
                    row = [0.0] * coord_size
                    row[j] = self.width_factor
                    row[-1] = shift/non_input_layers
                    hidden_coors.append(tuple(row))

        return hidden_coors
