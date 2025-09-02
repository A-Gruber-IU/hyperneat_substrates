class SubstrateGenerator:

    @staticmethod
    def get_all_mappings(hidden_depth):
        mappings = {
            "Ant-v5": {
                "input0": { # first draft, not including full obs space
                    0: [(1.0, [1, 2, 3, 4])], # orientation
                    1: [(1.0, [13, 14, 15])], # velocity
                    2: [(1.0, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])], # angular velocity
                    3: [(1.0, [5, 6, 7, 8, 9, 10, 11, 12])], # angle
                    4: [(1.0, [2, 13, 16])], # x
                    5: [(1.0, [3, 14, 17])], # y
                    6: [(1.0, [0, 4, 15, 18])], # z
                    7: [(1.0, [7, 8, 11, 12, 21, 22, 25, 26]), # right side
                        (-1.0, [5, 6, 9, 10, 19, 20, 23, 24])], # left side
                    8: [(1.0, [5, 6, 7, 8, 19, 20, 21, 22]), # front side
                        (-1.0, [9, 10, 11, 12, 23, 24, 25, 26])], # back side
                    9: [(1.0, [5, 7, 9, 11, 19, 21, 23, 25]), # hip joints
                        (-1.0, [6, 8, 10, 12, 20, 22, 24, 26])], # ankle joints
                    # 10: time signal
                    # 11: torque
                    # 12: layering direction
                    # not specificially included for torso: [(1.0, [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18])],
                },
                "input": {
                    0: [(1.0, [1, 2, 3, 4])], # orientation
                    1: [(1.0, [13, 14, 15])], # velocity
                    2: [(1.0, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])], # angular velocity
                    3: [(1.0, [5, 6, 7, 8, 9, 10, 11, 12])], # angle
                    4: [(1.0, [2, 13, 16])], # x
                    5: [(1.0, [3, 14, 17])], # y
                    6: [(1.0, [0, 4, 15, 18])], # z
                    7: [(1.0, [7, 8, 11, 12, 21, 22, 25, 26, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104]), # right side
                        (-1.0, [5, 6, 9, 10, 19, 20, 23, 24, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])], # left side
                    8: [(1.0, [5, 6, 7, 8, 19, 20, 21, 22, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]), # front side
                        (-1.0, [9, 10, 11, 12, 23, 24, 25, 26, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])], # back side
                    9: [(1.0, [5, 7, 9, 11, 19, 21, 23, 25, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]), # hip joints
                        (-1.0, [6, 8, 10, 12, 20, 22, 24, 26, 45, 46, 47, 48, 49, 50, 63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 99, 100, 101, 102, 103, 104])], # ankle joints
                    10: [(1.0, [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104])]}, # contact force
                    # 11: time signal
                    # 12: torque
                    # 13: layering direction
                "output": [
                    (0, 0, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0, 1, hidden_depth+1), # torque back right hip
                    (0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, 0, 1, hidden_depth+1), # torque back right ankle 
                    (0, 0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, 1, hidden_depth+1), # torque front left hip 
                    (0, 0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0, 1, hidden_depth+1), # torque front left ankle 
                    (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, hidden_depth+1), # torque front right hip
                    (0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 1, hidden_depth+1), # torque front right ankle
                    (0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 1, hidden_depth+1), # torque back left hip 
                    (0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 1, hidden_depth+1), # torque back left ankle  
                ]},
            "HalfCheetah-v5": {
                "input": {
                    0: [(1.0,  [5, 6, 7, 14, 15, 16]), # front leg components
                        (-1.0, [2, 3, 4, 11, 12, 13])], # back leg components
                    1: [(1.0,  [2, 5, 11, 14]), # thigh
                        (0.0,  [3, 6, 12, 15]), # shin
                        (-1.0, [4, 7, 13, 16])], # foot
                    2: [(1.0,  [1, 2, 3, 4, 5, 6, 7]), # angle
                        (-1.0, [10, 11, 12, 13, 14, 15, 16])], # angular velocity
                    3: [(1.0,  [1, 0, 8, 9, 10])], # front tip
                    4: [(1.0, [8])], # velocity x
                    5: [(1.0, [9])], # velocity z
                    # 6: time signal
                    # 7: torque
                    # 8: layering direction
                },
                "output": [
                    (-1, 1, 0, 0, 0, 0, 0, 1, hidden_depth+1), # torque back thigh rotor 
                    (-1, 0, 0, 0, 0, 0, 0, 1, hidden_depth+1), # torque back shin rotor 
                    (-1, -1, 0, 0, 0, 0, 0, 1, hidden_depth+1), # torque back foot rotor  
                    (1, 1, 0, 0, 0, 0, 0, 1, hidden_depth+1), # torque front thigh rotor  
                    (1, 0, 0, 0, 0, 0, 0, 1, hidden_depth+1), # torque front shin rotor 
                    (1, -1, 0, 0, 0, 0, 0, 1, hidden_depth+1), # torque front foor rotor  
                ]},
            "BipedalWalker-v3": {
                "input": {
                    0: [(1.0,  [4, 5, 6, 7, 8]), # leg 1
                        (-1.0, [9, 10, 11, 12, 13])], # leg 2
                    1: [(1.0, [4, 5, 9, 10]), # hip
                        (-1.0, [6, 7, 11, 12])], # knee
                    2: [(1.0, [0, 4, 6, 9, 11])], # angle
                    3: [(1.0, [1, 5, 7, 10, 12])],  # angular velocity
                    4: [(1.0, [0, 1])], # hull
                    5: [(1.0, [2]), # velocity x
                        (-1.0, [3])], # velocity y
                    6: [(1.0, [8, 13])], # ground contact
                    # 7: time signal
                    # 8: motor speed
                    # 9: layering direction
                },
                "output": [
                    (1, 1, 0, 0, 0, 0, 0, 0, 1, hidden_depth+1), # motor speed hip leg 1
                    (1, -1, 0, 0, 0, 0, 0, 0, 1, hidden_depth+1), # motor speed knee leg 1
                    (-1, 1, 0, 0, 0, 0, 0, 0, 1, hidden_depth+1), # motor speed hip leg 2
                    (-1, -1, 0, 0, 0, 0, 0, 0, 1, hidden_depth+1), # motor speed knee leg 2
                ]},
            "Swimmer-v5":{
                "input": {
                    0: [(1.0, [1, 6]), # first rotor
                        (-1.0, [2, 7])], # second rotor
                    1: [(1.0, [0, 3, 4, 5])], # front tip
                    2: [(1.0, [3])], # velocity x
                    3: [(1.0, [4])], # velocity y
                    4: [(1.0, [0, 1, 2])], # angle
                    5: [(1.0, [5, 6, 7])], # angular velocity
                    # 6: time signal
                    # 7: torque
                    # 8: layering direction
                },
                "output": [
                    (1, 0, 0, 0, 0, 0, 0, 1, hidden_depth+1), # torque first rotor
                    (-1, 0, 0, 0, 0, 0, 0, 1, hidden_depth+1), # torque second rotor
                ]},
        }
        return mappings
    
    def __init__(self, env_name, obs_size, act_size, hidden_layer_type, hidden_depth,
                 wavelengths, cosine_wave):
        self.env_name = env_name
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_layer_type = hidden_layer_type
        self.hidden_depth = hidden_depth
        self.wavelengths = wavelengths
        self.cosine_wave = cosine_wave

        all_mappings = self.get_all_mappings(self.hidden_depth)
        self.mapping = all_mappings.get(self.env_name, {})
        self.dim_mapping = self.mapping.get("input")

        # Fallback if no specific mapping exists for the environment
        if not self.dim_mapping:
            print(f"Warning: No input mapping for '{self.env_name}'. Using a generic one.")
            self.dim_mapping = {0: [(1.0, list(range(self.obs_size)))]}

        # Calculate coordinate dimensions
        self.num_time_nodes = len(self.wavelengths) * (1 + int(self.cosine_wave))
        self.num_obs_nodes = self.obs_size - self.num_time_nodes
        self.base_feature_width = max(self.dim_mapping.keys()) + 1
        
        # The single time dimension is added after the base features
        self.time_dim_idx = self.base_feature_width
        self.coord_size = self.base_feature_width + 1 + 2 # Features + Time Dim + 2 Special Dims

    def get_input_coors(self):
        coords = self._generate_obs_coordinates()
        self._append_time_signal_nodes(coords)
        coords.append(tuple([0.0] * self.coord_size))
        return coords, self.coord_size

    def get_hidden_coors(self, input_coors, coord_size):
        hidden_coors = []
        num_features = coord_size - 2
        for i in range(self.hidden_depth):
            shift = i + 1
            if self.hidden_layer_type == "two_hot":
                # One-hot
                for j in range(num_features):
                    row = [0.0] * coord_size; row[j] = 1.0; row[-1] = shift; hidden_coors.append(tuple(row))
                # Two-hot
                for j in range(num_features):
                    for k in range(j + 1, num_features):
                        row = [0.0] * coord_size; row[j] = 1.0; row[k] = 1.0; row[-1] = shift; hidden_coors.append(tuple(row))
            elif self.hidden_layer_type == "shift":
                base_coords = input_coors[:-1] # Exclude bias
                for c in base_coords:
                    new_c = list(c); new_c[-1] = shift; hidden_coors.append(tuple(new_c))
            elif self.hidden_layer_type == "one_hot":
                for j in range(num_features):
                    row = [0.0] * coord_size; row[j] = 1.0; row[-1] = shift; hidden_coors.append(tuple(row))
            elif self.hidden_layer_type == "one_double_hot":
                for j in range(num_features):
                    row = [0.0] * coord_size; row[j] = 1.0; row[-1] = shift; hidden_coors.append(tuple(row))
                    row = [0.0] * coord_size; row[j] = 2.0; row[-1] = shift; hidden_coors.append(tuple(row))
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

    def _generate_obs_coordinates(self):
        coordinates = []
        for node_index in range(self.num_obs_nodes):
            coord = [0.0] * self.coord_size
            for dim_idx, concepts in self.dim_mapping.items():
                for value, nodes in concepts:
                    if node_index in nodes:
                        coord[dim_idx] = value
                        break
            coordinates.append(tuple(coord))
        return coordinates

    def _append_time_signal_nodes(self, coords):
        for i, _ in enumerate(self.wavelengths):
            row = [0.0] * self.coord_size
            row[self.time_dim_idx] = i + 1  # 1, 2, 3 for sine waves
            coords.append(tuple(row))

        if self.cosine_wave:
            for i, _ in enumerate(self.wavelengths):
                row = [0.0] * self.coord_size
                row[self.time_dim_idx] = -(i + 1) # -1, -2, -3 for cosine waves
                coords.append(tuple(row))