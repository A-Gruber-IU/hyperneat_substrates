import numpy as np

class ManualIOMapper:

    @staticmethod
    def get_all_manual_mappings(depth_factor):
        manual_mappings = {
            "ant": {
                "input": { 
                    0: [(1.0, [1, 2, 3, 4])], # orientation
                    1: [(1.0, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])], # velocity / angular velocity
                    2: [(1.0, [5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])], # angle / angular velocity
                    3: [(1.0, [2, 13, 16])], # x direction (orientation / position)
                    4: [(1.0, [3, 14, 17])], # y direction (orientation / position)
                    5: [(1.0, [0, 4, 15, 18])], # z direction (orientation / position)
                    6: [(1.0, [7, 8, 11, 12, 21, 22, 25, 26]), # right side
                        (-1.0, [5, 6, 9, 10, 19, 20, 23, 24])], # left side
                    7: [(1.0, [5, 6, 7, 8, 19, 20, 21, 22]), # front side
                        (-1.0, [9, 10, 11, 12, 23, 24, 25, 26])], # back side
                    8: [(1.0, [5, 7, 9, 11, 19, 21, 23, 25]), # hip joints
                        (-1.0, [6, 8, 10, 12, 20, 22, 24, 26])], # ankle joints
                    # 9: layering direction / torque
                    # not specificially included for torso: [(1.0, [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18])],
                },
                "output": [
                    (0, 0, 0, 0, 0, 0, 1, -1, 1, depth_factor), # torque back right hip
                    (0, 0, 0, 0, 0, 0, 1, -1, -1, depth_factor), # torque back right ankle 
                    (0, 0, 0, 0, 0, 0, -1, 1, 1, depth_factor), # torque front left hip 
                    (0, 0, 0, 0, 0, 0, -1, 1, -1, depth_factor), # torque front left ankle 
                    (0, 0, 0, 0, 0, 0, 1, 1, 1, depth_factor), # torque front right hip
                    (0, 0, 0, 0, 0, 0, 1, 1, -1, depth_factor), # torque front right ankle
                    (0, 0, 0, 0, 0, 0, -1, -1, 1, depth_factor), # torque back left hip 
                    (0, 0, 0, 0, 0, 0, -1, -1, -1, depth_factor), # torque back left ankle  
                ]},
            "halfcheetah": {
                "input": {
                    0: [(1.0,  [6,7,8,15,16,17]), # front leg components
                        (0.0,  [0,1,2,9,10,11]), # root
                        (-1.0, [3,4,5,12,13,14])], # back leg components
                    1: [(1.0,  [3,6,12,15]), # thigh
                        (0.0,  [4,7,13,16]), # shin
                        (-1.0, [5,8,14,17])], # foot
                    2: [(1.0,  [2,3,4,5,6,7,8]), # angle
                        (-1.0, [11,12,13,14,15,16,17])], # angular velocity
                    3: [(1.0, [9]), # velocity x
                        (0.0, [10])],  # velocity z
                    4: [(1.0, [0]), # position x
                        (0.0, [1])],  # position z
                    # 5: layering direction
                },
                "output": [
                    (-1, 1, 0, 0, 0, depth_factor), # torque back thigh rotor 
                    (-1, 0, 0, 0, 0, depth_factor), # torque back shin rotor 
                    (-1, -1, 0, 0, 0, depth_factor), # torque back foot rotor  
                    (1, 1, 0, 0, 0, depth_factor), # torque front thigh rotor  
                    (1, 0, 0, 0, 0, depth_factor), # torque front shin rotor 
                    (1, -1, 0, 0, 0, depth_factor), # torque front foor rotor  
                ]},
            "swimmer":{
                "input": {
                    0: [(1.0, [1, 6]), # first rotor
                        (-1.0, [2, 7])], # second rotor
                    1: [(1.0, [0, 3, 4, 5])], # front tip
                    2: [(1.0, [3])], # velocity x
                    3: [(1.0, [4])], # velocity y
                    4: [(1.0, [0, 1, 2])], # angle
                    5: [(1.0, [5, 6, 7])], # angular velocity
                    # 6: layering direction
                },
                "output": [
                    (1, 0, 0, 0, 0, 0, depth_factor), # torque first rotor
                    (-1, 0, 0, 0, 0, 0, depth_factor), # torque second rotor
                ]},
        }
        return manual_mappings
    
    def __init__(self, 
                 env_name, 
                 obs_size, 
                 act_size, 
                 hidden_layer_type, 
                 width_factor,
                 depth_factor
                ):
        self.env_name = env_name
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_layer_type = hidden_layer_type
        self.width_factor = width_factor # make substrate potentially equally wide and deep
        self.depth_factor = depth_factor
        all_manual_mappings = self.get_all_manual_mappings(self.depth_factor)
        self.mapping = all_manual_mappings.get(self.env_name, {})
        self.dim_mapping = self.mapping.get("input")

        # Fallback if no specific mapping exists for the environment
        if not self.dim_mapping:
            print(f"Warning: No input mapping for '{self.env_name}'. Using a generic one.")
            self.dim_mapping = {0: [(1.0, list(range(self.obs_size)))]}
        else:
            print(f"Using user-defined input and output mapping for '{self.env_name}'.")

        # Calculate coordinate dimensions
        self.base_feature_width = max(self.dim_mapping.keys()) + 1
        print(f"Number of feature dimensions: {self.base_feature_width}")
        
        self.coord_size = self.base_feature_width + 1 # Features + Layering Dimension
        print(f"Total number after adding output and layering dimensions (coord_size): {self.coord_size}")

    def generate_io_coordinates(self):
        input_coors = self._generate_obs_coordinates()
        output_coors = self._get_output_coors(self.coord_size)

        if self.width_factor != 1.0:
            print(f"Applying width factor: {self.width_factor}")
            # Convert list of tuples to NumPy array
            input_coors_np = np.array(input_coors, dtype=float)
            output_coors_np = np.array(output_coors, dtype=float)
            # Apply scaling to all dimensions except the last one
            input_coors_np[:, :-1] *= self.width_factor
            output_coors_np[:, :-1] *= self.width_factor
            # Convert back to a list of tuples
            input_coors = [tuple(row) for row in input_coors_np]
            output_coors = [tuple(row) for row in output_coors_np]
        
        input_coors.append(tuple([0.0] * (self.base_feature_width) + [-self.depth_factor])) # bias input for normalization -1 to 1
        # input_coors.append(tuple([0.0] * self.coord_size)) # bias input for normalization 0 to 1
        print(f"Number of input nodes (obs + bias): {len(input_coors)}")

        return input_coors, output_coors, self.base_feature_width

    def _generate_obs_coordinates(self):
        obs_coors = []
        for node_index in range(self.obs_size):
            coord = [0.0] * self.coord_size
            # coord[-1] = -self.depth_factor # normalization -1 to 1 (remove for normalization 0 to 1)
            for dim_idx, concepts in self.dim_mapping.items():
                for value, nodes in concepts:
                    if node_index in nodes:
                        coord[dim_idx] = value
                        break
            obs_coors.append(tuple(coord))
        return obs_coors
    
    def _get_output_coors(self, coord_size):
        # Try to get the specific output mapping first
        output_coors = self.mapping.get("output")
        if output_coors:
            # Validate that the user-defined coordinates have the correct width
            for i, c in enumerate(output_coors):
                if len(c) != coord_size:
                    raise ValueError(f"Output coordinate at index {i} for '{self.env_name}' has width {len(c)}, but calculated coord_size is {coord_size}. Please check the mapping.")
            print(f"Number of output nodes: {len(output_coors)}")
            return output_coors
        
        # Fallback to generic output coordinates if none are defined
        print(f"Warning: No output mapping for '{self.env_name}'. Using a generic one.")
        output_coors = []
        feature_dims = coord_size - 1
        for i in range(self.act_size):
            coord = tuple([0] * feature_dims + [i + 1] + [self.depth_factor])
            output_coors.append(coord)
        return output_coors
