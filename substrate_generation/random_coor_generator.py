import numpy as np

class RandomCoordinateGenerator:
    """
    Generates random coordinates for a substrate, ensuring uniqueness.
    It adaptively expands the coordinate value space from binary to ternary to
    continuous if the number of required unique coordinates is too large for
    the given number of feature dimensions.
    """
    def __init__(self, obs_size, act_size, feature_dims, depth_factor, width_factor=1.0, seed=None):
        self.obs_size = obs_size
        self.act_size = act_size
        self.feature_dims = feature_dims
        self.depth_factor = depth_factor # This is equivalent to output_depth
        self.width_factor = width_factor
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def _generate_unique_feature_coords(self, num_coords: int, num_feature_dims: int) -> list:
        """
        Helper function that generates a set of unique random coordinates,
        automatically selecting the alphabet ({0,1}, {0,0.5,1}, or continuous)
        based on the required capacity.
        """
        binary_capacity = 2**num_feature_dims
        ternary_capacity = 3**num_feature_dims
        
        if num_coords <= binary_capacity:
            print(f"Using binary alphabet [0, 1] for {num_coords} coordinates.")
            alphabet = [0., 1.]
            generation_mode = 'discrete'
        elif num_coords <= ternary_capacity:
            print(f"Warning: Binary space (cap: {binary_capacity}) is too small for {num_coords} coords. "
                  f"Expanding to ternary alphabet [0, 0.5, 1].")
            alphabet = [0., 0.5, 1.]
            generation_mode = 'discrete'
        else:
            print(f"Warning: Ternary space (cap: {ternary_capacity}) is also too small for {num_coords} coords. "
                  f"Falling back to continuous random coordinates in [0, 1).")
            generation_mode = 'continuous'

        unique_coords = set()
        # Add a safety break to prevent rare infinite loops in continuous mode
        max_attempts = num_coords * 200 
        attempts = 0

        while len(unique_coords) < num_coords and attempts < max_attempts:
            if generation_mode == 'discrete':
                # Generate a random vector by choosing from the selected alphabet
                random_vector = self.rng.choice(alphabet, size=num_feature_dims)
            else: # generation_mode == 'continuous'
                # Generate a random vector from a uniform distribution
                random_vector = self.rng.random(size=num_feature_dims)
                # Round to a few decimal places to avoid floating point precision issues
                random_vector = np.round(random_vector, 5)

            unique_coords.add(tuple(random_vector))
            attempts += 1
        
        if len(unique_coords) < num_coords:
            raise RuntimeError(f"Failed to generate {num_coords} unique coordinates after {max_attempts} attempts. "
                               "Consider increasing the number of feature dimensions.")
            
        return list(unique_coords)

    def generate_io_coordinates(self):
        """
        Generates random input and output coordinates and augments them with a
        layering dimension.
        """
        print(f"Generating random coordinates with {self.feature_dims} feature dimensions...")
        
        # Generate unique random coordinates using the new robust helper
        input_feature_coors_list = self._generate_unique_feature_coords(self.obs_size, self.feature_dims)
        output_feature_coors_list = self._generate_unique_feature_coords(self.act_size, self.feature_dims)

        # Convert to NumPy arrays for easier manipulation
        input_feature_coors = np.array(input_feature_coors_list, dtype=float)
        output_feature_coors = np.array(output_feature_coors_list, dtype=float)

        # Augment coordinates with the layering dimension
        input_layer_dim = np.full((input_feature_coors.shape[0], 1), -self.depth_factor)
        output_layer_dim = np.full((output_feature_coors.shape[0], 1), self.depth_factor)
        
        input_coors_full = np.hstack([input_feature_coors, input_layer_dim])
        output_coors_full = np.hstack([output_feature_coors, output_layer_dim])

        # Apply the width factor
        if self.width_factor != 1.0:
            print(f"Applying width factor: {self.width_factor}")
            input_coors_full[:, :-1] *= self.width_factor
            output_coors_full[:, :-1] *= self.width_factor

        # The total coordinate size is feature_dims + 1
        final_coord_size = self.feature_dims + 1
        print(f"Added layering dimension. Final coordinate size: {final_coord_size}")

        # Convert to list of tuples and add the bias node
        input_coors_list = [tuple(row) for row in input_coors_full]
        # input_coors_list.append(tuple([0.0] * self.feature_dims + [-self.depth_factor])) # normalization -1 to 1
        input_coors_list.append(tuple([0.0] * final_coord_size)) # normalization 0 to 1
        
        output_coors_list = [tuple(row) for row in output_coors_full]
        
        # Return coord_size for a consistent interface with other analyzers
        return input_coors_list, output_coors_list