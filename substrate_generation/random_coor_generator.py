import numpy as np

class RandomCoordinateGenerator:
    """
    Generates random binary coordinates for a substrate, ensuring uniqueness within
    the input and output layers. Serves as a baseline for comparison against
    analysis-driven coordinate generation methods.
    """
    def __init__(self, obs_size, act_size, max_dims, depth_factor, width_factor=1.0, seed=None):
        self.obs_size = obs_size
        self.act_size = act_size
        self.feature_dims = max_dims
        self.depth_factor = depth_factor
        self.width_factor = width_factor
        self.seed = seed
        self.rng = np.random.default_rng(self.seed) # Create a NumPy random number generator

    def _generate_unique_binary_coords(self, num_coords: int, num_feature_dims: int) -> list:
        """
        Helper function to generate a set of unique random binary coordinates.
        """
        # Safety check: Is it possible to generate this many unique coordinates?
        max_possible_unique = 2**num_feature_dims
        if num_coords > max_possible_unique:
            raise ValueError(
                f"Cannot generate {num_coords} unique coordinates with only {num_feature_dims} binary dimensions. "
                f"Maximum possible is {max_possible_unique}."
            )
        
        unique_coords = set()
        while len(unique_coords) < num_coords:
            # Generate a random vector of 0s and 1s
            random_vector = self.rng.integers(0, 2, size=num_feature_dims)
            # Add the tuple version to the set (sets require hashable types like tuples)
            unique_coords.add(tuple(random_vector))
            
        return list(unique_coords)

    def generate_io_coordinates(self):
        """
        Generates random binary input and output coordinates and augments them with a
        layering dimension.
        """
        print(f"Generating random binary coordinates with {self.feature_dims} feature dimensions...")
        
        # 1. Generate unique random coordinates for the FEATURE dimensions
        input_feature_coors_list = self._generate_unique_binary_coords(self.obs_size, self.feature_dims)
        output_feature_coors_list = self._generate_unique_binary_coords(self.act_size, self.feature_dims)

        # Convert to NumPy arrays for easier manipulation
        input_feature_coors = np.array(input_feature_coors_list, dtype=float)
        output_feature_coors = np.array(output_feature_coors_list, dtype=float)

        # 2. Augment coordinates with the layering dimension
        input_layer_dim = np.zeros((input_feature_coors.shape[0], 1))
        output_layer_dim = np.full((output_feature_coors.shape[0], 1), self.depth_factor)
        
        input_coors_full = np.hstack([input_feature_coors, input_layer_dim])
        output_coors_full = np.hstack([output_feature_coors, output_layer_dim])

        # 3. Apply the width factor (this will scale the 0s and 1s)
        if self.width_factor != 1.0:
            print(f"Applying width factor: {self.width_factor}")
            input_coors_full[:, :-1] *= self.width_factor
            output_coors_full[:, :-1] *= self.width_factor

        # 4. The total coordinate size is feature_dims + 1
        final_coord_size = self.feature_dims + 1
        print(f"Added layering dimension. Final coordinate size: {final_coord_size}")

        # 5. Convert to list of tuples and add the bias node
        input_coors_list = [tuple(row) for row in input_coors_full]
        input_coors_list.append(tuple([0.0] * final_coord_size))
        
        output_coors_list = [tuple(row) for row in output_coors_full]
        
        return input_coors_list, output_coors_list