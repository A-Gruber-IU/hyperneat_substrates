import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler

class SparseDictionaryAnalyzer:
    """
    Applies Sparse Dictionary Learning (SDL) to environment data to determine
    substrate coordinates.
    """
    def __init__(self, data, obs_size, act_size, max_dims, alpha=1.0, max_iter=1000):
        if data.shape[1] != obs_size + act_size:
            raise ValueError(f"Data shape mismatch. Expected {obs_size + act_size} features, but got {data.shape[1]}.")
        self.data = data
        self.obs_size = obs_size
        self.act_size = act_size
        # The number of dictionary "atoms" to learn.
        self.n_components = max_dims
        # The regularization parameter that encourages sparsity.
        self.alpha = alpha
        self.max_iter = max_iter
        self.sdl = None

    def generate_io_coordinates(self):
        """
        Runs Sparse Dictionary Learning to generate feature coordinates (loadings on each atom)
        and then augments them with a dedicated dimension for network layering.
        """
        print(
            f"Running Sparse Dictionary Learning to find {self.n_components} dictionary atoms..."
        )
        
        # 1. Standardize and fit the SDL model
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        
        # transform_alpha is the L1 regularization term that promotes sparsity.
        self.sdl = DictionaryLearning(
            n_components=self.n_components, 
            transform_alpha=None, 
            random_state=0,
            max_iter=self.max_iter,
        )
        # Note: SDL is often fit on the features directly, not the samples. 
        # But for consistency with PCA/FA to get node coordinates, we fit on the samples.
        self.sdl.fit(scaled_data)

        print("SDL complete. Extracting coordinates (loadings on dictionary atoms).")

        # 2. Extract feature coordinates. The dictionary atoms are in .components_
        # Shape is (n_components, n_features). Transpose to get coordinates for each node.
        all_feature_coords = self.sdl.components_.T
        input_feature_coors = all_feature_coords[:self.obs_size]
        output_feature_coors = all_feature_coords[self.obs_size:]

        # 3. Augment coordinates with the layering dimension
        input_layer_dim = np.zeros((input_feature_coors.shape[0], 1))
        output_layer_dim = np.ones((output_feature_coors.shape[0], 1))

        input_coors_full = np.hstack([input_feature_coors, input_layer_dim])
        output_coors_full = np.hstack([output_feature_coors, output_layer_dim])

        # 4. The total coordinate size is n_components + 1
        final_coord_size = self.n_components + 1
        print(f"Added layering dimension. Final coordinate size: {final_coord_size}")

        # 5. Convert to list of tuples
        input_coors_list = [tuple(row) for row in input_coors_full]
        output_coors_list = [tuple(row) for row in output_coors_full]
        
        return input_coors_list, output_coors_list

    def plot_dictionary_atoms(self, save_path: str):
        """
        Generates a heatmap of the learned dictionary atoms, which is the most
        informative visualization for SDL.
        """
        if self.sdl is None:
            raise RuntimeError("You must call `generate_io_coordinates()` before calling this method.")

        fig, ax = plt.subplots(figsize=(10, 12))
        
        # The dictionary atoms are the components of the fitted model
        atoms = self.sdl.components_
        
        # Use a diverging colormap since values can be positive or negative
        cmap = plt.cm.RdBu
        vmax = np.abs(atoms).max()
        im = ax.imshow(atoms.T, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
        cbar.set_label("Atom Loading Value", weight='bold')

        ax.set_xticks(np.arange(self.n_components))
        ax.set_xticklabels([f"Atom {i+1}" for i in range(self.n_components)])
        
        ax.set_ylabel("Original Nodes (Sensors & Motors)", weight='bold')
        ax.set_yticks(np.arange(self.obs_size + self.act_size))
        
        ax.axhline(y=self.obs_size - 0.5, color='white', linewidth=2.5, linestyle='--')
        ax.text(self.n_components, self.obs_size / 2, 'Sensors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        ax.text(self.n_components, self.obs_size + self.act_size / 2, 'Motors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        
        ax.set_title(f"Learned Dictionary Atoms ({self.n_components} Components)", fontsize=16, weight='bold')
        fig.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Dictionary atoms heatmap saved to: {save_path}")