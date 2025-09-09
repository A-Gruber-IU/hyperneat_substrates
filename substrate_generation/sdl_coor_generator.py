import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler
from substrate_generation.io_coor_processing import process_coordinates

class SparseDictionaryAnalyzer:
    """
    Applies Sparse Dictionary Learning (SDL) to environment data to determine
    substrate coordinates.
    """
    def __init__(self, data, obs_size, act_size, max_dims, hidden_depth, alpha=1.0, max_iter=1000, width_factor=1.0, normalize_coors=True):
        if data.shape[1] != obs_size + act_size:
            raise ValueError(f"Data shape mismatch. Expected {obs_size + act_size} features, but got {data.shape[1]}.")
        self.data = data
        self.obs_size = obs_size
        self.act_size = act_size
        # The number of dictionary "atoms" to learn.
        self.feature_dims = max_dims
        # The regularization parameter that encourages sparsity.
        self.alpha = alpha
        self.max_iter = max_iter
        self.width_factor = width_factor
        self.normalize_coors = normalize_coors
        self.sdl = None
        self.output_depth = hidden_depth + 1

    def generate_io_coordinates(self):
        """
        Runs Sparse Dictionary Learning to generate feature coordinates (loadings on each atom)
        and then augments them with a dedicated dimension for network layering.
        """
        print(
            f"Running Sparse Dictionary Learning to find {self.feature_dims} dictionary atoms..."
        )
        
        # 1. Standardize and fit the SDL model
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        
        # transform_alpha is the L1 regularization term that promotes sparsity.
        self.sdl = DictionaryLearning(
            n_components=self.feature_dims, 
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
        all_feature_coors = self.sdl.components_.T

        input_coors_list, output_coors_list = process_coordinates(
            all_feature_coors=all_feature_coors,
            normalize_coors=self.normalize_coors,
            width_factor=self.width_factor,
            obs_size=self.obs_size,
            output_depth=self.output_depth,
            feature_dims=self.feature_dims,
        )
        
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

        ax.set_xticks(np.arange(self.feature_dims))
        ax.set_xticklabels([f"Atom {i+1}" for i in range(self.feature_dims)])
        
        ax.set_ylabel("Original Nodes (Sensors & Motors)", weight='bold')
        ax.set_yticks(np.arange(self.obs_size + self.act_size))
        
        ax.axhline(y=self.obs_size - 0.5, color='white', linewidth=2.5, linestyle='--')
        ax.text(self.feature_dims, self.obs_size / 2, 'Sensors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        ax.text(self.feature_dims, self.obs_size + self.act_size / 2, 'Motors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        
        ax.set_title(f"Learned Dictionary Atoms ({self.feature_dims} Components)", fontsize=16, weight='bold')
        fig.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Dictionary atoms heatmap saved to: {save_path}")