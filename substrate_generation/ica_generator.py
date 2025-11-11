import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from substrate_generation.io_coor_processing import process_coordinates

class ICAanalyzer:
    """
    Handles Independent Component Analysis (ICA) of environment data.
    """
    def __init__(self, data, obs_size, act_size, feature_dims, width_factor=1.0, normalize_coors=True, depth_factor=1, seed=None):
        if data.shape[1] != obs_size + act_size:
            raise ValueError(f"Data shape mismatch. Expected {obs_size + act_size} features, but got {data.shape[1]}.")
        self.data = data
        self.obs_size = obs_size
        self.act_size = act_size
        self.feature_dims = feature_dims # This is used as n_components for ICA
        self.width_factor = width_factor
        self.normalize_coors = normalize_coors
        self.depth_factor = depth_factor
        self.seed = seed
        
        self.ica = None
        self.raw_feature_coords = None # To store coordinates for the heatmap

    def generate_io_coordinates(self):
        """
        Runs ICA to generate feature coordinates and then calls the shared
        processing function to normalize, scale, and add layering.
        """
        print(f"Running Independent Component Analysis (ICA) to find {self.feature_dims} components...")
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        
        self.ica = FastICA(n_components=self.feature_dims, random_state=self.seed)
        self.ica.fit(scaled_data)

        print("ICA complete. Extracting coordinates (component loadings).")

        all_feature_coors = self.ica.components_.T
        self.raw_feature_coords = np.copy(all_feature_coors)

        input_coors_list, output_coors_list = process_coordinates(
            all_feature_coors=all_feature_coors,
            normalize_coors=self.normalize_coors,
            width_factor=self.width_factor,
            obs_size=self.obs_size,
            depth_factor=self.depth_factor,
            feature_dims=self.feature_dims
        )
        
        return input_coors_list, output_coors_list

    def plot_independent_components(self, save_path: str):
        """
        Generates a heatmap of the independent component loadings.
        """
        if self.ica is None or self.raw_feature_coords is None:
            raise RuntimeError("You must call `generate_io_coordinates()` before calling this method.")

        fig, ax = plt.subplots(figsize=(10, 12))
        
        components_to_plot = self.raw_feature_coords
        
        cmap = plt.cm.RdBu
        vmax = np.abs(components_to_plot).max()
        im = ax.imshow(components_to_plot, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
        cbar.set_label("Component Loading", weight='bold')

        ax.set_xticks(np.arange(self.feature_dims))
        ax.set_xticklabels([f"IC {i+1}" for i in range(self.feature_dims)])
        
        ax.set_ylabel("Nodes (Sensors & Motors)", weight='bold')
        ax.set_yticks(np.arange(self.obs_size + self.act_size))
        
        # Separator between sensors and motors
        ax.axhline(y=self.act_size - 0.5, color='white', linewidth=2.5, linestyle='--')

        ax.set_title(f"Independent Component Loadings ({self.feature_dims} Components)", fontsize=16, weight='bold')
        fig.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Independent component heatmap saved to: {save_path}")