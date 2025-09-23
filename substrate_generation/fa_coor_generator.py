import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from substrate_generation.io_coor_processing import process_coordinates

class FactorAnalyzer:
    """
    Handles Factor Analysis of environment data to determine substrate coordinates.
    It serves as an alternative to the PCAanalyzer.
    """
    def __init__(self, data, obs_size, act_size, feature_dims, hidden_depth, width_factor=1.0, normalize_coors=True, depth_factor=1):
        if data.shape[1] != obs_size + act_size:
            raise ValueError(f"Data shape mismatch. Expected {obs_size + act_size} features, but got {data.shape[1]}.")
        self.data = data
        self.obs_size = obs_size
        self.act_size = act_size
        # For Factor Analysis, we must specify the number of components (factors) beforehand.
        self.feature_dims = feature_dims
        self.output_depth = hidden_depth + 1
        self.width_factor = width_factor
        self.normalize_coors = normalize_coors
        self.fa = None
        self.depth_factor = depth_factor

    def generate_io_coordinates(self):
        """
        Runs Factor Analysis to generate feature coordinates and then augments them with a
        dedicated dimension for network layering (inputs at 0, outputs at 1).
        """
        print(
            f"Running Factor Analysis to find {self.feature_dims} latent factors..."
        )
        
        # 1. Standardize and fit FA to find the latent factors
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        
        self.fa = FactorAnalysis(n_components=self.feature_dims, random_state=0)
        self.fa.fit(scaled_data)

        print(f"Factor Analysis complete. Extracting coordinates (factor loadings).")

        # 2. Extract the feature coordinates from the factor loadings.
        # fa.components_ has a shape of (num_factors, num_features).
        # The values are the "loadings" of each node onto each factor.
        # The values are transposed to get (num_features, num_factors).
        all_feature_coors = self.fa.components_.T

        input_coors_list, output_coors_list = process_coordinates(
            all_feature_coors=all_feature_coors,
            normalize_coors=self.normalize_coors,
            width_factor=self.width_factor,
            obs_size=self.obs_size,
            depth_factor=self.depth_factor,
            feature_dims=self.feature_dims,
        )
        
        return input_coors_list, output_coors_list

    def plot_factor_loadings(self, save_path: str):
        """
        Generates a heatmap of the factor loadings, which is more informative for FA
        than a scree plot.
        """
        if self.fa is None:
            raise RuntimeError("You must call `generate_io_coordinates()` before calling `plot_factor_loadings()`.")

        fig, ax = plt.subplots(figsize=(10, 12))
        
        # The components_ matrix has shape (n_factors, n_features)
        loadings = self.fa.components_
        
        # Use a diverging colormap because loadings can be positive or negative
        cmap = plt.cm.RdBu
        vmax = np.abs(loadings).max()
        im = ax.imshow(loadings.T, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
        cbar.set_label("Factor Loading", weight='bold')

        # Set up ticks and labels
        ax.set_xticks(np.arange(self.feature_dims))
        ax.set_xticklabels([f"Factor {i+1}" for i in range(self.feature_dims)])
        
        ax.set_ylabel("Original Nodes (Sensors & Motors)", weight='bold')
        ax.set_yticks(np.arange(self.obs_size + self.act_size))
        
        # Add a line to separate sensors from motors
        ax.axhline(y=self.obs_size - 0.5, color='white', linewidth=2.5, linestyle='--')
        ax.text(self.feature_dims, self.obs_size / 2, 'Sensors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        ax.text(self.feature_dims, self.obs_size + self.act_size / 2, 'Motors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        
        ax.set_title(f"Factor Loadings for {self.feature_dims} Latent Factors", fontsize=16, weight='bold')
        fig.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Factor loadings heatmap saved to: {save_path}")