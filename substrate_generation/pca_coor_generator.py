import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from substrate_generation.io_coor_processing import process_coordinates

class PCAanalyzer:
    """
    Handles PCA analysis of environment data to determine substrate coordinates which express highest variance.
    """
    def __init__(self, data, obs_size, act_size, variance_threshold, max_dims, hidden_depth, width_factor=1.0, normalize_coors=True):
        if data.shape[1] != obs_size + act_size:
            raise ValueError(f"Data shape mismatch. Expected {obs_size + act_size} features, but got {data.shape[1]}.")
        self.data = data
        self.obs_size = obs_size
        self.act_size = act_size
        self.variance_threshold = variance_threshold
        self.max_dims = max_dims
        self.output_depth = hidden_depth + 1
        self.width_factor = width_factor
        self.normalize_coors = normalize_coors
        self.pca = None
        self.final_dims = None

    def generate_io_coordinates(self):
        """
        Runs PCA to generate feature coordinates and then augments them with a
        dedicated dimension for network layering (inputs at 0, outputs at 1).
        """
        print(
            f"Running PCA to find feature dimensions covering {self.variance_threshold*100:.1f}% of variance "
            f"(with a hard limit of {self.max_dims} dimensions)..."
        )
        
        # Standardize and fit PCA to find the feature dimensions
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        self.pca = PCA()
        self.pca.fit(scaled_data)

        # Determine the number of feature dimensions
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        dims_for_variance = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        self.final_dims = min(dims_for_variance, self.max_dims)

        print(
            f"PCA found {dims_for_variance} dimensions needed for {self.variance_threshold*100:.1f}% variance."
        )
        print(f"Applying max limit. Final number of feature dimensions: {self.final_dims}")

        # Extract the feature coordinates from PCA results
        # pca.components_ has a shape of (num_components, num_features). Each row is a PC. Each column is one of the original nodes. (The .T transposes this)
        # this effectively uses the PCs' loadings as coordinates, e.g. for 3 PCs a coor would be (loading_on_PC1, loading_on_PC2, loading_on_PC3)
        all_feature_coors = self.pca.components_[:self.final_dims].T

        input_coors_list, output_coors_list = process_coordinates(
            all_feature_coors=all_feature_coors,
            normalize_coors=self.normalize_coors,
            width_factor=self.width_factor,
            obs_size=self.obs_size,
            output_depth=self.output_depth,
            feature_dims=self.final_dims,
        )
        
        return input_coors_list, output_coors_list

    def plot_variance(self, save_path: str):
        """
        Generates and saves a scree plot of the PCA explained variance.
        This method must be called after generate_input_coordinates has been run.
        """
        if self.pca is None or self.final_dims is None:
            raise RuntimeError("You must call `generate_coordinates()` before calling `plot_variance()`.")

        display_components = len(self.pca.explained_variance_ratio_)
        fig, ax1 = plt.subplots(figsize=(12, 7))

        ax1.set_xlabel('Principal Component (Feature Dimensions)')
        ax1.set_ylabel('Explained Variance (%)', color='tab:blue')
        
        ax1.bar(
            range(1, display_components + 1),
            self.pca.explained_variance_ratio_ * 100,
            alpha=0.6, color='tab:blue', label='Individual Explained Variance'
        )
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xticks(range(1, display_components + 1))

        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative Explained Variance (%)', color='tab:red')
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_) * 100
        ax2.plot(
            range(1, display_components + 1), cumulative_variance,
            color='tab:red', marker='o', linestyle='-', label='Cumulative Explained Variance'
        )
        ax2.tick_params(axis='y', labelcolor='tab:red')

        ax2.axhline(y=self.variance_threshold * 100, color='g', linestyle='--',
                    label=f'{self.variance_threshold*100:.0f}% Variance Threshold')
        
        ax1.axvline(x=self.final_dims, color='purple', linestyle=':',
                    label=f'Selected Dimensions: {self.final_dims}')
        
        plt.xlim([0.5, display_components + 0.5])
        ax2.set_ylim([0, 105])

        fig.suptitle('PCA Explained Variance', fontsize=16)
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"PCA variance plot saved to: {save_path}")

    def plot_principal_components(self, save_path: str):
        """
        Generates a heatmap of the principal component loadings, which represent
        the feature coordinates. This provides a direct comparison to FA and SDL heatmaps.
        """
        if self.pca is None or self.final_dims is None:
            raise RuntimeError("You must call `generate_io_coordinates()` before calling this method.")

        fig, ax = plt.subplots(figsize=(10, 12))
        
        # The data for the heatmap is the component matrix (transposed for plotting)
        # This is the same matrix used to generate the coordinates.
        components_to_plot = self.pca.components_[:self.final_dims].T
        
        # Use a diverging colormap because loadings can be positive or negative
        cmap = plt.cm.RdBu
        vmax = np.abs(components_to_plot).max()
        im = ax.imshow(components_to_plot, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
        cbar.set_label("Component Loading", weight='bold')

        ax.set_xticks(np.arange(self.final_dims))
        ax.set_xticklabels([f"PC {i+1}" for i in range(self.final_dims)])
        
        ax.set_ylabel("Original Nodes (Sensors & Motors)", weight='bold')
        ax.set_yticks(np.arange(self.obs_size + self.act_size))
        
        ax.axhline(y=self.obs_size - 0.5, color='white', linewidth=2.5, linestyle='--')
        ax.text(self.final_dims, self.obs_size / 2, 'Sensors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        ax.text(self.final_dims, self.obs_size + self.act_size / 2, 'Motors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        
        ax.set_title(f"Principal Component Loadings ({self.final_dims} Components)", fontsize=16, weight='bold')
        fig.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Principal component heatmap saved to: {save_path}")