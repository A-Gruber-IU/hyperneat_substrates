import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from substrate_generation.io_coor_processing import process_coordinates

class InvPCAanalyzer:
    """
    Handles inverse PCA analysis of environment data to determine substrate coordinates which express least variance.
    """
    def __init__(self, data, obs_size, act_size, feature_dims, hidden_depth, seed=None, width_factor=1.0, normalize_coors=True, depth_factor=1):
        if data.shape[1] != obs_size + act_size:
            raise ValueError(f"Data shape mismatch...")
        self.data = data
        self.obs_size = obs_size
        self.act_size = act_size
        self.feature_dims = feature_dims
        self.output_depth = hidden_depth + 1
        self.seed = seed
        self.width_factor = width_factor
        self.depth_factor = depth_factor
        self.normalize_coors = normalize_coors
        self.pca = None
        self.final_dims_indices = None # Renamed for clarity

    def generate_io_coordinates(self):
        """
        Runs PCA and selects the components with the LEAST variance to use as coordinates.
        """
        print(f"Running inverse PCA to find {self.feature_dims} feature dimensions covering least variance...")
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        self.pca = PCA(random_state=self.seed)
        self.pca.fit(scaled_data)

        # Here is the distinction from the regular PCA class: variances are sorted and smallest values up to feature_dims setting are selected
        sorted_indices = np.argsort(self.pca.explained_variance_ratio_)
        self.final_dims_indices = sorted_indices[:self.feature_dims]

        all_feature_coors = self.pca.components_[self.final_dims_indices].T

        input_coors_list, output_coors_list = process_coordinates(
            all_feature_coors=all_feature_coors,
            normalize_coors=self.normalize_coors,
            width_factor=self.width_factor,
            obs_size=self.obs_size,
            depth_factor=self.depth_factor,
            feature_dims=self.feature_dims,
        )
        
        # This was missing from your original code, which would cause issues later
        return input_coors_list, output_coors_list

    def plot_least_variance_components(self, save_path: str):
        """
        Generates a plot that resembles the original PCA variance plot but highlights
        the components with the lowest explained variance.
        """
        if self.pca is None or self.final_dims_indices is None:
            raise RuntimeError("You must call `generate_io_coordinates()` before calling this method.")

        all_variances = self.pca.explained_variance_ratio_
        num_total_components = len(all_variances)

        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # All components
        x_coors = np.arange(num_total_components)
        ax1.bar(
            x_coors,
            all_variances * 100,
            alpha=0.6,
            color='tab:blue',
            label='Individual Explained Variance (All PCs)'
        )
        
        # Highlight of the selected low-variance components
        selected_variances = all_variances[self.final_dims_indices] * 100
        
        # Draw a second set of bars in red, only at the selected indices
        ax1.bar(
            self.final_dims_indices,
            selected_variances,
            color='tab:red',
            label=f'Selected {self.feature_dims} Lowest Variance Components'
        )

        ax1.set_xlabel('Principal Component Index (Ordered by Variance)')
        ax1.set_ylabel('Explained Variance (%)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xticks(x_coors)
        ax1.set_xticklabels(x_coors + 1)

        # Cumulative variance line
        ax2 = ax1.twinx()
        cumulative_variance = np.cumsum(all_variances) * 100
        ax2.plot(
            x_coors,
            cumulative_variance,
            color='tab:orange', # Changed color to avoid confusion with red bars
            marker='o',
            linestyle='--',
            label='Cumulative Explained Variance'
        )
        ax2.set_ylabel('Cumulative Explained Variance (%)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylim([0, 105])

        fig.suptitle('PCA Explained Variance (Lowest Variance Components Highlighted)', fontsize=16)
        # Combine legends from both axes for a clean look
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right')
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Inverse PCA variance plot saved to: {save_path}")

    def plot_principal_components(self, save_path: str):
        if self.pca is None or self.final_dims_indices is None:
            raise RuntimeError("You must call `generate_io_coordinates()` before calling this method.")

        fig, ax = plt.subplots(figsize=(10, 12))
        
        components_to_plot = self.pca.components_[self.final_dims_indices].T

        cmap = plt.cm.RdBu
        vmax = np.abs(components_to_plot).max()
        im = ax.imshow(components_to_plot, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
        cbar.set_label("Component Loading", weight='bold')

        # Use feature_dims for ranges and the indices for labels
        ax.set_xticks(np.arange(self.feature_dims))
        # Label with the *actual* PC index for better interpretability
        xtick_labels = [f"PC {i+1}" for i in self.final_dims_indices]
        ax.set_xticklabels(xtick_labels)
        
        ax.set_ylabel("Original Nodes (Sensors & Motors)", weight='bold')
        ax.set_yticks(np.arange(self.obs_size + self.act_size))
        
        ax.axhline(y=self.obs_size - 0.5, color='white', linewidth=2.5, linestyle='--')
        # Use feature_dims for text placement
        ax.text(self.feature_dims, self.obs_size / 2, 'Sensors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        ax.text(self.feature_dims, self.obs_size + self.act_size / 2, 'Motors', ha='center', va='center', rotation=-90, color='white', weight='bold')
        
        ax.set_title(f"Principal Component Loadings (Lowest {self.feature_dims} Variance)", fontsize=16, weight='bold')
        fig.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Principal component heatmap saved to: {save_path}")