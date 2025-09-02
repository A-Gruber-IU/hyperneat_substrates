import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAanalyzer:
    """
    Handles PCA analysis of environment data to determine substrate coordinates.
    It stores the fitted PCA model to allow for separate analysis and plotting steps.
    """
    def __init__(self, data, obs_size, act_size, variance_threshold, max_dims):
        if data.shape[1] != obs_size + act_size:
            raise ValueError(
                f"Data shape mismatch. Expected {obs_size + act_size} features, "
                f"but got {data.shape[1]}."
            )
        self.data = data
        self.obs_size = obs_size
        self.act_size = act_size
        self.variance_threshold = variance_threshold
        self.max_dims = max_dims
        
        # Attributes that will be computed and stored
        self.pca = None
        self.final_dims = None

    def generate_input_coordinates(self):
        """
        Runs PCA on the data and returns the input/output coordinates based on the
        variance threshold and max dimension limit.
        """
        print(
            f"Running PCA to find dimensions covering {self.variance_threshold*100:.1f}% of variance "
            f"(with a hard limit of {self.max_dims} dimensions)..."
        )
        
        # 1. Standardize the data and fit the PCA model
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        self.pca = PCA()
        self.pca.fit(scaled_data)

        # 2. Determine the number of dimensions dynamically
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        dims_for_variance = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        self.final_dims = min(dims_for_variance, self.max_dims)

        print(
            f"PCA found {dims_for_variance} dimensions are needed for {self.variance_threshold*100:.1f}% variance."
        )
        print(f"Applying max limit. Final number of dimensions: {self.final_dims}")

        # 3. Extract and return the coordinates
        all_coords = self.pca.components_[:self.final_dims].T
        input_coors = all_coords[:self.obs_size]
        output_coors = all_coords[self.obs_size:]
        
        input_coors_list = [tuple(row) for row in input_coors]
        output_coors_list = [tuple(row) for row in output_coors]

        input_coors_list.append(tuple([0.0] * self.final_dims))

        return input_coors_list, output_coors_list, self.final_dims

    def plot_variance(self, save_path: str):
        """
        Generates and saves a scree plot of the PCA explained variance.
        This method must be called after generate_input_coordinates has been run.
        """
        if self.pca is None or self.final_dims is None:
            raise RuntimeError("You must call `generate_input_coordinates()` before calling `plot_variance()`.")

        display_components = len(self.pca.explained_variance_ratio_)
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        ax1.set_xlabel('Principal Component')
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
