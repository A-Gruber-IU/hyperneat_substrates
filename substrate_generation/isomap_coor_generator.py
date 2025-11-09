import numpy as np
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from substrate_generation.io_coor_processing import process_coordinates
import matplotlib.pyplot as plt


class IsomapIOEmbedder:
    """
    Uses Isomap to compute low-dimensional coordinates for IO features (sensors + motors).
    """
    def __init__(
        self,
        data: np.ndarray,
        obs_size: int,
        act_size: int,
        feature_dims: int,
        n_neighbors: int = 8,
        distance: str = "correlation",  # 'correlation' | 'cosine' | 'euclidean'
        width_factor: float = 1.0,
        normalize_coors: bool = True,
        depth_factor: float = 1.0,
    ):
        if data.shape[1] != obs_size + act_size:
            raise ValueError(f"Data shape mismatch. Expected {obs_size + act_size} features, but got {data.shape[1]}.")
        self.data = data
        self.obs_size = obs_size
        self.act_size = act_size
        self.feature_dims = feature_dims
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.width_factor = width_factor
        self.normalize_coors = normalize_coors
        self.depth_factor = depth_factor
        self.embedding_ = None  # (n_features, feature_dims)

    def _compute_feature_distance(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute a feature–feature distance matrix from standardized data Z (n_samples x n_features).
        """
        # Work across samples; we need per-feature vectors, so transpose:
        F = Z.T  # (n_features, n_samples)
        n_features = F.shape[0]

        if self.distance == "correlation":
            # Correlation similarity matrix:
            C = np.corrcoef(F)  # shape: (n_features, n_features)
            # Guard against numerical issues:
            C = np.clip(C, -1.0, 1.0)
            # Turn into a distance. Using 1 - |corr| is common for layout purposes.
            D = 1.0 - np.abs(C)
        elif self.distance == "cosine":
            # Cosine distance = 1 - cosine similarity
            # Normalize rows:
            norms = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
            Fn = F / norms
            S = Fn @ Fn.T  # cosine similarity
            S = np.clip(S, -1.0, 1.0)
            D = 1.0 - S
        elif self.distance == "euclidean":
            # Euclidean distance between feature vectors across samples
            # Using broadcasting to compute pairwise squared distances:
            G = F @ F.T
            sq = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
            sq = np.maximum(sq, 0.0)
            D = np.sqrt(sq)
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")

        # Zero out diagonal explicitly
        np.fill_diagonal(D, 0.0)
        # Check symmetry
        D = 0.5 * (D + D.T)
        return D

    def generate_io_coordinates(self):
        """
        Runs Isomap on the feature–feature distance matrix to get low-dimensional feature coordinates,
        then splits into input and output coordinates and augments with the depth dimension.
        """
        print(f"Standardizing data and building feature–feature distances using '{self.distance}' distance...")
        Z = StandardScaler().fit_transform(self.data)

        D = self._compute_feature_distance(Z)

        print(f"Running Isomap with n_neighbors={self.n_neighbors}, feature_dims={self.feature_dims}...")
        iso = Isomap(
            n_neighbors=self.n_neighbors,
            n_components=self.feature_dims,
            metric="precomputed",
            path_method="auto",
            neighbors_algorithm="auto"
        )
        # Isomap expects distances between "samples". Here, samples = features.
        self.embedding_ = iso.fit_transform(D)  # shape: (n_features, feature_dims)

        print("Isomap complete. Converting to IO coordinates...")
        all_feature_coors = self.embedding_  # (n_features, feature_dims)

        input_coors_list, output_coors_list = process_coordinates(
            all_feature_coors=all_feature_coors,
            normalize_coors=self.normalize_coors,
            width_factor=self.width_factor,
            obs_size=self.obs_size,
            depth_factor=self.depth_factor,
            feature_dims=self.feature_dims,
        )
        return input_coors_list, output_coors_list
    

    def plot_embedding_heatmap(self, save_path: str):
        """
        FA-style heatmap, but for Isomap embedding coordinates.
        Rows = features (Sensors first, then Motors), Columns = embedding dimensions.
        Saves to `save_path`.
        """
        if self.embedding_ is None:
            raise RuntimeError("You must call `generate_io_coordinates()` before calling `plot_embedding_heatmap()`.")

        # embedding_: (n_features, feature_dims)
        coords = self.embedding_
        n_features, d = coords.shape

        fig, ax = plt.subplots(figsize=(10, 12))

        # Use a diverging colormap since coordinates can be negative/positive.
        cmap = plt.cm.RdBu
        vmax = np.abs(coords).max() if np.isfinite(coords).all() else 1.0
        if vmax == 0:
            vmax = 1.0
        im = ax.imshow(coords, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
        cbar.set_label("Embedding Coordinate Value", weight='bold')

        # X ticks: embedding dimensions
        ax.set_xticks(np.arange(d))
        ax.set_xticklabels([f"Dim {i+1}" for i in range(d)])

        # Y ticks: features
        ax.set_ylabel("Nodes (Sensors & Motors)", weight='bold')
        ax.set_yticks(np.arange(n_features))

        # Separator between sensors and motors
        ax.axhline(y=self.obs_size - 0.5, color='white', linewidth=2.5, linestyle='--')
        ax.text(d, self.obs_size / 2, 'Sensors', ha='center', va='center',
                rotation=-90, color='white', weight='bold')
        ax.text(d, self.obs_size + self.act_size / 2, 'Motors', ha='center', va='center',
                rotation=-90, color='white', weight='bold')

        ax.set_title(f"Isomap Embedding Coordinates (d = {self.feature_dims})", fontsize=16, weight='bold')
        fig.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Isomap embedding heatmap saved to: {save_path}")