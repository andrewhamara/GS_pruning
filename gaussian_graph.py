from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


@dataclass
class GaussianNode:
    """A single Gaussian primitive: position, rotation (quat), scaling, opacity, features."""
    index: int
    xyz: np.ndarray            # (3,)
    rotation: np.ndarray       # (4,) quaternion
    scaling: np.ndarray        # (S,)  S=3 or 6 depending on file
    opacity: float
    features: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def __repr__(self):
        return (
            f"GaussianNode(i={self.index}, xyz={np.round(self.xyz, 4).tolist()}, "
            f"opacity={float(self.opacity):.4f}, "
            f"scale_dim={self.scaling.shape[0]}, feat_dim={self.features.shape[0]})"
        )


class GaussianSplatGraph:
    """
    A KNN graph over Gaussian splat primitives.

    Storage is column-oriented (one numpy array per attribute, length N) so loading
    is cheap; per-node views are materialized on demand via `get_node`.
    """

    def __init__(self, knn_neighbors: int = 10):
        self.knn_neighbors = knn_neighbors

        # node attributes (filled by load_ply)
        self.xyz       = np.empty((0, 3),  dtype=np.float32)
        self.rotation  = np.empty((0, 4),  dtype=np.float32)
        self.scaling   = np.empty((0, 0),  dtype=np.float32)
        self.opacity   = np.empty((0, 1),  dtype=np.float32)
        self.features  = np.empty((0, 0),  dtype=np.float32)

        # graph structure (filled by build_graph)
        self.neighbors       = None   # (N, K) int — directed KNN per node (self at col 0)
        self.edge_index      = None   # (2, |E|) int — derived from `adjacency`
        self.adjacency       = None   # (N, N) scipy.sparse.csr_matrix, symmetric by default
        self._symmetric_mode = None   # one of "union" | "mutual" | "none"

    # ------------------------------------------------------------------ loading

    def load_ply(self, path: str):
        """
        Load Gaussian primitives from a .ply file written in this project's format
        (see GaussianModel.save_ply). Missing optional attributes are zero-filled.
        """
        plydata = PlyData.read(path)
        v = plydata.elements[0]
        props = {p.name for p in v.properties}
        n = len(v)

        self.xyz = np.stack(
            [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1
        ).astype(np.float32)

        self.opacity = (
            np.asarray(v["opacity"])[:, None].astype(np.float32)
            if "opacity" in props else np.zeros((n, 1), dtype=np.float32)
        )

        self.scaling   = self._stack_indexed(v, props, "scale_")
        self.rotation  = self._stack_indexed(v, props, "rot")
        self.features  = self._stack_indexed(v, props, "feats_")

        # Default identity quaternion when no rotation present.
        if self.rotation.shape[1] == 0:
            self.rotation = np.zeros((n, 4), dtype=np.float32)
            self.rotation[:, 0] = 1.0

        return self

    @staticmethod
    def _stack_indexed(elem, props, prefix: str) -> np.ndarray:
        """Stack all properties named `<prefix>{i}` in numeric order into (N, K)."""
        names = [p for p in props if p.startswith(prefix) and p.split("_")[-1].isdigit()]
        names.sort(key=lambda s: int(s.split("_")[-1]))
        if not names:
            return np.empty((len(elem), 0), dtype=np.float32)
        return np.stack([np.asarray(elem[name]) for name in names], axis=1).astype(np.float32)

    # ------------------------------------------------------------------- graph

    def build_graph(self, k: Optional[int] = None, symmetric: str = "union"):
        """
        Build the KNN graph from node positions.

        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors (self counted, so K=10 gives 9 true neighbors).
        symmetric : {"union", "mutual", "none"}
            How to symmetrize the directed KNN graph:
              - "union":  edge if j in N(i) OR i in N(j).  A = max(A_dir, A_dir^T).
                          Preserves all KNN edges; never disconnects a node.
              - "mutual": edge only if j in N(i) AND i in N(j). A = A_dir * A_dir^T.
                          Sparser; may produce isolated nodes.
              - "none":   leave the adjacency directed (asymmetric).

        Attributes set
        --------------
        neighbors    : (N, K) int — raw directed KNN result (self at column 0).
                       Kept directed so per-node "K nearest" queries stay meaningful.
        adjacency    : (N, N) sparse, binary. Symmetric unless symmetric="none".
        edge_index   : (2, |E|) int — derived from `adjacency`; for symmetric graphs
                       contains both (i,j) and (j,i).
        """
        if self.xyz.shape[0] == 0:
            raise RuntimeError("No nodes loaded. Call load_ply first.")
        if k is not None:
            self.knn_neighbors = k

        n, K = self.xyz.shape[0], self.knn_neighbors
        tree = cKDTree(self.xyz)
        _, idx = tree.query(self.xyz, k=K)              # (N, K), self is column 0
        idx = idx.astype(np.int64)
        self.neighbors = idx

        src = np.repeat(np.arange(n, dtype=np.int64), K)
        dst = idx.reshape(-1)
        data = np.ones(src.shape[0], dtype=np.float32)
        A_dir = csr_matrix((data, (src, dst)), shape=(n, n))
        # de-duplicate any repeated (i,j) entries
        A_dir.sum_duplicates()
        A_dir.data[:] = 1.0

        if symmetric == "union":
            A = A_dir.maximum(A_dir.T)
        elif symmetric == "mutual":
            A = A_dir.multiply(A_dir.T)
        elif symmetric == "none":
            A = A_dir
        else:
            raise ValueError(f"symmetric must be 'union' | 'mutual' | 'none', got {symmetric!r}")

        A = A.tocsr()
        A.eliminate_zeros()
        A.data[:] = 1.0
        self.adjacency = A
        self._symmetric_mode = symmetric

        coo = A.tocoo()
        self.edge_index = np.stack(
            [coo.row.astype(np.int64), coo.col.astype(np.int64)], axis=0
        )
        return self

    # --------------------------------------------------------- spectral / prune

    def laplacian(self, normalized: bool = False):
        """
        Graph Laplacian of the (symmetric) adjacency.
          - combinatorial:  L = D - A
          - normalized sym: L_sym = I - D^{-1/2} A D^{-1/2}
        Requires a symmetric adjacency; raises otherwise.
        """
        if self.adjacency is None:
            raise RuntimeError("Graph not built. Call build_graph first.")
        if self._symmetric_mode == "none":
            raise RuntimeError(
                "Laplacian requires a symmetric adjacency. "
                "Rebuild with symmetric='union' or 'mutual'."
            )
        A = self.adjacency
        d = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
        if not normalized:
            return (diags(d) - A).tocsr()
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_is = diags(d_inv_sqrt)
        n = A.shape[0]
        return (diags(np.ones(n)) - D_is @ A @ D_is).tocsr()

    def compute_fiedler_vector(self,
                               normalized: bool = False,
                               sigma: float = -1e-8) -> Tuple[float, np.ndarray]:
        """
        Return (lambda_2, fiedler_vector) using scipy.sparse.linalg.eigsh.

        The graph Laplacian is rank-deficient (its smallest eigenvalue is 0
        for any connected component), so shift-invert at exactly sigma=0 calls
        splu on a singular matrix and fails. We default to a tiny *negative*
        shift, which makes `L - sigma*I = L + |sigma|*I` positive definite —
        cheap to factor — while still positioning the search near zero so
        ARPACK returns the smallest eigenvalues fast.

        If shift-invert still throws (e.g. pathological connectivity), we
        fall back to which="SA" which runs direct Lanczos and never needs
        a factorization, at the cost of being slower for large graphs.
        """
        L = self.laplacian(normalized=normalized).astype(np.float64)
        n = L.shape[0]
        if n < 3:
            raise RuntimeError(f"Need at least 3 nodes for a Fiedler vector; got {n}.")

        try:
            vals, vecs = eigsh(L, k=2, sigma=sigma, which="LM")
        except RuntimeError as e:
            if "singular" not in str(e).lower():
                raise
            print(f"[fiedler] shift-invert factor singular ({e}); "
                  f"falling back to which='SA' (slower).")
            vals, vecs = eigsh(L, k=2, which="SA")

        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        return float(vals[1]), vecs[:, 1].astype(np.float32)

    def prune_by_fiedler_magnitude(self,
                                   magnitudes: np.ndarray,
                                   k_percent: float,
                                   rebuild: bool = True) -> "GaussianSplatGraph":
        """
        Lower-level prune: given a precomputed per-node score (typically
        |Fiedler_i|), drop the top k% by that score. This is the work-horse
        used by sweeps that reuse the same eigenvector across many cuts.

        Parameters
        ----------
        magnitudes : (N,) array-like
            Per-node score, typically np.abs(fiedler_vec).
        k_percent : float
            Percentage of nodes to prune (must be in (0, 100)).
        rebuild : bool
            If True, rebuild the KNN graph on the survivors under the same
            symmetrization mode the parent used. If False, leave the graph
            structure on the returned object empty (useful when the caller
            only needs the surviving node arrays).
        """
        if not 0.0 < k_percent < 100.0:
            raise ValueError("k_percent must be in (0, 100), exclusive.")

        n = len(self)
        if magnitudes.shape != (n,):
            raise ValueError(f"magnitudes must have shape ({n},), got {magnitudes.shape}")

        n_remove = max(1, int(np.ceil(n * k_percent / 100.0)))
        keep_idx = np.argsort(magnitudes, kind="stable")[: n - n_remove]
        keep_idx.sort()                                # preserve original order
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep_idx] = True

        pruned = GaussianSplatGraph(knn_neighbors=self.knn_neighbors)
        pruned.xyz      = self.xyz[keep_mask]
        pruned.rotation = self.rotation[keep_mask]
        pruned.scaling  = self.scaling[keep_mask] if self.scaling.shape[1] else self.scaling
        pruned.opacity  = self.opacity[keep_mask]
        pruned.features = self.features[keep_mask] if self.features.shape[1] else self.features

        if rebuild:
            mode = self._symmetric_mode if self._symmetric_mode is not None else "union"
            pruned.build_graph(self.knn_neighbors, symmetric=mode)
        return pruned

    def prune_top_k_percent(self, k_percent: float, normalized: bool = False) -> "GaussianSplatGraph":
        """
        Convenience: compute the Fiedler vector here, then prune the top k%
        of nodes by |Fiedler_i|. For sweeps, prefer computing the Fiedler
        vector once and calling `prune_by_fiedler_magnitude` per cut.
        """
        _, fv = self.compute_fiedler_vector(normalized=normalized)
        return self.prune_by_fiedler_magnitude(np.abs(fv), k_percent, rebuild=True)

    def prune_random(self, k_percent: float, rng: Optional[np.random.Generator] = None,
                     rebuild: bool = True) -> "GaussianSplatGraph":
        """
        Prune a uniformly random k% of nodes. Used as the baseline against
        the Fiedler-magnitude prune to check whether spectral selection is
        actually buying anything over chance.
        """
        if rng is None:
            rng = np.random.default_rng()
        scores = rng.random(len(self)).astype(np.float32)
        return self.prune_by_fiedler_magnitude(scores, k_percent, rebuild=rebuild)

    # ------------------------------------------------------------- inspection

    def __len__(self):
        return int(self.xyz.shape[0])

    def __repr__(self):
        built = "unbuilt" if self.neighbors is None else f"K={self.knn_neighbors}"
        return (
            f"GaussianSplatGraph(N={len(self)}, feat_dim={self.features.shape[1]}, "
            f"scale_dim={self.scaling.shape[1]}, graph={built})"
        )

    def get_node(self, i: int) -> GaussianNode:
        return GaussianNode(
            index=i,
            xyz=self.xyz[i],
            rotation=self.rotation[i],
            scaling=self.scaling[i],
            opacity=float(self.opacity[i, 0]),
            features=self.features[i] if self.features.size else np.zeros(0, dtype=np.float32),
        )

    def neighbors_of(self, i: int) -> np.ndarray:
        if self.neighbors is None:
            raise RuntimeError("Graph not built. Call build_graph first.")
        return self.neighbors[i]

    def summary(self) -> dict:
        """A small dict of stats for quick inspection."""
        s = {
            "num_nodes":   len(self),
            "feat_dim":    int(self.features.shape[1]),
            "scale_dim":   int(self.scaling.shape[1]),
            "xyz_min":     self.xyz.min(axis=0).tolist() if len(self) else None,
            "xyz_max":     self.xyz.max(axis=0).tolist() if len(self) else None,
            "opacity_mean": float(self.opacity.mean()) if len(self) else None,
        }
        if self.neighbors is not None and self.edge_index is not None and self.adjacency is not None:
            s["K"]          = self.knn_neighbors
            s["symmetric"]  = self._symmetric_mode
            s["num_edges"]  = int(self.edge_index.shape[1])
            s["nnz_adj"]    = int(self.adjacency.nnz)
        return s
