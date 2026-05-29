"""
CLI inspector for GaussianSplatGraph.

Examples
--------
    python inspect_graph.py path/to/scene.ply
    python inspect_graph.py path/to/scene.ply --k 16
    python inspect_graph.py path/to/scene.ply --k 10 --node 0 --node 42
"""

import argparse
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from gaussian_graph import GaussianSplatGraph


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a Gaussian splat KNN graph from a .ply file.")
    p.add_argument("ply", help="Path to the input .ply file.")
    p.add_argument("--k", type=int, default=10, help="Number of KNN neighbors (default: 10).")
    p.add_argument("--symmetric", choices=["union", "mutual", "none"], default="union",
                   help="Adjacency symmetrization mode (default: union).")
    p.add_argument("--prune-percent", type=float, default=0.0,
                   help="Prune the top K%% of nodes by |Fiedler vector| magnitude.")
    p.add_argument("--normalized-laplacian", action="store_true",
                   help="Use the symmetric normalized Laplacian for the Fiedler vector.")
    p.add_argument("--node", type=int, action="append", default=[],
                   help="Index of a node to drill into. Repeatable.")
    p.add_argument("--show-edges", type=int, default=0,
                   help="Print this many edges from edge_index for a quick peek.")
    p.add_argument("--show-adj-block", type=int, default=0,
                   help="Print the top-left NxN block of the adjacency matrix as dense.")
    return p.parse_args()


def section(title: str):
    print()
    print(f"== {title} ==")


def main():
    args = parse_args()

    print(f"Loading {args.ply} ...")
    g = GaussianSplatGraph(knn_neighbors=args.k)
    g.load_ply(args.ply)
    print(f"  loaded: {g!r}")

    print(f"Building KNN graph with K={args.k} symmetric={args.symmetric} ...")
    g.build_graph(args.k, symmetric=args.symmetric)
    print(f"  built:  {g!r}")

    # ------------------------------------------------------------------ summary
    section("Graph summary")
    for k, v in g.summary().items():
        print(f"  {k:>13}: {v}")

    # ----------------------------------------------------------- degree stats
    section("Degree stats")
    assert g.adjacency is not None
    adj = g.adjacency
    is_sym = (adj != adj.T).nnz == 0
    print(f"  adjacency symmetric: {is_sym} (mode={g._symmetric_mode})")

    deg = np.asarray(adj.sum(axis=1)).ravel()
    print(f"  degree:  min={deg.min():.0f}  max={deg.max():.0f}  "
          f"mean={deg.mean():.2f}  std={deg.std():.2f}  median={np.median(deg):.0f}")
    isolated = int((deg == 0).sum())
    print(f"  isolated nodes: {isolated}")

    # Reconstruct the *directed* KNN matrix from neighbors so we can measure how
    # asymmetric the raw KNN was before symmetrization.
    assert g.neighbors is not None
    n_nodes, K = g.neighbors.shape
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), K)
    dst = g.neighbors.reshape(-1)
    A_dir = csr_matrix((np.ones_like(src, dtype=np.float32), (src, dst)),
                       shape=(n_nodes, n_nodes))
    A_dir.sum_duplicates(); A_dir.data[:] = 1.0

    mutual_dir = A_dir.multiply(A_dir.T)   # directed entries (i,j) with (j,i) also present
    n_dir       = int(A_dir.nnz)           # directed KNN edges (incl. self-loops at col 0)
    n_mutual    = int(mutual_dir.nnz)      # counts (i,j) and (j,i) separately
    n_oneway    = n_dir - n_mutual
    print(f"  directed KNN entries: {n_dir}   one-way: {n_oneway}   "
          f"mutual: {n_mutual} ({100.0 * n_mutual / max(n_dir, 1):.1f}% of directed)")
    n_undir_edges = adj.nnz // 2 if is_sym else adj.nnz
    print(f"  undirected edges in adjacency: {n_undir_edges}")

    # ------------------------------------------------------ connected components
    section("Connectivity")
    n_cc, labels = connected_components(adj, directed=not is_sym, return_labels=True)
    counts = np.bincount(labels)
    counts_sorted = np.sort(counts)[::-1]
    print(f"  components: {n_cc}")
    print(f"  largest component sizes: {counts_sorted[:5].tolist()}"
          + ("" if len(counts_sorted) <= 5 else f" ... (+{len(counts_sorted)-5} more)"))

    # ---------------------------------------------------- Fiedler / pruning
    if args.prune_percent > 0.0:
        section(f"Spectral prune (top {args.prune_percent:g}% by |Fiedler|)")
        lam2, fv = g.compute_fiedler_vector(normalized=args.normalized_laplacian)
        mag = np.abs(fv)
        print(f"  algebraic connectivity lambda_2 = {lam2:.6f}  "
              f"(normalized={args.normalized_laplacian})")
        print(f"  |fiedler| stats: min={mag.min():.4e}  mean={mag.mean():.4e}  "
              f"max={mag.max():.4e}")
        pruned = g.prune_top_k_percent(args.prune_percent,
                                       normalized=args.normalized_laplacian)
        print(f"  pruned graph: {pruned!r}")
        print(f"  surviving nodes: {len(pruned)} / {len(g)} "
              f"({100.0 * len(pruned) / len(g):.1f}%)")
        # swap g -> pruned so downstream sections inspect the pruned graph
        g = pruned
        assert g.adjacency is not None and g.neighbors is not None

    # ------------------------------------------------------ neighbor distances
    section("Neighbor distance stats")
    # column 0 is self; use columns 1.. for true neighbor distances
    assert g.neighbors is not None
    nbrs = g.neighbors
    pos  = g.xyz
    diffs = pos[nbrs[:, 1:]] - pos[:, None, :]
    dists = np.linalg.norm(diffs, axis=-1)            # (N, K-1)
    nearest = dists[:, 0]
    farthest = dists[:, -1]
    print(f"  nearest-neighbor dist:  min={nearest.min():.5f}  "
          f"mean={nearest.mean():.5f}  max={nearest.max():.5f}")
    print(f"  farthest-of-K  dist:    min={farthest.min():.5f}  "
          f"mean={farthest.mean():.5f}  max={farthest.max():.5f}")

    # --------------------------------------------------------------- per-node
    for i in args.node:
        section(f"Node {i}")
        if i < 0 or i >= len(g):
            print(f"  index {i} out of range [0, {len(g)})")
            continue
        node = g.get_node(i)
        print(f"  {node}")
        nbr = g.neighbors_of(i)
        print(f"  neighbors (incl. self at col 0): {nbr.tolist()}")
        if len(nbr) > 1:
            d = np.linalg.norm(g.xyz[nbr[1:]] - g.xyz[i], axis=-1)
            print(f"  neighbor distances:              {np.round(d, 5).tolist()}")

    # -------------------------------------------------------- raw edge sample
    if args.show_edges > 0:
        section(f"First {args.show_edges} edges (src -> dst)")
        assert g.edge_index is not None
        ei = g.edge_index[:, : args.show_edges]
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            print(f"  {s} -> {d}")

    # -------------------------------------------------------- adjacency block
    if args.show_adj_block > 0:
        n = min(args.show_adj_block, len(g))
        section(f"Adjacency top-left {n}x{n} block (dense)")
        print(g.adjacency[:n, :n].toarray().astype(int))

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
