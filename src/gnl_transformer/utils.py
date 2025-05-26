import numpy as np
import networkx as nx
from itertools import combinations
from joblib import Parallel, delayed
from typing import Any, Optional, Union


# Geometry helpers
def angle_between(v1: np.ndarray,
                  v2: np.ndarray,
                  origin: Optional[np.ndarray] = None) -> float:
    """
    Return the angle (rad) between vectors v1 and v2, optionally translated so
    that *origin* acts as the common tail.
    """
    if origin is not None:
        v1, v2 = v1 - origin, v2 - origin
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    # Clip protects against FP inaccuracy
    return float(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))


# Line-graph builder
def line_graph_undirected(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    with_triplet_angles: bool = False,
    include_selfloops: bool = False,
    create_using: Optional[Any] = None,
    relabel_nodes: bool = False,
) -> nx.Graph:
    """
    Build the undirected line graph L(G).

    Every edge (u, v[, key]) in *G* becomes a node in L(G).  
    Two nodes in L(G) share an edge iff their corresponding edges in *G* are
    incident on a common endpoint.

    Parameters
    ----------
    G : nx.Graph | nx.MultiGraph
    with_triplet_angles : bool, default False
        Replicates the original “triplet feature”:
        * joint node attributes of the shared endpoint
        * triplet center         (mean of three endpoint positions)
        * list of incident angles (first is ∠ (u-common-v); the rest come from
          optional `'pts2'` samples stored on each edge in *G*)
    include_selfloops : bool, default False
        Keep e–e self-loops in L(G).  They are ignored by default.
    create_using : graph constructor, optional
        Lets you build e.g. an `nx.MultiGraph` line graph by passing
        `create_using=nx.MultiGraph`.
    relabel_nodes : bool, default False
        If True, the nodes in the line graph will be relabeled to integers
        starting from 0. This is useful for compatibility with some algorithms
        that expect integer node labels.

    Returns
    -------
    nx.Graph (or the type you supply via *create_using*)
    """
    # Small helpers
    def canonical(u, v, k=None):
        """Stable, hashable identifier for an (undirected) edge."""
        return (u, v, k) if u <= v else (v, u, k)

    # Prep: collect edge list and fast incidence map
    if G.is_multigraph():
        all_edges = [(u, v, k, d)
                     for u, v, k, d in G.edges(keys=True, data=True)]
    else:
        all_edges = [(u, v, None, d)
                     for u, v, d in G.edges(data=True)]

    incidence = {n: [] for n in G}               # node → list[(canonical-id, raw-edge)]
    for u, v, k, d in all_edges:
        cid = canonical(u, v, k)
        incidence[u].append((cid, (u, v, k, d)))
        incidence[v].append((cid, (u, v, k, d)))

    # Create L(G) and populate edge-nodes
    L = nx.empty_graph(0, create_using)          # preserves the template type

    for cid, (_, _, _, data) in {canonical(u, v, k): (u, v, k, d)
                                 for u, v, k, d in all_edges}.items():
        L.add_node(cid, **data)

    # Connect edge-nodes sharing a common endpoint
    for shared in G:
        incident = incidence[shared]
        if not incident:
            continue

        if include_selfloops:
            # (e_i, e_j) including i == j
            pair_iter = ((incident[i], incident[j])
                         for i in range(len(incident))
                         for j in range(i, len(incident)))
        else:
            pair_iter = combinations(incident, 2)

        for (cid_a, (u1, v1, k1, d1)), (cid_b, (u2, v2, k2, d2)) in pair_iter:
            if cid_a == cid_b:
                continue                                          # skip trivial
            attr = {}
            if with_triplet_angles:
                pos_shared = G.nodes[shared]['pos']
                other1 = v1 if u1 == shared else u1
                other2 = v2 if u2 == shared else u2
                pos1, pos2 = G.nodes[other1]['pos'], G.nodes[other2]['pos']

                attr['joint_node_attr'] = G.nodes[shared]
                attr['triplet_center'] = np.mean([pos_shared, pos1, pos2],
                                                 axis=0)

                angles = [angle_between(pos1, pos2, origin=pos_shared)]
                for pts in d1.get('pts2', ()):
                    angles.append(angle_between(pos2, pts,
                                                origin=pos_shared))
                for pts in d2.get('pts2', ()):
                    angles.append(angle_between(pos1, pts,
                                                origin=pos_shared))
                attr['angle'] = np.asarray(angles, dtype=np.float32)

            L.add_edge(cid_a, cid_b, **attr)

    # ensure predictable ordering if someone iterates over L.nodes
    if relabel_nodes:
        # deterministic node ordering for edge-tuples
        rank = {n: i for i, n in enumerate(G)}
        def lnode_sort_key(e):
            u, v, _ = e
            return (rank[u], rank[v])
        
        sorted_nodes = sorted(L.nodes, key=lnode_sort_key)
        mapping = {old: new for new, old in enumerate(sorted_nodes)}

        # preserve the canonical tuple under node-attr "cid"
        nx.set_node_attributes(L, {old: {"cid": old} for old in L.nodes})
        L = nx.relabel_nodes(L, mapping, copy=False)
    
    return L

# --- HSG-topology helper functions --- #
def _simple_copy_with_multiplicity(g_multi: nx.MultiGraph):
    """Collapse a (multi)graph into a simple Graph, recording multiplicity."""
    if not g_multi.is_multigraph():
        return g_multi          # already simple, nothing to do

    g_simple = nx.Graph() if isinstance(g_multi, nx.MultiGraph) else nx.DiGraph()
    g_simple.add_nodes_from(g_multi.nodes(data=True))

    for u, v, _k, data in g_multi.edges(keys=True, data=True):
        if g_simple.has_edge(u, v):
            g_simple[u][v]["m"] += 1          # bump multiplicity
        else:
            g_simple.add_edge(u, v, **data, m=1)
    return g_simple

def wl_hash_safe(g, iters=3):
    """WL hash that tolerates MultiGraphs by collapsing them first."""
    g_for_hash = _simple_copy_with_multiplicity(g)
    return nx.weisfeiler_lehman_graph_hash(
        g_for_hash,
        iterations=iters,
        edge_attr="m"  # include multiplicity in the label
    )

def isomorphism_classes_by_WL_hash(graphs, iters=3, n_jobs=-1):
    hashes = Parallel(n_jobs=n_jobs, batch_size=512)(
        delayed(wl_hash_safe)(g, iters=iters) for g in graphs
    )
    class_indices = {}
    for idx, h in enumerate(hashes):
        class_indices.setdefault(h, []).append(idx)
    return list(class_indices.values())