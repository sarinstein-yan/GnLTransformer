import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from typing import Any, Optional, Union


# helper ─ angle between two vectors (optionally translated to a common origin)
def angle_between(v1: np.ndarray,
                  v2: np.ndarray,
                  origin: Optional[np.ndarray] = None) -> float:
    if origin is not None:
        v1, v2 = v1 - origin, v2 - origin

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0

    # numerical safety
    return float(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))


# main API ─ undirected line graph
def line_graph_undirected(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    with_triplet_features: bool = False,
    create_using: Any = None,
    add_selfloops: bool = False,
    relabel_nodes: bool = False,
):
    """
    Construct the undirected *line graph* L of G.

    Parameters
    ----------
    G : nx.Graph | nx.MultiGraph
        Source (multi-)graph.
    with_triplet_features : bool, optional
        Add ``joint_node_attr``, ``triplet_center`` and ``angle`` features.
    create_using : graph constructor, optional
        Graph type for the output (default → ``G.__class__``).
    add_selfloops : bool, optional
        If True, allow self-loops in L (an edge of G paired with itself).
    relabel_nodes : bool, default False
        If True, the nodes in the line graph will be relabeled to integers
        starting from 0. This is useful for compatibility with some algorithms
        that expect integer node labels.
    
    Returns
    -------
    nx.Graph | nx.MultiGraph
        The resulting line graph.
    """
    # ------------------------------------------------------------------
    # 0.  Boiler-plate / helpers
    # ------------------------------------------------------------------
    L = nx.empty_graph(0, create_using, default=G.__class__)
    node_index = {n: i for i, n in enumerate(G)}          # stable order

    # edge iterator that always yields (u, v, key, data_dict)
    if G.is_multigraph():
        edge_iter = lambda n: G.edges(n, keys=True, data=True)
    else:
        edge_iter = lambda n: ((u, v, None, d) for u, v, d in G.edges(n, data=True))

    # canonical representation of an edge (u < v by node_index)
    def canonical(u, v, k):
        return (u, v, k) if node_index[u] < node_index[v] else (v, u, k)

    edge_seen = set()      # avoid double-adding edges to L
    shift = 0 if add_selfloops else 1

    # ------------------------------------------------------------------
    # 1.  For every *vertex* in G …
    #     * collect its incident edges (already canonicalised)
    #     * add the corresponding nodes to L
    #     * connect every unordered pair of those edges
    # ------------------------------------------------------------------
    for tail in G:
        incident = []  # [(canon_edge, raw_edge_tuple)] for this tail

        for u, v, k, data in edge_iter(tail):
            c = canonical(u, v, k)
            L.add_node(c, **data)          # edge->node transfer
            if 'pts5' in data:
                L.nodes[c]['pos'] = data['pts5'][4:6]
            incident.append((c, (u, v, k)))  # remember the *raw* tuple too

        # vertex of degree-1 ⇒ its single incident edge becomes isolated node
        if len(incident) == 1:
            continue

        # pairwise combinations (shift skips or keeps the i==j case)
        for i, (c_a, raw_a) in enumerate(incident):
            for c_b, raw_b in incident[i + shift:]:

                # undirected → ensure ordering in the (frozenset-like) key
                ekey = tuple(sorted((c_a, c_b), key=lambda x: (node_index[x[0]], node_index[x[1]])))
                if ekey in edge_seen:
                    continue
                edge_seen.add(ekey)

                # ------------------------------------------------------
                # shared tail vertex  (might be ambiguous for self-loops;
                # pop() replicates the “arbitrary but deterministic” choice
                # of the original implementation)
                # ------------------------------------------------------
                shared = set(raw_a[:2]).intersection(raw_b[:2]).pop()

                # ------------------------------------------------------
                # optional triplet-level geometric features
                # ------------------------------------------------------
                attr = {}
                if with_triplet_features:
                    # node attributes of the shared vertex
                    attr["joint_node_attr"] = G.nodes[shared]

                    # the *other* endpoints of raw_a / raw_b
                    v = raw_a[0] if raw_a[0] != shared else raw_a[1]
                    w = raw_b[0] if raw_b[0] != shared else raw_b[1]

                    pos_shared = attr["joint_node_attr"]["pos"]
                    pos_v      = G.nodes[v]["pos"]
                    pos_w      = G.nodes[w]["pos"]

                    # centre of the triangle (shared, v, w)
                    attr["triplet_center"] = np.mean([pos_v, pos_w, pos_shared], axis=0)

                    # angles:   ∠(v, w) plus those with intermediate pts2’s
                    angles = [angle_between(pos_v, pos_w, origin=pos_shared)]

                    data_a = G.get_edge_data(*raw_a) if raw_a[2] is None \
                            else G.get_edge_data(*raw_a[:3])
                    data_b = G.get_edge_data(*raw_b) if raw_b[2] is None \
                            else G.get_edge_data(*raw_b[:3])

                    if "pts2" in data_a and "pts2" in data_b:
                        for p in data_a["pts2"]:
                            angles.append(angle_between(pos_w, p, origin=pos_shared))
                        for p in data_b["pts2"]:
                            angles.append(angle_between(pos_v, p, origin=pos_shared))

                    attr["angle"] = np.asarray(angles, dtype=np.float32)

                # finally: add the line-graph edge
                L.add_edge(c_a, c_b, **attr)

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