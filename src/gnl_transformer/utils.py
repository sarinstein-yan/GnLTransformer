import numpy as np
import networkx as nx
from functools import partial
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
    with_triplet_features: bool = False,
    create_using: Optional[Any] = None,
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
        Add `pos`, `joint_node_attr`, `triplet_center` and `angle` features.
    create_using : graph constructor, optional
        Graph type for the output (default → `G.__class__`).
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
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Graph specific functions for edges.
    get_edges = partial(G.edges, keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    
    # Determine if we include self-loops or not.
    shift = 0 if add_selfloops else 1

    # Introduce numbering of nodes
    node_index = {n: i for i, n in enumerate(G)}

    # Lift canonical representation of nodes to edges in line graph
    edge_key_function = lambda edge: (node_index[edge[0]], node_index[edge[1]])

    edges = set()
    for u in G:
        # Label nodes as a sorted tuple of nodes in original graph.
        # Decide on representation of {u, v} as (u, v) or (v, u) depending on node_index.
        # -> This ensures a canonical representation and avoids comparing values of different types.
        nodes = [tuple(sorted(x[:2], key=node_index.get)) + (x[2],) for x in get_edges(u)]

        if len(nodes) == 1:
            # Then the edge will be an isolated node in L.
            edge = nodes[0]
            canonical_edge = (min(edge[0], edge[1]), max(edge[0], edge[1]), edge[2])
            data = G.get_edge_data(*edge[:3])
            L.add_node(canonical_edge, **data)
            if 'pts5' in data:
                L.nodes[canonical_edge]['pos'] = data['pts5'][4:6]

        for i, a in enumerate(nodes):

            canonical_a = (min(a[0], a[1]), max(a[0], a[1]), a[2])
            data_a = G.get_edge_data(*a[:3])
            L.add_node(canonical_a, **data_a)  # Transfer edge attributes to node
            if 'pts5' in data_a:
                L.nodes[canonical_a]['pos'] = data_a['pts5'][4:6]
            
            for b in nodes[i + shift:]:
                canonical_b = (min(b[0], b[1]), max(b[0], b[1]), b[2])
                data_b = G.get_edge_data(*b[:3])
                edge = tuple(sorted((canonical_a, canonical_b), key=edge_key_function))

                if edge not in edges:
                    # find the common node u. TODO: modify for self-loops
                    u = set(a[:2]).intersection(set(b[:2])).pop()

                    # optional triplet-level geometric features
                    attr = {}
                    if with_triplet_features:
                        attr['joint_node_attr'] = G.nodes[u]
                        v = a[0] if a[0] != u else a[1]
                        w = b[0] if b[0] != u else b[1]
                        pos_shared = attr['joint_node_attr']['pos']
                        pos_v      = G.nodes[v]["pos"]
                        pos_w      = G.nodes[w]["pos"]
                        
                        # Calculate the center of the triplet
                        attr['triplet_center'] = np.mean([pos_v, pos_w, pos_shared], axis=0)
                        
                        # angles:   ∠(v, w) plus those with intermediate pts2’s
                        angles = [angle_between(pos_v, pos_w, origin=pos_shared)]

                        if 'pts2' in data_a and 'pts2' in data_b:
                            for p in data_a['pts2']:
                                angles.append(angle_between(pos_w, p, origin=pos_shared))
                            for p in data_b['pts2']:
                                angles.append(angle_between(pos_v, p, origin=pos_shared))
                        attr['angle'] = np.array(angles, dtype=np.float32)
                    L.add_edge(canonical_a, canonical_b, **attr)
                    edges.add(edge)
                    # print(f"Added edge: {canonical_a} -> {canonical_b} with attributes {attr}") # Debugging
    
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