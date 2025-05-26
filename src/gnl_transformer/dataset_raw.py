import os, time, pickle
import numpy as np
import sympy as sp
import networkx as nx
import itertools as it
from joblib import Parallel, delayed
import poly2graph as p2g
from typing import Sequence, Any
from numpy.typing import ArrayLike


def free_coefficients_sampler(samples_per_dim=7, dim=6, c_max=1.2):
    """Generate an array of coefficient combinations for polynomials.

    Creates a grid of points in a `dim`-dimensional space, where each
    dimension ranges from `-c_max` to `c_max` with `samples_per_dim`
    equally spaced points. Each point in the grid represents a list
    of coefficients.

    Parameters
    ----------
    samples_per_dim : int, optional
        Number of sample points to generate for each dimension (coefficient).
        Default is 7.
    dim : int, optional
        Dimensionality of the coefficient space, i.e., the number of
        coefficients in each polynomial. Default is 6.
    c_max : float, optional
        Maximum absolute value for each coefficient. The range of values
        for each coefficient will be `[-c_max, c_max]`. Default is 1.2.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row is a list of coefficients.
        The shape of the array is `(samples_per_dim**dim, dim)`.
    """
    values = list(np.linspace(-c_max, c_max, samples_per_dim))
    combinations = list(it.product(values, repeat=dim))
    return np.asarray(combinations)


def remove_reciprocal_polynomials(
        coeffs: ArrayLike,
        filter: callable = min,
    ) -> list[list]:
    """Filters a list of coefficient lists to remove reciprocal duplicates.

    Reciprocal polynomials are identified by comparing their binarized form
    (non-zero elements become 1, zero elements remain 0) with its reverse.
    The `filter` function determines which of the pair (original or reversed)
    is kept.

    Parameters
    ----------
    coeffs : ArrayLike
        A 2D array-like structure where each row represents the coefficients
        of a polynomial.
    filter : callable, optional
        A function that takes two lists (the binarized coefficients and its
        reverse) and returns the one to be kept as the canonical key.
        Default is `min`, which keeps the lexicographically smaller key.

    Returns
    -------
    list[list]
        An array of keys, where each key represents a unique polynomial (or its
        reciprocal pair) based on the binarized coefficients.
    """
    coeffs = np.asarray(coeffs)
    # Binarize coefficients to create a key for hashing
    binarized = (coeffs != 0).astype(int).tolist()
    # De-duplicate reciprocal polynomials
    keys = [filter(b, b[::-1]) for b in binarized]
    return keys

def hash_multilabels(
    multilabels: ArrayLike,
    base: int,
    reindex: bool = False
) -> np.ndarray:
    """Hash multi-label arrays to unique integer identifiers.

    Treats each row of `multilabels` (a label list) as digits of a number
    in a specified `base` and converts it to its decimal representation.
    Optionally, these hash values can be reindexed to form a contiguous
    range of integers starting from 0.

    Parameters
    ----------
    multilabels : np.ndarray
        A 2D numpy array of integers where each row is a list of labels
        to be hashed. Shape `(n_samples, dim)`.
    base : int
        The base to use for the hashing process. This is akin to the 'n'
        in "base-n" representation.
    reindex : bool, optional
        If True, the resulting hash values are mapped to a new set of
        integers starting from 0 up to `m-1`, where `m` is the number of
        unique hash values. Default is False.

    Returns
    -------
    np.ndarray
        A 1D numpy array containing the hashed integer values for each
        input multi-label. If `reindex` is True, these are the
        reindexed values.

    Raises
    ------
    AssertionError
        If `multilabels` is not a 2D array.
    """
    multilabels = np.asarray(multilabels)
    assert multilabels.ndim == 2, "labels must be 2D array"
    dim = multilabels.shape[1]
    base_vec = np.array([base**i for i in range(dim)])
    hash_value = base_vec @ multilabels.T
    if reindex:
        unique_hash = np.unique(hash_value)
        hash_map = {hash_val: i for i, hash_val in enumerate(unique_hash)}
        reassigned_hash_value = np.array([hash_map[val] for val in hash_value])
        return reassigned_hash_value
    else:
        return hash_value

def generate_dataset_in_coeff_hypercube(
    characteristic: Any 
        = "-E + z**(-4) + z**4"\
        + " + a0*z**(-3) + a1*z**(-2) + a2*z**(-1)"\
        + " + a3*z       + a4*z**2    + a5*z**3",
    params: Sequence[sp.Symbol] = sp.symbols('a:6'),
    param_walk: Sequence[float] = np.linspace(-1.2, 1.2, 7),
    save_dir: str = "raw",
    num_partition: int = 200,
    short_edge_threshold: int = 20,
    n_jobs: int = -1,
) -> None:
    """Generates a dataset of spectral graphs from a characteristic polynomial.

    This function systematically explores a hypercube of parameter values for
    a given characteristic polynomial. For each combination of parameters,
    it computes the corresponding spectral graph. The generated graphs and
    associated metadata are saved to disk.

    The process involves:
    1. Defining a characteristic polynomial with symbolic parameters.
    2. Generating all combinations of parameter values from `param_walk`.
    3. Splitting these combinations into partitions for batch processing.
    4. For each partition, computing spectral graphs in parallel.
    5. Saving serialized graphs and parameter values for each partition.
    6. Combining all partitions into a single NPZ file.
    7. Generating and saving metadata, including hashed labels for the
       polynomials.

    Parameters
    ----------
    characteristic : Any, optional
        A string representation of the characteristic polynomial.
        It should use 'E' for energy, 'z' for the phase factor
        (e.g., `exp(i*k)`), and symbolic parameters defined in `params`.
        Default is a specific polynomial string.
    params : Sequence[sp.Symbol], optional
        A sequence of SymPy symbols representing the free parameters in the
        characteristic polynomial. Default is `sp.symbols('a:6')`.
    param_walk : Sequence[float], optional
        A sequence of float values that each parameter in `params` will take.
        All combinations of these values will be explored.
        Default is `np.linspace(-1.2, 1.2, 7)`.
    save_dir : str, optional
        The directory where the generated dataset partitions and the final
        combined dataset will be saved. Default is "raw".
    num_partition : int, optional
        The number of partitions to split the parameter combinations into.
        This helps manage memory and allows for checkpointing.
        Default is 200.
    short_edge_threshold : int, optional
        Threshold used in `CharPolyClass.spectral_graph` to filter out
        short edges in the computed graphs. Default is 20.
    n_jobs : int, optional
        The number of jobs to run in parallel for graph generation and
        serialization. -1 means using all available processors.
        Default is -1.

    Returns
    -------
    None
        This function does not return any value but saves the dataset
        to disk.

    Side Effects
    ------------
    - Creates the directory specified by `save_dir` if it doesn't exist.
    - Saves temporary partition files (e.g., `part_0.npz`, `part_1.npz`, ...)
      in `save_dir` during processing. These are removed after the final
      dataset is created.
    - Saves the final combined dataset as `nx_multigraph_dataset.npz` in
      `save_dir`. This file contains:
        - `graphs_pickle`: Pickled NetworkX multigraph objects.
        - `y`: Hashed integer labels for unique polynomials (after removing
          reciprocals and reindexing).
        - `y_multi`: Multi-labels (binarized coefficients) after removing
          reciprocal duplicates.
        - `free_coefficient_lists`: The original array of parameter combinations.
        - Additional metadata derived from the characteristic polynomial.
    """
    k, z, E = sp.symbols('k z E')
    cp = p2g.CharPolyClass(characteristic, k, z, E, set(params))

    # Generate all combinations of coefficient lists
    param_arr = np.asarray(list(
        it.product(param_walk, repeat=len(params))
    ))
    print(f"Generated {len(param_arr)} coefficient combinations...")
    # Split the parameter array into partitions
    param_parts = np.array_split(param_arr, num_partition)

    os.makedirs(save_dir, exist_ok=True)

    batcher = Parallel(n_jobs=n_jobs, prefer='threads')
    for i, vals in enumerate(param_parts):
        print(f"Partition {i} - computing {len(vals)} graphs...")
        t0 = time.perf_counter()
        
        # Generate graphs for the current partition
        graphs, _ = cp.spectral_graph(
            {a: vals[:, j] for j, a in enumerate(params)},
            short_edge_threshold=short_edge_threshold,
        )

        # Save serialized graphs and their parameter values
        out = os.path.join(save_dir, f'part_{i}.npz')
        graphs_ser = batcher(
            delayed(pickle.dumps)(g) for g in graphs
        )
        np.savez_compressed(
            out,
            graphs_pickle=graphs_ser,
        )
        dt = (time.perf_counter() - t0) / 60
        print(f"Partition {i} saved → {out} ({len(graphs_ser)} graphs, took {dt:.4f} min)")

    # Combine all partitions into a single NPZ
    all_pickles = []
    for i in range(num_partition):
        d = np.load(
            os.path.join(save_dir, f'part_{i}.npz'), 
            allow_pickle=True
        )
        # .tolist() because np.savez loads object arrays as dtype=object
        all_pickles.extend(d['graphs_pickle'].tolist())
        os.remove(os.path.join(save_dir, f'part_{i}.npz'))
    
    # meta data
    y_multi = remove_reciprocal_polynomials(param_arr)
    y = hash_multilabels(y_multi, base=2, reindex=True)
    meta = {
        'latex': sp.latex(cp.ChP),
        'sympy_repr': sp.srepr(cp.ChP),
        'parameter_symbols': tuple(str(p) for p in params),
        'generator_symbols': tuple(str(g) for g in cp.ChP.gens),
        'number_of_bands': cp.num_bands,
        'max_left_hopping': cp.poly_q,
        'max_right_hopping': cp.poly_p,
    }
    
    out = os.path.join(save_dir, 'nx_multigraph_dataset.npz')
    np.savez_compressed(
        out,
        graphs_pickle=np.array(all_pickles, dtype=object),
        y=np.asarray(y),
        y_multi=np.asarray(y_multi),
        free_coefficient_lists=param_arr,
        **meta,
    )
    print(f"The full dataset saved → {out}")

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

def unique_indices_by_hash(graphs, iters=3, n_jobs=-1):
    hashes = Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(wl_hash_safe)(g, iters=iters) for g in graphs
    )
    first_seen = {}
    for idx, h in enumerate(hashes):
        first_seen.setdefault(h, idx)
    return list(first_seen.values())


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(
        description="Generate the dataset of spectral graphs used in the companion paper."
    )
    # parser.add_argument(
    #     "--characteristic", type=str, default="-E + z**(-4) + z**4"\
    #         + " + a0*z**(-3) + a1*z**(-2) + a2*z**(-1)"\
    #         + " + a3*z       + a4*z**2    + a5*z**3",
    #     help="Characteristic polynomial to use for generating graphs."
    # )
    # parser.add_argument(
    #     "--params", type=str, default="a:6",
    #     help="SymPy symbols for the polynomial parameters (e.g., 'a:6' for a0 to a5)."
    # )
    parser.add_argument(
        "--param_walk", type=float, nargs='+', default=np.linspace(-1.2, 1.2, 7).tolist(),
        help="Values for the parameters to explore in the hypercube."
    )
    parser.add_argument(
        "--save_dir", type=str, default="raw",
        help="Directory to save the generated dataset."
    )
    parser.add_argument(
        "--num_partition", type=int, default=200,
        help="Number of partitions to split the parameter combinations into."
    )
    parser.add_argument(
        "--short_edge_threshold", type=int, default=20,
        help="Threshold for filtering short edges in the spectral graph."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of parallel jobs to run for graph generation. -1 means all available processors."
    )
    args = parser.parse_args()

    # Example usage
    print(f"Compute in {args.num_partition} partitions with a {args.n_jobs}-thread worker...")
    print(f"Parameter walk: {args.param_walk}")
    print(f"Saving to directory: {Path(args.save_dir).expanduser().resolve()}")
    print(f"Short edge threshold: {args.short_edge_threshold}")

    generate_dataset_in_coeff_hypercube(
        param_walk=args.param_walk,
        save_dir=args.save_dir,
        num_partition=args.num_partition,
        short_edge_threshold=args.short_edge_threshold,
        n_jobs=args.n_jobs
    )