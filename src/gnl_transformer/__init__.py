__version__ = '0.0.1'

__description__ = "GnLTransformer: A Graph Transformer for Non-Hermitian Spectral Graph Representation Learning. Leveraging parallel line graph channels for improved performance and higher-order topology explanability."
    
__all__ = [
    'free_coefficients_sampler',
    'remove_reciprocal_polynomials',
    'hash_multilabels',
    'generate_dataset_in_coeff_hypercube',

    'NHSG117K',

    'AttentiveGnLConv',
    'GnLTransformer_Paired',
    'GnLTransformer_Hetero',
    'XAGnLConv',
    'XGnLTransformer_Paired',
    'XGnLTransformer_Hetero',

    'line_graph_undirected',
    'isomorphism_classes_by_WL_hash',
]

from gnl_transformer.dataset_raw import (
    free_coefficients_sampler,
    remove_reciprocal_polynomials,
    hash_multilabels,
    generate_dataset_in_coeff_hypercube,
)

from gnl_transformer.dataset_pyg import NHSG117K

from gnl_transformer.gnl import (
    AttentiveGnLConv,
    GnLTransformer_Paired,
    GnLTransformer_Hetero,
    XAGnLConv,
    XGnLTransformer_Paired,
    XGnLTransformer_Hetero,
)

from gnl_transformer.utils import (
    line_graph_undirected,
    isomorphism_classes_by_WL_hash,
)

# from gnl_transformer.models import (
#     BasicGNNBaselines,
#     MF,
#     AFP,
#     GnLTransformer_ablation,
#     GnLTransformer,
# )

from gnl_transformer.training import (
    NHSG117K_Lit,
    LitGNN,
    summarise_csv,
    run_experiment,
)

# from .explain_gnl import (
# normalize_color,
# visualize_attention_scores,
# visualize_node_embeddings,
# ExplanationSummary
# )