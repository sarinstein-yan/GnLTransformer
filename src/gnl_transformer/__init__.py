__version__ = '0.0.1'

__description__ = "GnLTransformer: A Graph Transformer for Non-Hermitian Spectral Graph Representation Learning. Leveraging parallel line graph channels for improved performance and higher-order topology explanability."
    
__all__ = [
    'free_coefficients_sampler',
    'remove_reciprocal_polynomials',
    'hash_multilabels',
    'generate_dataset_in_coeff_hypercube',

    'line_graph_undirected',
    'isomorphism_classes_by_WL_hash',
]

from gnl_transformer.dataset_raw import (
    free_coefficients_sampler,
    remove_reciprocal_polynomials,
    hash_multilabels,
    generate_dataset_in_coeff_hypercube,
)

from gnl_transformer.utils import (
    line_graph_undirected,
    isomorphism_classes_by_WL_hash,
)

# from .GnLTransformer import (
# AttentiveGnLConv,
# GnLTransformer_Paired,
# GnLTransformer_Hetero,
# XAGnLConv,
# XGnLTransformer_Paired
# )

# from .explain_gnl import (
# normalize_color,
# visualize_attention_scores,
# visualize_node_embeddings,
# ExplanationSummary
# )

# from .in_memory_dataset import (Dataset_nHSG,
#                                 Dataset_nHSG_Paired,
#                                 Dataset_nHSG_Hetero)