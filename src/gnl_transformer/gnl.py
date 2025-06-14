import torch
import torch.nn.functional as F
from torch.nn import Linear, GRUCell
from torch_geometric.nn import aggr, GATv2Conv, TransformerConv, SAGPooling, MLP

from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.typing import Metadata, NodeType, EdgeType

class AttentiveGnLConv(torch.nn.Module):
    """A multi-layer graph convolutional network with attention and GRU updates.

    This module applies a sequence of graph convolutions, where each convolution
    is followed by a Gated Recurrent Unit (GRU) update. The first layer uses a
    `TransformerConv`, while subsequent layers use `GATv2Conv`. The outputs
    from each layer's GRU update are summed to produce the final output,
    creating a hierarchical representation.

    Args:
        in_channels (int): The dimensionality of input node features.
        hidden_channels (int): The dimensionality of hidden features.
        num_layers (int): The total number of convolutional layers.
        num_heads (int, optional): The number of multi-head-attention heads.
            Defaults to 4.
        edge_dim (int, optional): The dimensionality of edge features. If -1,
            it is inferred in a lazy manner. Defaults to -1.
        dropout (float, optional): Dropout probability for the attention
            weights. Defaults to 0.0.
        conv_kwargs (dict, optional): Additional arguments passed to the
            convolutional layers. Defaults to {}.
    """
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_heads: Optional[int] = 4,
        edge_dim: Optional[int] = -1,
        dropout: Optional[float] = 0.,
        conv_kwargs: Optional[dict] = {},
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.dropout = dropout

        if num_heads == 1:
            self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim,
                                   dropout=dropout, **conv_kwargs)
            self.gru1 = GRUCell(hidden_channels, in_channels)
            self.lin1 = Linear(in_channels, hidden_channels)
        elif num_heads >= 2:
            self.conv1 = TransformerConv(in_channels, hidden_channels//2, edge_dim=edge_dim,
                                   heads=num_heads, dropout=dropout, **conv_kwargs)
            self.lin0 = Linear(in_channels, hidden_channels*(num_heads//2))
            self.gru1 = GRUCell((hidden_channels//2)*num_heads, hidden_channels*(num_heads//2))
            self.lin1 = Linear(hidden_channels*(num_heads//2), hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim,
                                        add_self_loops=False, dropout=dropout, **conv_kwargs))
            self.grus.append(GRUCell(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.gru1.reset_parameters()
        self.lin0.reset_parameters() if hasattr(self, 'lin0') else None
        self.lin1.reset_parameters()
        for conv, gru in zip(self.convs, self.grus):
            conv.reset_parameters()
            gru.reset_parameters()

    def forward(self, 
                x: Tensor,
                edge_index: Tensor, 
                edge_attr: Tensor
            ) -> Tensor:
        """Forward pass for the AttentiveGnLConv module.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The graph connectivity in COO format.
            edge_attr (torch.Tensor): The edge features.

        Returns:
            torch.Tensor: The updated node features after hierarchical aggregation.
        """
        # Atom Embedding:
        if self.num_heads == 1:
            h = F.elu_(self.conv1(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.gru1(h, x).relu_()
            x = F.leaky_relu_(self.lin1(x))
        elif self.num_heads >= 2:
            h = F.elu_(self.conv1(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = F.leaky_relu_(self.lin0(x))
            x = self.gru1(h, x).relu_()
            x = F.leaky_relu_(self.lin1(x))
        g = [x]

        for conv, gru in zip(self.convs, self.grus):
            h = F.elu(conv(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
            g.append(x)

        return sum(g)   # sum hierarchical node embeddings

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'num_heads={self.num_heads}, '
                f'dropout={self.dropout})'
                f')')


class GnLTransformer_Paired(torch.nn.Module):
    """A dual-channel graph transformer model for paired graph data.

    This model processes a standard graph (G) and its corresponding line graph
    (L) in parallel channels. Each channel consists of an `AttentiveGnLConv`
    module, followed by `SAGPooling` and `SortAggregation`. The resulting
    graph-level embeddings from both channels are concatenated and passed
    through a final MLP to produce the output.

    This class expects two separate graph data objects in its forward pass.

    Args:
        dim_in_G (int): Input feature dimension for the graph (G) nodes.
        dim_in_L (int): Input feature dimension for the line graph (L) nodes.
        dim_h_conv (int): Hidden dimension for the `AttentiveGnLConv` layers.
        dim_h_lin (int): Hidden dimension for the final MLP layers.
        dim_out (int): The final output dimension of the model.
        num_layer_conv (int): The number of layers in each `AttentiveGnLConv`
            module.
        num_layer_lin (int): The number of layers in the final MLP.
        num_heads (int, optional): The number of attention heads in the
            convolutional layers. Defaults to 4.
        pool_k_G (int, optional): The number of nodes to keep for graph G after
            pooling. Defaults to 30.
        pool_k_L (int, optional): The number of nodes to keep for graph L after
            pooling. Defaults to 30.
        dropout (float, optional): Dropout probability used throughout the
            model. Defaults to 0.0.
    """
    def __init__(self,
        dim_in_G: int,
        dim_in_L: int,
        dim_h_conv: int,
        dim_h_lin: int,
        dim_out: int,
        num_layer_conv: int, 
        num_layer_lin: int,
        num_heads: Optional[int] = 4,
        pool_k_G: Optional[int] = 30,
        pool_k_L: Optional[int] = 30,
        dropout: Optional[float] = 0.,
        edge_dim_L: Optional[Union[int, float]] = -1,
    ):
        super().__init__()
        self.conv_G = AttentiveGnLConv(in_channels=dim_in_G,
                                hidden_channels=dim_h_conv,
                                num_layers=num_layer_conv,
                                num_heads=num_heads,
                                dropout=dropout,
                                edge_dim=dim_in_L)
        self.conv_L = AttentiveGnLConv(in_channels=dim_in_L,
                                hidden_channels=dim_h_conv,
                                num_layers=num_layer_conv,
                                num_heads=num_heads,
                                dropout=dropout,
                                edge_dim=edge_dim_L)

        self.pool_G = SAGPooling(dim_h_conv, ratio=pool_k_G)#, GNN=GATv2Conv)
        self.pool_L = SAGPooling(dim_h_conv, ratio=pool_k_L)#, GNN=GATv2Conv)
        self.sort_G = aggr.SortAggregation(k=pool_k_G)
        self.sort_L = aggr.SortAggregation(k=pool_k_L)

        self.mlp = MLP(in_channels=dim_h_conv*(pool_k_G+pool_k_L),
                       hidden_channels=dim_h_lin,
                       out_channels=dim_out,
                       num_layers=num_layer_lin,
                       dropout=dropout)
        
        self.hps = {
            'dim_in_G': dim_in_G,
            'dim_in_L': dim_in_L,
            'dim_h_conv': dim_h_conv,
            'dim_h_lin': dim_h_lin,
            'dim_out': dim_out,
            'num_layer_conv': num_layer_conv,
            'num_layer_lin': num_layer_lin,
            'num_heads': num_heads,
            'pool_k_G': pool_k_G,
            'pool_k_L': pool_k_L,
            'dropout': dropout,
        }
        [setattr(self, k, v) for k, v in self.hps.items()]

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_G.reset_parameters()
        self.conv_L.reset_parameters()
        self.pool_G.reset_parameters()
        self.pool_L.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data_G, data_L):
        """Forward pass for the GnLTransformer_Paired model.

        Args:
            data_G (torch_geometric.data.Data): The data object for the
                primary graph, containing `x`, `edge_index`, `edge_attr`,
                and `batch` attributes.
            data_L (torch_geometric.data.Data): The data object for the
                line graph, containing `x`, `edge_index`, `edge_attr`, and
                `batch` attributes.

        Returns:
            torch.Tensor: The final output tensor from the MLP.
        """
        x_G, edge_index_G, edge_attr_G, batch_G = \
            data_G.x, data_G.edge_index, data_G.edge_attr, data_G.batch
        x_L, edge_index_L, edge_attr_L, batch_L = \
            data_L.x, data_L.edge_index, data_L.edge_attr, data_L.batch
        x_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L = self.conv_L(x_L, edge_index_L, edge_attr_L)

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)

        return x
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'[({self.dim_in_G}, {self.dim_in_L}), {self.dim_out}], '
                f'dim_h_conv={self.dim_h_conv}, '
                f'dim_h_lin={self.dim_h_lin}, '
                f'num_layer_conv={self.num_layer_conv}, '
                f'num_layer_lin={self.num_layer_lin}, '
                f'num_heads={self.num_heads}, '
                f'pool_k=({self.pool_k_G}, {self.pool_k_L}), '
                f')')


class GnLTransformer_Hetero(GnLTransformer_Paired):
    """A dual-channel graph transformer for heterogeneous graph data.

    This model modifies `GnLTransformer_Paired` to work with a single
    `HeteroData` object. It processes two types of nodes and edges:
    - Graph (G): node_type='node', edge_type=('node', 'n2n', 'node')
    - Line Graph (L): node_type='edge', edge_type=('edge', 'e2e', 'edge')

    The architecture mirrors `GnLTransformer_Paired`: parallel G and L channels
    with `AttentiveGnLConv`, pooling, and sorting, followed by concatenation and
    a final MLP.

    Args:
        dim_in_G (int): Input feature dimension for 'node' type nodes.
        dim_in_L (int): Input feature dimension for 'edge' type nodes.
        dim_h_conv (int): Hidden dimension for the `AttentiveGnLConv` layers.
        dim_h_lin (int): Hidden dimension for the final MLP layers.
        dim_out (int): The final output dimension of the model.
        num_layer_conv (int): The number of layers in each `AttentiveGnLConv`
            module.
        num_layer_lin (int): The number of layers in the final MLP.
        num_heads (int, optional): The number of attention heads. Defaults to 4.
        pool_k_G (int, optional): The number of 'node' type nodes to keep after
            pooling. Defaults to 30.
        pool_k_L (int, optional): The number of 'edge' type nodes to keep after
            pooling. Defaults to 30.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        """Forward pass using dictionaries from a `HeteroData` object.

        Args:
            x_dict (Dict[NodeType, Tensor]): A dictionary mapping node types
                ('node', 'edge') to their feature tensors.
            edge_index_dict (Dict[EdgeType, Tensor]): A dictionary mapping edge
                types to their connectivity tensors.
            edge_attr_dict (Dict[EdgeType, Tensor]): A dictionary mapping edge
                types to their feature tensors.
            batch_dict (Dict[NodeType, Tensor]): A dictionary mapping node types
                to their batch index tensors.

        Returns:
            torch.Tensor: The final output tensor from the MLP.
        """
        x_G, edge_index_G, edge_attr_G, batch_G = \
            x_dict['node'], edge_index_dict[('node', 'n2n', 'node')], \
                edge_attr_dict[('node', 'n2n', 'node')], batch_dict['node']
        x_L, edge_index_L, edge_attr_L, batch_L = \
            x_dict['edge'], edge_index_dict[('edge', 'e2e', 'edge')], \
                edge_attr_dict[('edge', 'e2e', 'edge')], batch_dict['edge']

        x_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L = self.conv_L(x_L, edge_index_L, edge_attr_L)

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)

        return x


# X... classes for visualization and explanation
# cpu only, attention summaries are returned on CPU
class XAGnLConv(AttentiveGnLConv):
    """An explainable AttentiveGnLConv layer that captures intermediate values.

    This class extends `AttentiveGnLConv` to store and return intermediate
    attention weights and feature representations from each layer. This data
    is collected for visualization and model interpretability.

    The core architecture and arguments are identical to `AttentiveGnLConv`.
    """
    def forward(self, x, edge_index, edge_attr):
        """Forward pass that returns embeddings and visualization data.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The graph connectivity in COO format.
            edge_attr (torch.Tensor): The edge features.

        Returns:
            Tuple[torch.Tensor, Dict[str, Tensor]]: A tuple containing:
            - The final node embeddings after hierarchical aggregation.
            - A dictionary (`vis_data`) with intermediate representations,
              including attention weights (`att_w_conv_*`), post-convolution
              features (`x_conv_*`), and post-GRU features (`x_gru_*`).
        """
        vis_data = {}
        if self.num_heads == 1:
            h, att = self.conv1(x, edge_index, edge_attr, 
                                return_attention_weights=True)
            # vis_data['att_eid_conv_1'] = att[0].detach().cpu().numpy()
            vis_data['att_w_conv_1'] = att[1].detach().cpu().numpy()
            h = F.elu_(h)
            vis_data['x_conv_1'] = h.clone().squeeze().detach().cpu().numpy()
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.gru1(h, x).relu_()
            x = self.lin1(x)
            x = F.leaky_relu_(x)
            vis_data['x_gru_1'] = x.clone().squeeze().detach().cpu().numpy()
        elif self.num_heads >= 2:
            h, att = self.conv1(x, edge_index, edge_attr,
                                return_attention_weights=True)
            # vis_data['att_eid_conv_1'] = att[0].detach().cpu().numpy()
            vis_data['att_w_conv_1'] = att[1].detach().cpu().numpy()
            h = F.elu_(h)
            vis_data['x_conv_1'] = h.clone().squeeze().detach().cpu().numpy()
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = F.leaky_relu_(self.lin0(x))
            x = self.gru1(h, x).relu_()
            x = self.lin1(x)
            x = F.leaky_relu_(x)
            vis_data['x_gru_1'] = x.clone().squeeze().detach().cpu().numpy()
        g = [x]

        for i, (conv, gru) in enumerate(zip(self.convs, self.grus)):
            h, att = conv(x, edge_index, edge_attr, 
                          return_attention_weights=True)
            # vis_data[f'att_eid_conv_{i+2}'] = att[0].detach().cpu().numpy()
            vis_data[f'att_w_conv_{i+2}'] = att[1].detach().cpu().numpy()
            h = F.elu(h)
            vis_data[f'x_conv_{i+2}'] = h.clone().squeeze().detach().cpu().numpy()
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x)
            x = F.relu(x)
            vis_data[f'x_gru_{i+2}'] = x.clone().squeeze().detach().cpu().numpy()
            g.append(x)
            # sum hierarchical node embeddings
        g = sum(g)
        return g, vis_data


class XGnLTransformer_Paired(GnLTransformer_Paired):
    """An explainable dual-channel graph transformer for paired graph data.

    This model is a variant of `GnLTransformer_Paired` that uses `XAGnLConv`
    as its convolutional layer. It is designed to capture and return
    intermediate feature representations and attention weights from both the
    graph (G) and line graph (L) channels for explainability purposes.

    The architecture and arguments are identical to `GnLTransformer_Paired`.
    
    Args:
        dim_in_G (int): Input feature dimension for graph G.
        dim_in_L (int): Input feature dimension for graph L.
        dim_h_conv (int): Hidden dimension for the `XAGnLConv` layers.
        dim_h_lin (int): Hidden dimension for the MLP layers.
        dim_out (int): Output dimension of the model.
        num_layer_conv (int): Number of `XAGnLConv` layers per channel.
        num_layer_lin (int): Number of layers in the final MLP.
        num_heads (int): Number of attention heads.
        pool_k_G (int): Number of nodes to keep for graph G after pooling.
        pool_k_L (int): Number of nodes to keep for graph L after pooling.
        dropout (float, optional): Dropout rate. Defaults to 0.0.    
    """
    def __init__(self, dim_in_G, dim_in_L, dim_h_conv, dim_h_lin, dim_out, 
            num_layer_conv, num_layer_lin, num_heads, pool_k_G, pool_k_L, dropout=0.):
        super().__init__(dim_in_G, dim_in_L, dim_h_conv, dim_h_lin, dim_out, 
                num_layer_conv, num_layer_lin, num_heads, pool_k_G, pool_k_L, dropout)
        self.conv_G = XAGnLConv(dim_in_G, dim_h_conv, num_layer_conv, num_heads, dropout)
        self.conv_L = XAGnLConv(dim_in_L, dim_h_conv, num_layer_conv, num_heads, dropout)
    
    def forward(self, data_G, data_L):
        """
        Forward pass for the XGnLTransformer_Paired model.

        Processes the input graph data for G and L through their respective
        XAGnLConv branches, applies pooling and sorting, concatenates the
        resulting graph-level representations, and passes them through an MLP.
        Intermediate activations are collected for visualization.

        Args:
            data_G (torch_geometric.data.Data): Graph data object for graph G.
                Must contain `x` (node features), `edge_index` (connectivity),
                `edge_attr` (edge features), and `batch` (batch assignment for nodes).
            data_L (torch_geometric.data.Data): Graph data object for graph L.
                Must contain `x` (node features), `edge_index` (connectivity),
                `edge_attr` (edge features), and `batch` (batch assignment for nodes).

        Returns:
            Tuple[torch.Tensor, dict]:
                - x (torch.Tensor): The final output tensor from the MLP,
                representing the combined features of the paired graphs.
                - vis_data (dict): A dictionary containing intermediate tensors
                for visualization. Keys include:
                    'G_node': Visualization data from the G branch's XAGnLConv.
                    'L_node': Visualization data from the L branch's XAGnLConv.
                    'G_graph_1': Graph G features after pooling and sorting.
                    'L_graph_1': Graph L features after pooling and sorting.
                    'GnL_graph_-1': Combined G and L features before the final
                                    output of the MLP.
        """
        vis_data = {}
        x_G, edge_index_G, edge_attr_G, batch_G = data_G.x, data_G.edge_index, data_G.edge_attr, data_G.batch
        x_L, edge_index_L, edge_attr_L, batch_L = data_L.x, data_L.edge_index, data_L.edge_attr, data_L.batch
        x_G, vis_data_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L, vis_data_L = self.conv_L(x_L, edge_index_L, edge_attr_L)
        vis_data['G_node'] = vis_data_G
        vis_data['L_node'] = vis_data_L

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)
        vis_data['G_graph_1'] = x_G.clone().squeeze().detach().cpu().numpy()
        vis_data['L_graph_1'] = x_L.clone().squeeze().detach().cpu().numpy()

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)
        vis_data['GnL_graph_-1'] = x.clone().squeeze().detach().cpu().numpy()
        return x, vis_data


class XGnLTransformer_Hetero(GnLTransformer_Hetero):
    """An explainable dual-channel transformer for heterogeneous graphs.

    This model is a variant of `GnLTransformer_Hetero` that uses `XAGnLConv`
    as its convolutional layer. It processes a single `HeteroData` object
    and returns both the final prediction and a dictionary of intermediate
    activations and attention weights for visualization and analysis.

    Args:
        dim_in_G (int): Input feature dimension for graph G.
        dim_in_L (int): Input feature dimension for graph L.
        dim_h_conv (int): Hidden dimension for the XAGnLConv layers.
        dim_h_lin (int): Hidden dimension for the linear layers (e.g., in the MLP).
        dim_out (int): Output dimension of the model.
        num_layer_conv (int): Number of XAGnLConv layers for each branch (G and L).
        num_layer_lin (int): Number of linear layers in the final MLP.
        num_heads (int): Number of attention heads in the XAGnLConv layers.
        pool_k_G (float or int): Pooling ratio or number of nodes to keep after
                             convolution for graph G.
        pool_k_L (float or int): Pooling ratio or number of nodes to keep after
                             convolution for graph L.
        dropout (float, optional): Dropout rate to be used in XAGnLConv and
                               potentially other layers. Defaults to 0.0.
    """
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        """
        Forward pass for the XGnLTransformer_Hetero model.

        Processes the input HeteroData through the graph and line graph channels,
        applies XAGnLConv, pooling, sorting, concatenation, and passes through
        an MLP. Collects intermediate activations for visualization.

        Args:
            x_dict (Dict[str, torch.Tensor]): Dictionary containing node features
                for both graph and line graph. Keys should be 'node' for graph G
                and 'edge' for line graph L.
            edge_index_dict (Dict[EdgeType, torch.Tensor]): Dictionary containing
                edge indices for both graph and line graph. Keys should be:
                - ('node', 'n2n', 'node') for graph edges
                - ('edge', 'e2e', 'edge') for line graph edges
            edge_attr_dict (Dict[EdgeType, torch.Tensor]): Dictionary containing
                edge attributes for both graph and line graph.
            batch_dict (Dict[str, torch.Tensor]): Dictionary containing batch
                assignments for nodes in both graph and line graph. Keys should be:
                - 'node' for graph G
                - 'edge' for line graph L

        Returns:
            Tuple[torch.Tensor, dict]:
                - x (torch.Tensor): The final output tensor from the MLP,
                representing the combined features of the paired graphs.
                - vis_data (dict): A dictionary containing intermediate tensors
                for visualization. Keys include:
                    'G_node': Visualization data from the G branch's XAGnLConv.
                    'L_node': Visualization data from the L branch's XAGnLConv.
                    'G_graph_1': Graph G features after pooling and sorting.
                    'L_graph_1': Graph L features after pooling and sorting.
                    'GnL_graph_-1': Combined G and L features before the final
                                    output of the MLP.
        """
        vis_data = {}
        x_G, edge_index_G, edge_attr_G, batch_G = x_dict['node'], edge_index_dict[('node', 'n2n', 'node')], edge_attr_dict[('node', 'n2n', 'node')], batch_dict['node']
        x_L, edge_index_L, edge_attr_L, batch_L = x_dict['edge'], edge_index_dict[('edge', 'e2e', 'edge')], edge_attr_dict[('edge', 'e2e', 'edge')], batch_dict['edge']

        x_G, vis_data_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L, vis_data_L = self.conv_L(x_L, edge_index_L, edge_attr_L)
        vis_data['G_node'] = vis_data_G
        vis_data['L_node'] = vis_data_L

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)
        vis_data['G_graph_1'] = x_G.clone().squeeze().detach().cpu().numpy()
        vis_data['L_graph_1'] = x_L.clone().squeeze().detach().cpu().numpy()

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)
        vis_data['GnL_graph_-1'] = x.clone().squeeze().detach().cpu().numpy()
        
        return x, vis_data