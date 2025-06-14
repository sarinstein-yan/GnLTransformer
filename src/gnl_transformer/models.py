'''Adapted from PyTorch Geometric'''

import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Parameter, GRUCell
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
    MFConv,
)
from torch_geometric.nn import (
    aggr,
    global_add_pool, 
    Linear,
    SAGPooling,
)
from torch_geometric.nn.models import MLP, NeuralFingerprint, AttentiveFP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.utils._trim_to_layer import TrimToLayer

from gnl_transformer import (
    AttentiveGnLConv,
    # GnLTransformer_Paired,
    GnLTransformer_Hetero,
)

__all__ = [
    'GCN',
    'GraphSAGE',
    'GIN',
    'GINE',
    'GAT',
    'PNA',
    'EdgeCNN',
    'BasicGNN',
    'BasicGNNBaselines',
    'MF',
    'AFP',
    'GnLTransformer_ablation',
    'GnLTransformer',
]


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    supports_edge_weight: Final[bool]
    supports_edge_attr: Final[bool]
    supports_norm_batch: Final[bool]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = ModuleList()
        norm_layer = normalization_resolver(
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        )
        if norm_layer is None:
            norm_layer = torch.nn.Identity()

        self.supports_norm_batch = False
        if hasattr(norm_layer, 'forward'):
            norm_params = inspect.signature(norm_layer.forward).parameters
            self.supports_norm_batch = 'batch' in norm_params

        for _ in range(num_layers - 1):
            self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None:
            self.norms.append(copy.deepcopy(norm_layer))
        else:
            self.norms.append(torch.nn.Identity())

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

        # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        self._trim = TrimToLayer()

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if (not torch.jit.is_scripting()
                    and num_sampled_nodes_per_hop is not None):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, 'jk'):
                    xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x

        return x

    @torch.no_grad()
    def inference_per_layer(
        self,
        layer: int,
        x: Tensor,
        edge_index: Adj,
        batch_size: int,
    ) -> Tensor:

        x = self.convs[layer](x, edge_index)[:batch_size]

        if layer == self.num_layers - 1 and self.jk_mode is None:
            return x

        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norms is not None:
            x = self.norms[layer](x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        if layer == self.num_layers - 1 and hasattr(self, 'lin'):
            x = self.lin(x)

        return x

    @torch.no_grad()
    def inference(
        self,
        loader: NeighborLoader,
        device: Optional[Union[str, torch.device]] = None,
        embedding_device: Union[str, torch.device] = 'cpu',
        progress_bar: bool = False,
        cache: bool = False,
    ) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using a
        :class:`~torch_geometric.loader.NeighborLoader`, where
        :class:`~torch_geometric.loader.NeighborLoader` should sample the
        full neighborhood for only one layer.
        This is an efficient way to compute the output embeddings for all
        nodes in the graph.
        Only applicable in case :obj:`jk=None` or `jk='last'`.

        Args:
            loader (torch_geometric.loader.NeighborLoader): A neighbor loader
                object that generates full 1-hop subgraphs, *i.e.*,
                :obj:`loader.num_neighbors = [-1]`.
            device (torch.device, optional): The device to run the GNN on.
                (default: :obj:`None`)
            embedding_device (torch.device, optional): The device to store
                intermediate embeddings on. If intermediate embeddings fit on
                GPU, this option helps to avoid unnecessary device transfers.
                (default: :obj:`"cpu"`)
            progress_bar (bool, optional): If set to :obj:`True`, will print a
                progress bar during computation. (default: :obj:`False`)
            cache (bool, optional): If set to :obj:`True`, caches intermediate
                sampler outputs for usage in later epochs.
                This will avoid repeated sampling to accelerate inference.
                (default: :obj:`False`)
        """
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training
        # assert not loader.shuffle  # TODO (matthias) does not work :(
        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        x_all = loader.data.x.to(embedding_device)

        if cache:

            # Only cache necessary attributes:
            def transform(data: Data) -> Data:
                kwargs = dict(n_id=data.n_id, batch_size=data.batch_size)
                if hasattr(data, 'adj_t'):
                    kwargs['adj_t'] = data.adj_t
                else:
                    kwargs['edge_index'] = data.edge_index

                return Data.from_dict(kwargs)

            loader = CachedLoader(loader, device=device, transform=transform)

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                batch_size = batch.batch_size
                if hasattr(batch, 'adj_t'):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)

                x = self.inference_per_layer(i, x, edge_index, batch_size)
                xs.append(x.to(embedding_device))

                if progress_bar:
                    pbar.update(1)

            x_all = torch.cat(xs, dim=0)

        if progress_bar:
            pbar.close()

        return x_all

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, 
                       add_self_loops=False,
                       **kwargs)


class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)

class GINE(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, 
                        edge_dim=-1,
                        **kwargs)

class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, 
                    add_self_loops=False,
                    edge_dim=-1,
                    **kwargs)


class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, 
                       edge_dim=-1,
                       **kwargs)


class EdgeCNN(BasicGNN):
    r"""The Graph Neural Network from the `"Dynamic Graph CNN for Learning on
    Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper, using the
    :class:`~torch_geometric.nn.conv.EdgeConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.EdgeConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [2 * in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return EdgeConv(mlp, **kwargs)


class BasicGNNBaselines(torch.nn.Module):
    def __init__(self, base, dim_in, dim_h_conv, dim_h_lin, dim_out,
                 num_layers_conv, num_layers_lin, dropout=0., **kwargs):
        super().__init__()
        
        self.baseline = base(
            in_channels=dim_in, hidden_channels=dim_h_conv,
            num_layers=num_layers_conv, out_channels=dim_h_conv, 
            dropout=dropout, **kwargs
        )
        self.mlp = MLP(in_channels=dim_h_conv, hidden_channels=dim_h_lin,
            out_channels=dim_out, num_layers=num_layers_lin, dropout=dropout)
        
        self.hps = {
            'dim_in': dim_in,
            'dim_h_conv': dim_h_conv,
            'dim_h_lin': dim_h_lin,
            'dim_out': dim_out,
            'num_layers_conv': num_layers_conv,
            'num_layers_lin': num_layers_lin,
            'dropout': dropout,
        }
    
    def forward(self, batch):
        # x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = batch.x_dict['node']
        edge_index = batch.edge_index_dict[('node', 'n2n', 'node')]
        edge_attr = batch.edge_attr_dict[('node', 'n2n', 'node')]
        edge_weight = edge_attr[:, 0]
        batch = batch.batch_dict['node']

        h = self.baseline(x=x, edge_index=edge_index,
                          edge_weight=edge_weight, edge_attr=edge_attr)
        
        # Apply global pooling and MLP
        h = global_add_pool(h, batch)
        x = self.mlp(h)
        
        return x


class MF(torch.nn.Module):
    r"""Adapted from The Neural Fingerprint model in the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`__ paper to generate fingerprints
    of molecules.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output fingerprint.
        num_layers (int): Number of layers.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MFConv`.
    """
    def __init__(
        self,
        dim_in: int,
        dim_h_conv: int,
        dim_h_lin: int,
        dim_out: int,
        num_layers_conv: int,
        num_layers_lin: int = 1,
        dropout: float = 0.,
        **kwargs,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers_conv):
            dim_in = self.dim_in if i == 0 else self.dim_h_conv
            self.convs.append(MFConv(dim_in, dim_h_conv, **kwargs))

        self.lins = torch.nn.ModuleList()
        for _ in range(self.num_layers_conv):
            self.lins.append(Linear(dim_h_conv, dim_out, bias=False))
        
        self.mlp = MLP(in_channels=dim_h_conv, hidden_channels=dim_h_lin,
            out_channels=dim_out, num_layers=num_layers_lin, dropout=dropout)

        self.hps = {
            'dim_in': dim_in,
            'dim_h_conv': dim_h_conv,
            'dim_h_lin': dim_h_lin,
            'dim_out': dim_out,
            'num_layers_conv': num_layers_conv,
            'num_layers_lin': num_layers_lin,
            'dropout': dropout,
        }
        [setattr(self, k, v) for k, v in self.hps.items()]

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, batch):
        x = batch.x_dict['node']
        edge_index = batch.edge_index_dict[('node', 'n2n', 'node')]
        batch = batch.batch_dict['node']

        outs = []
        for conv, lin in zip(self.convs, self.lins):
            x = conv(x, edge_index).sigmoid()
            y = lin(x)#.softmax(dim=-1) # use logits to match other baselines
            outs.append(global_add_pool(y, batch))
        h = sum(outs)

        return self.mlp(h)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.dim_in}, '
                f'{self.dim_out}, num_layers={self.num_layers_conv})')

# class MF(torch.nn.Module):
#     def __init__(self, dim_in_G, dim_h_conv, dim_out, num_layers_conv, dropout=0.):
#         super().__init__()
#         self.mf = NeuralFingerprint(
#             in_channels=dim_in_G, hidden_channels=dim_h_conv,
#             out_channels=dim_out, num_layers=num_layers_conv)

#     def forward(self, batch):
#         x = batch.x_dict['node']
#         edge_index = batch.edge_index_dict[('node', 'n2n', 'node')]
#         batch = batch.batch_dict['node']
#         h = self.mf(x=x, edge_index=edge_index, batch=batch)
#         return h



class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        # propagate_type: (x: Tensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        dim_in: int,
        dim_h_conv: int,
        dim_h_lin: int,
        dim_out: int,
        edge_dim: int,
        num_layers_conv: int,
        num_layers_lin: int = 1,
        num_timesteps: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.lin1 = Linear(dim_in, dim_h_conv)

        self.gate_conv = GATEConv(dim_h_conv, dim_h_conv, edge_dim,
                                  dropout)
        self.gru = GRUCell(dim_h_conv, dim_h_conv)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers_conv - 1):
            conv = GATConv(dim_h_conv, dim_h_conv, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(dim_h_conv, dim_h_conv))

        self.mol_conv = GATConv(dim_h_conv, dim_h_conv,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(dim_h_conv, dim_h_conv)

        self.lin2 = MLP(in_channels=dim_h_conv, hidden_channels=dim_h_lin,
            out_channels=dim_out, num_layers=num_layers_lin, dropout=dropout)
        
        self.hps = {
            'dim_in': dim_in,
            'dim_h_conv': dim_h_conv,
            'dim_h_lin': dim_h_lin,
            'dim_out': dim_out,
            'num_layers_conv': num_layers_conv,
            'num_layers_lin': num_layers_lin,
            'dropout': dropout,
            'edge_dim': edge_dim,
            'num_timesteps': num_timesteps,
        }
        [setattr(self, k, v) for k, v in self.hps.items()]
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    # def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
    #             batch: Tensor) -> Tensor:
    def forward(self, batch):
        """"""  # noqa: D419
        x = batch.x_dict['node']
        edge_index = batch.edge_index_dict[('node', 'n2n', 'node')]
        edge_attr = batch.edge_attr_dict[('node', 'n2n', 'node')]
        batch = batch.batch_dict['node']

        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.dim_in}, '
                f'hidden_channels={self.dim_h_conv}, '
                f'out_channels={self.dim_out}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers_conv={self.num_layers_conv}, '
                f'num_timesteps={self.num_timesteps}, '
                f'num_layers_lin={self.num_layers_lin}, '
                f')')

# class AFP(torch.nn.Module):
#     def __init__(self, dim_in_G, dim_in_L, dim_h_conv, dim_out, num_layers_conv, dropout=0.):
#         super().__init__()
#         self.afp = AttentiveFP(
#             in_channels=dim_in_G, hidden_channels=dim_h_conv,
#             out_channels=dim_out, num_layers=num_layers_conv,
#             edge_dim=dim_in_L, num_timesteps=3, dropout=dropout)

#     def forward(self, batch):
#         x = batch.x_dict['node']
#         edge_index = batch.edge_index_dict[('node', 'n2n', 'node')]
#         edge_attr = batch.edge_attr_dict[('node', 'n2n', 'node')]
#         batch = batch.batch_dict['node']
#         h = self.afp(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
#         return h


class GnLTransformer_ablation(torch.nn.Module):
    """An ablation model of the GnLTransformer with only the graph channel.

    This model serves as an ablation study for the `GnLTransformer`. It removes
    the line-graph processing channel entirely. The architecture consists of a
    single `AttentiveGnLConv` stream for the primary graph, followed by
    `SAGPooling`, `SortAggregation`, and a final MLP for prediction.

    Args:
        dim_in_G (int): Input feature dimension for graph nodes.
        dim_h_conv (int): Hidden dimension for the `AttentiveGnLConv` layers.
        dim_h_lin (int): Hidden dimension for the MLP layers.
        dim_out (int): The final output dimension of the model.
        num_layer_conv (int): The number of layers in `AttentiveGnLConv`.
        num_layer_lin (int): The number of layers in the final MLP.
        num_heads (int): The number of attention heads.
        pool_k_G (int): The number of nodes to keep after pooling.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, dim_in_G, dim_h_conv, dim_h_lin, dim_out,
                 num_layer_conv, num_layer_lin, num_heads, pool_k_G, dropout=0.):
        super().__init__()
        self.conv_G = AttentiveGnLConv(
            in_channels=dim_in_G,
            hidden_channels=dim_h_conv,
            num_layers=num_layer_conv,
            num_heads=num_heads,
            dropout=dropout
        )
        self.pool_G = SAGPooling(dim_h_conv, ratio=pool_k_G)#, GNN=GATv2Conv)
        self.sort_G = aggr.SortAggregation(k=pool_k_G)
        self.mlp = MLP(
            in_channels=dim_h_conv * pool_k_G,
            hidden_channels=dim_h_lin,
            out_channels=dim_out,
            num_layers=num_layer_lin,
            dropout=dropout
        )

        self.hps = {
            'dim_in_G': dim_in_G,
            'dim_h_conv': dim_h_conv,
            'dim_h_lin': dim_h_lin,
            'dim_out': dim_out,
            'num_layer_conv': num_layer_conv,
            'num_layer_lin': num_layer_lin,
            'num_heads': num_heads,
            'dropout': dropout,
            'pool_k_G': pool_k_G,
        }
        [setattr(self, k, v) for k, v in self.hps.items()]

    def forward(self, batch):
        """Forward pass for the single-channel ablation model.

        Args:
            batch (torch_geometric.data.HeteroData): A batch of heterogeneous
                graphs, from which data for 'node' type is extracted.

        Returns:
            Tensor: Classification logits.
        """
        x = batch.x_dict['node']
        edge_index = batch.edge_index_dict[('node', 'n2n', 'node')]
        edge_attr = batch.edge_attr_dict[('node', 'n2n', 'node')]
        batch_idx = batch.batch_dict['node']

        h = self.conv_G(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h, edge_index, edge_attr, batch_idx, _, _ = self.pool_G(
            x=h, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx)
        h = self.sort_G(x=h, batch=batch_idx)
        x = self.mlp(x=h)

        return x

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"dim_in_G={self.dim_in_G}, "
                f"dim_h_conv={self.dim_h_conv}, "
                f"dim_h_lin={self.dim_h_lin}, "
                f"dim_out={self.dim_out}, "
                f"num_layer_conv={self.num_layer_conv}, "
                f"num_layer_lin={self.num_layer_lin}, "
                f"num_heads={self.num_heads}, "
                f"pool_k_G={self.pool_k_G}, "
        )


class GnLTransformer(GnLTransformer_Hetero):
    """The Graph and Line Graph Transformer (GnLTransformer) model.

    This class provides a convenient wrapper around `GnLTransformer_Hetero`.
    It is designed to directly process a batched `HeteroData` object,
    automatically unpacking the node and edge data for the graph (G) and
    line graph (L) channels before feeding them through the dual-channel
    architecture.

    Inherits all arguments from `GnLTransformer_Hetero`.

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
    def forward(self, batch):
        """Forward pass for the GnLTransformer model.

        Args:
            batch (torch_geometric.data.HeteroData): A batch of heterogeneous
                graphs containing data for both 'node' and 'edge' types.
        
        Returns:
            Tensor: Classification logits.
        """
        x_G, edge_index_G, edge_attr_G, batch_G = (
            batch.x_dict['node'],
            batch.edge_index_dict[('node', 'n2n', 'node')],
            batch.edge_attr_dict[('node', 'n2n', 'node')],
            batch.batch_dict['node']
        )
        x_L, edge_index_L, edge_attr_L, batch_L = (
            batch.x_dict['edge'],
            batch.edge_index_dict[('edge', 'e2e', 'edge')],
            batch.edge_attr_dict[('edge', 'e2e', 'edge')],
            batch.batch_dict['edge']
        )

        x_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L = self.conv_L(x_L, edge_index_L, edge_attr_L)

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)

        return x