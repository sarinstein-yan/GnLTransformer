import os.path as osp
import pickle
import numpy as np
import networkx as nx
from urllib.request import urlretrieve
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from joblib import Parallel, delayed
from huggingface_hub import snapshot_download
from gnl_transformer.utils import line_graph_undirected

class NHSG117K(InMemoryDataset):
    def __init__(
        self, 
        root,
        *,
        transform=None, 
        pre_transform=None, 
        pre_filter=None,
        n_jobs=-1,
    ):
        
        self.n_jobs = n_jobs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["nx_multigraph_dataset.npz", "isomorphism_classes.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        cache_dir = snapshot_download(
            repo_id="sarinstein-yan/NHSG117K",
            repo_type="dataset",
            allow_patterns=["raw/" + fname for fname in self.raw_file_names] + \
                            ["processed/" + fname for fname in self.processed_file_names],
            local_dir=self.root,
        )
        print("Raw and Processed Files Downloaded to:", cache_dir)
    
    def process(self):
        threader = Parallel(n_jobs=self.n_jobs, prefer="threads")

        # prepare raw data
        data = np.load(osp.join(self.raw_dir, 'nx_multigraph_dataset.npz'), allow_pickle=True)
        nx_graphs = threader(delayed(pickle.loads)(g) 
                             for g in tqdm(data['graphs_pickle'], 
                                           desc="Loading networkx multigraphs"))
        y = data['y']
        y_multi = data['y_multi']
        free_coeffs = data['free_coefficient_lists']
        
        mask = pickle.load(open(osp.join(self.raw_dir, 'isomorphism_classes.pkl'), 'rb'))
        isomorphism_classes = np.empty(len(nx_graphs), dtype=int)
        for i, m in enumerate(mask):
            isomorphism_classes[m] = i

        def process_graph(graph, i):
            """Process a networkx multigraph to a HeteroData object."""
            G, L = self._pyg_Data_pairs(graph)

            return HeteroData({
                'node': {
                    'x': G.x,
                    'pos': G.x
                },
                ('node', 'n2n', 'node'): {
                    'edge_index': G.edge_index,
                    'edge_attr': G.edge_attr
                },

                'edge': {
                    'x': L.x,
                    'pos': L.pos,
                },
                ('edge', 'e2e', 'edge'): {
                    'edge_index': L.edge_index,
                    'edge_attr': L.edge_attr
                },
                
                'y': torch.tensor([y[i]], dtype=torch.long),
                'y_multi': torch.tensor([y_multi[i]], dtype=torch.long),
                'free_coeffs': torch.tensor([free_coeffs[i]], dtype=torch.float32),
            })
        
        data_list = threader(delayed(process_graph)(graph, i)
                          for i, graph in tqdm(enumerate(nx_graphs), 
                                               total=len(nx_graphs), 
                                               desc="Processing networkx multigraphs"))

        # add 9 copies of the [0,0,0,0,0,0] sample
        idx0 = len(data_list)//2
        assert torch.allclose(data_list[idx0].y_multi, torch.tensor([0, 0, 0, 0, 0, 0]))
        data_list = data_list[:idx0] + \
                    [data_list[idx0]] * 9 + \
                    data_list[idx0:]

        if self.pre_filter is not None:
            data_list = [self.pre_filter(d) for d in data_list]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        self.save(data_list, self.processed_paths[0])

    # Graph attribute stripping
    @staticmethod
    def _process_edge_pts(graph: nx.MultiGraph) -> nx.MultiGraph:
        """Return a *new* graph with compact node / edge attributes."""
        g = graph.copy()

        node_pos, node_pot, node_dos = {}, {}, {}
        for nid, nd in g.nodes(data=True):
            pos = np.asarray(nd.pop("pos"), dtype=np.float32).reshape(-1)
            pot = np.float32(nd.pop("potential", 0.0))
            dos = np.float32(nd.pop("dos", 0.0))
            # nd["x"] = np.array([*pos, pot, dos], dtype=np.float32)
            nd["pos"] = pos; nd["potential"] = pot; nd["dos"] = dos
            node_pos[nid], node_pot[nid], node_dos[nid] = pos, pot, dos

        for u, v, ed in g.edges(data=True):
            w = np.float32(ed.pop("weight"))
            if "pts" in ed:
                pts = ed.pop("pts")
                pts5_idx = np.round(np.arange(1, 6) * (len(pts) - 1) / 6).astype(int)
                pts5 = pts[pts5_idx].astype(np.float32).reshape(-1)
                pts2_idx = np.round(np.arange(1, 3) * (len(pts) - 1) / 3).astype(int)
                pts2 = pts[pts2_idx].astype(np.float32)
                avg_pot = np.float32(ed.pop("avg_potential", 0.5 * (node_pot[u] + node_pot[v])))
                avg_dos = np.float32(ed.pop("avg_dos", 0.5 * (node_dos[u] + node_dos[v])))
            else:
                mid = 0.5 * (node_pos[u] + node_pos[v])
                pts5 = np.tile(mid, 5).astype(np.float32)
                pts2 = np.tile(mid, 2).astype(np.float32)
                avg_pot = np.float32(0.5 * (node_pot[u] + node_pot[v]))
                avg_dos = np.float32(0.5 * (node_dos[u] + node_dos[v]))
            # ed["edge_attr"] = np.concatenate(([w, avg_pot, avg_dos], pts5), dtype=np.float32)
            ed["weight"] = w; ed["pts5"] = pts5; ed["pts2"] = pts2
            ed["avg_potential"] = avg_pot; ed["avg_dos"] = avg_dos
        return g

    # Construct graph and line graph pairs as PyG Data objects
    @staticmethod
    def _pyg_Data_pairs(graph: nx.multigraph):
        """Return a pair of PyG Data objects: (graph, line_graph)."""
        nx_G = NHSG117K._process_edge_pts(graph)

        nx_L = line_graph_undirected(nx_G, with_triplet_features=True)
        if nx_L.number_of_edges() == 0:
            nx_L = line_graph_undirected(nx_G, with_triplet_features=True, add_selfloops=True)
        
        pyg_G = from_networkx(nx_G, 
                              group_node_attrs=['pos'],
                              group_edge_attrs=['weight', 'pts5'])
        pyg_L = from_networkx(nx_L, 
                              group_node_attrs=['weight', 'pts5'],
                              group_edge_attrs=['triplet_center', 'angle'])
        return pyg_G, pyg_L