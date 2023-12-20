from typing import Optional, Union, List
import torch
from scipy import sparse
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class BasicGCNDenseLayer(torch.nn.Module):
    """Base block for Graph Convolutional Network
    Implemented for dense graphs (no sparse matrix)
    """

    def __init__(
        self, input_dim, output_dim, normalized_adjacency_matrix: torch.Tensor,
        bias=False,
        apply_non_linearity=True
    ):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim, bias=bias)
        self.adj = normalized_adjacency_matrix
        self.apply_non_linearity = apply_non_linearity

    def forward(self, inp: torch.Tensor):
        x = self.fc1(inp)
        x = self.adj @ x
        if not self.apply_non_linearity:
            return x
        x = torch.nn.functional.relu(x)
        return x


class GCN(torch.nn.Module):
    """Graph Convolutional Network for dense graphs
    """
    @staticmethod
    def get_normalized_adjacency_matrix(adjacency_matrix: torch.Tensor, add_self_loops=True):
        if add_self_loops:
            adj = adjacency_matrix + \
                torch.eye(adjacency_matrix.shape[0], device=adjacency_matrix.device)  # add a self loop
        else:
            adj = adjacency_matrix
        degree = adj.sum(dim=1)  # Degree of each node
        d_inv_sqrt = torch.diag(1./torch.sqrt(degree))  # D^-1/2
        adj = d_inv_sqrt @ adj @ d_inv_sqrt  # D^-1/2 A D^-1/2
        return adj

    def __init__(self, input_dim, adjacency: torch.Tensor, hdim, p_dropout=0.):
        super().__init__()
        hdim1 = hdim
        hdim2 = hdim1
        hdim3 = hdim2
        output_dim = 1  # Binary classification here
        self.adj = GCN.get_normalized_adjacency_matrix(adjacency)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.gcn1 = BasicGCNDenseLayer(input_dim, hdim1, self.adj)
        self.gcn2 = BasicGCNDenseLayer(hdim1, hdim2, self.adj)
        self.gcn3 = BasicGCNDenseLayer(hdim2, hdim3, self.adj)
        self.classifier = torch.nn.Linear(hdim2, output_dim)
        self.classifier2 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.gcn1(x)
        x = self.dropout(x)
        x = self.gcn2(x) + x
        x = self.dropout(x)
        x = self.gcn3(x) + x
        x = self.dropout(x)
        logit = self.classifier(x)
        return logit.squeeze()


class Aggregate(torch.nn.Module):
    def __init__(self, input_dim, output_dim, mat=None):
        super().__init__()
        self.fc = torch.nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.mat = mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mat is not None:
            y = self.mat @ x
        else:
            y = x
        y = self.fc(y)
        return y


class SimpleTchebconv(torch.nn.Module):
    def __init__(self, input_dim, adjacency: torch.Tensor, output_dim: int = 1, K=3):
        super().__init__()
        adj = GCN.get_normalized_adjacency_matrix(adjacency, add_self_loops=False)
        lap = torch.eye(adj.shape[0], device=adj.device) - adj
        lambda_max = lap.max()
        lap /= lambda_max
        lap -= torch.eye(adj.shape[0], device=adj.device)  # Remove self loop
        self.t0 = None
        self.t1 = lap
        self.t2 = 2.*lap@lap - torch.eye(adj.shape[0], device=adj.device)
        # self.t3 = 4*lap@lap@lap - 3*lap
        self.gcn0 = Aggregate(input_dim, output_dim, self.t0)
        self.gcn1 = Aggregate(input_dim, output_dim, self.t1)
        self.gcn2 = Aggregate(input_dim, output_dim, self.t2)
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor):
        logit = self.gcn0(x) + self.gcn1(x) + self.gcn2(x) + self.bias
        return logit.squeeze()


class SimpleGraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__()
        print(f"in_feat {in_features} out_feat {out_features}")
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.out_features)
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.fc.bias:
            self.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        support = self.fc(x)
        output = torch.spmm(adj, support)

        return output


class SparseGCN(torch.nn.Module):
    """Graph Convolutional Network based on http://arxiv.org/abs/1609.02907"""

    def get_normalized_adjacency_matrix(adj: np.array, dtype: torch.dtype = torch.float32):
        """ Apply the renormalization trick 
        return a sparse matrix """

        # warning : addition is valif if adj is a numpy.array but not if it a tensor
        adj = adj + sparse.eye(adj.shape[0])
        tilde_d = np.array(1 / adj.sum(1)).flatten()
        tilde_d[np.isinf(tilde_d)] = 0.
        tilde_d = sparse.diags(tilde_d)
        adj = tilde_d.dot(adj)

        return torch.tensor(adj, dtype=dtype).to_sparse()

    def __init__(self, nfeat: int, nhid: Union[int, List[int]], nclass: int, adjacency: np.array, proba_dropout: float = 0.5, device="cpu"):
        super().__init__()
        assert type(adjacency) == np.ndarray, "For initialization, adjacency matrix must be an array"
        self.adj = SparseGCN.get_normalized_adjacency_matrix(adjacency).to(device)
        if isinstance(nhid, int):
            nhid = [nhid]
        self.gc_first = SimpleGraphConvolution(nfeat, nhid[0], bias=False)
        self.gc_hid = torch.nn.ModuleList()
        for i, hid in enumerate(nhid[1:]):
            self.gc_hid.append(
                SimpleGraphConvolution(nhid[i], hid, bias=False)
            )
        self.gc_last = SimpleGraphConvolution(nhid[-1], nclass, bias=False)

        self.proba_dropout = proba_dropout

    def forward(self, x: torch.Tensor):
        x = F.relu(self.gc_first(x, self.adj))
        x = F.dropout(x, self.proba_dropout, training=self.training)

        for gc in self.gc_hid:
            x = F.relu(gc(x, self.adj))
            x = F.dropout(x, self.proba_dropout, training=self.training)

        x = self.gc_last(x, self.adj)  # (n, nclass)
        # x = F.softmax(x, dim=1) # warning : usually for multi-class or when the loss is nnllloss
        return x


class ChebGCN(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 adjacency: np.ndarray,
                 K: int = 3,
                 proba_dropout: float = 0.3,
                 trim=None,
                 decimate=None,
                 device=None):
        super().__init__()
        self.proba_dropout = proba_dropout
        self.chebconv = ChebConv(in_features, out_features, K)
        self.adj = adjacency

        self.edge_index = self.compute_edge_index()
        self.edge_weight = self.compute_edge_weight()
        self.edge_index = self.edge_index.to(device)

        if trim is not None:
            self.edge_index = self.edge_index[:, :trim]
        if decimate is not None:
            self.edge_index = self.edge_index[:, ::decimate]
        self.edge_weight = self.edge_weight.to(device)
        if trim is not None:
            self.edge_weight = self.edge_weight[:trim]
        if decimate is not None:
            self.edge_weight = self.edge_weight[::decimate]

    def compute_edge_index(self):
        edge_index = np.array(self.adj.nonzero())
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        return edge_index

    def compute_edge_weight(self):
        edge_weight = self.adj[self.edge_index[0], self.edge_index[1]]
        return torch.tensor(edge_weight, dtype=torch.float32)

    def forward(self, x: torch.tensor):
        x = self.chebconv(x, self.edge_index, self.edge_weight)
        x = F.dropout(x, self.proba_dropout, training=self.training)
        # One shall not do dropout on a single feature layer!
        return x.squeeze()
