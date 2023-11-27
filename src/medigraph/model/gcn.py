import torch


class BasicGCNDenseLayer(torch.nn.Module):
    """Base block for Graph Convolutional Network
    Implemented for dense graphs (no sparse matrix)
    """

    def __init__(self, input_dim, output_dim, normalized_adjacency_matrix: torch.Tensor):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim, dtype=torch.float64)
        self.adj = normalized_adjacency_matrix

    def forward(self, inp: torch.Tensor):
        x = self.fc1(inp)
        x = self.adj @ x
        x = torch.nn.functional.relu(x, inplace=True)
        return x


class GCN(torch.nn.Module):
    """Graph Convolutional Network for dense graphs
    """
    def get_normalized_adjacency_matrix(adjacency_matrix: torch.Tensor):
        adj = adjacency_matrix + torch.eye(adjacency_matrix.shape[0], device=adjacency_matrix.device)  # add a self loop
        degree = adj.sum(dim=1)  # Degree of each node
        d_inv_sqrt = torch.diag(1./torch.sqrt(degree))  # D^-1/2
        adj = d_inv_sqrt @ adj @ d_inv_sqrt  # D^-1/2 A D^-1/2
        return adj

    def __init__(self, input_dim, adjacency: torch.Tensor):
        super().__init__()
        hdim1 = 512
        hdim2 = hdim1//2
        output_dim = 1  # Binary classification here
        self.adj = GCN.get_normalized_adjacency_matrix(adjacency)
        self.gcn1 = BasicGCNDenseLayer(input_dim, hdim1, self.adj)
        self.gcn2 = BasicGCNDenseLayer(hdim1, hdim2, self.adj)
        self.classifier = torch.nn.Linear(hdim2, output_dim, dtype=torch.float64)

    def forward(self, inp: torch.Tensor):
        x = self.gcn1(inp)
        x = self.gcn2(x)
        logit = self.classifier(x)
        return logit
