import torch


class DenseNN(torch.nn.Module):
    """Baseline Dense Neural Network"""

    def __init__(self, input_dim, hdim=64, p_dropout=0.1):
        super().__init__()
        hdim1 = hdim
        hdim2 = hdim1
        output_dim = 1
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(input_dim, hdim1)
        self.fc2 = torch.nn.Linear(hdim1, hdim2)
        self.classifier = torch.nn.Linear(hdim2, output_dim)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, inp: torch.Tensor):
        x = self.relu(self.fc1(inp))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        logit = self.classifier(x)
        return logit.squeeze()


class DenseNNSingle(torch.nn.Module):
    """Baseline Dense Neural Network Single layer"""

    def __init__(self, input_dim, hdim=64, p_dropout=0.1):
        super().__init__()
        output_dim = 1
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(input_dim, hdim)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.classifier = torch.nn.Linear(hdim, output_dim)

    def forward(self, inp: torch.Tensor):
        x = self.relu(self.fc1(inp))
        x = self.dropout(x)
        logit = self.classifier(x)
        return logit.squeeze()
