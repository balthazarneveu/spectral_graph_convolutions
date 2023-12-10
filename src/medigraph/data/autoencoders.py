"""Training loop
"""
from tqdm.notebook import tqdm
from medigraph.data.properties import INPUTS, LABELS
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(AutoEncoder, self).__init__()
        self.fc_encoder = nn.Linear(input_dim, bottleneck_dim, dtype = torch.float32)
        self.fc_decoder = nn.Linear(bottleneck_dim, input_dim, dtype = torch.float32)
                  
    def forward(self, x):
        x = self.fc_encoder(x)
        x = torch.tanh(x)
        x = self.fc_decoder(x)

        return x

def get_loader(train_dataset, batch_size=64, num_workers=1, mode='train'):
    """Build and return data loader."""
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             drop_last=True)
  
    return data_loader

def training_loop(
    model: torch.nn.Module,
    X,
    device,
    optimizer_params={
        "lr": 1.E-3,
        "weight_decay": 0.001
    },
    n_epochs: Optional[int] = 150,
    batch_size: Optional[int] = 256,
) -> Tuple[torch.nn.Module, dict]:
    
    inp_train, inp_test = train_test_split(X, test_size=0.1, random_state=42)

    criterion = torch.nn.MSELoss()
    train_input = inp_train
    val_input = inp_test
    train_loader = DataLoader(train_input, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_input, batch_size=batch_size, shuffle=False)

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), **optimizer_params)
    training_losses = []
    validation_losses = []
    for ep in tqdm(range(n_epochs)):
        model.train()
        for batch_data in train_loader:
            batch_data = batch_data.to(device, dtype = torch.float32)
            optim.zero_grad()
            output = model(batch_data)
            loss = criterion(batch_data, output)
            loss.backward()
            optim.step()

        print(f"Epoch {ep} training loss: {loss.item():10f}")
        training_losses.append(loss.detach().cpu())
        
        model.eval()
        with torch.no_grad():
            val_loss_mean = 0
            i = 0
            for val_batch_data in val_loader:
                i += 1
                val_batch_data = val_batch_data.to(device, dtype = torch.float32)
                val_output = model(val_batch_data)
                val_loss = criterion(val_batch_data, val_output)
                validation_losses.append(val_loss.item())
                val_loss_mean += val_loss.item()
            val_loss_mean /= i
            if ep % 10 == 0:
                print(f"Epoch {ep} validation loss: {val_loss_mean:10f}")
    
    metrics = {
        "training_losses": training_losses,
        "validation_losses": validation_losses
        }
    return model, metrics