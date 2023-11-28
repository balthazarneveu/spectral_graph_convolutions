"""Training loop
"""
import torch
try:
    __IPYTHON__  # noqa: F821
    from tqdm.notebook import tqdm
except Exception as _e:  # noqa: F841
    from tqdm import tqdm
from medigraph.data.properties import INPUTS, LABELS
from typing import Tuple, Optional


def training_loop(
    model: torch.nn.Module,
    train_data: dict, device,
    valid_data: Optional[dict] = None,
    optimizer_params={
        "lr": 1.E-4,
        "weight_decay": 0.1
    },
    n_epochs: Optional[int] = 1000
) -> Tuple[torch.nn.Module, dict]:
    criterion = torch.nn.BCEWithLogitsLoss()
    train_input = train_data[INPUTS]
    train_label = train_data[LABELS]
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), **optimizer_params)
    training_losses = []
    training_accuracies = []
    for ep in tqdm(range(n_epochs)):
        model.train()
        optim.zero_grad()
        logit = model(train_input)
        loss = criterion(logit, train_label)
        loss.backward()
        optim.step()
        with torch.no_grad():
            predicted_prob = torch.sigmoid(logit).squeeze()  # Apply sigmoid and remove extra dimensions if any
            predicted = (predicted_prob >= 0.5).long()  # Convert probabilities to 0 or 1
            correct = (predicted == train_label).sum().item()
            total = train_label.shape[0]
            accuracy = correct / total
            training_accuracies.append(accuracy)
        if ep % 100 == 0:
            print(f"Epoch {ep} loss: {loss.item():10f} - accuracy: {accuracy:.2%}")
        training_losses.append(loss.detach().cpu())
    metrics = {
        "training_losses": training_losses,
        "training_accuracies": training_accuracies
    }
    return model, metrics
