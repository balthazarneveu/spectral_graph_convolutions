import matplotlib.pyplot as plt
import torch


def whiten(inp: torch.Tensor) -> torch.Tensor:
    """Make data white over the dataset
    µ=0 and sigma=1 for each of the F components of the input
    Args:
        inp (torch.Tensor): [V, F] non normalized input

    Returns:
        torch.Tensor: normalized output, mean=1, std_dev=1
    """
    return (inp - inp.mean(axis=0)) / inp.std(axis=0)


def visual_sanity_check_input(input: torch.Tensor) -> None:
    """Plot mean and std of input data
    """
    avg = input.mean(axis=0)
    std = input.std(axis=0)
    plt.plot(avg.detach().cpu().numpy(), label="mean over all subjects")
    plt.plot(std.detach().cpu().numpy(), label="std over all subjects")
    plt.xlabel("feature index")
    plt.legend()
    plt.grid()
    plt.show()


def sanitize_data(input: torch.Tensor) -> torch.Tensor:
    """Remove all non informative features (std_dev=0 across dataset)

    Args:
        input (torch.Tensor): [V, F] input data

    Returns:
        torch.Tensor: [V, F'] sanitized input data, F' <= F
    """
    std = input.std(axis=0)
    non_zero_std_indices = torch.nonzero(std, as_tuple=True)[0]
    clean_inp = input[:, non_zero_std_indices]
    return clean_inp
