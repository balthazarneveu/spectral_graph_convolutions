import matplotlib.pyplot as plt
import numpy as np
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
LOSS = "loss"
ACCURACY = "accuracy"


def plot_metrics(metric_dict: dict) -> None:
    """Compare training metrics of different models

    Args:
        metric_dict (dict): Dictionary of metrics for each model
        ```
        {
            "model1": {
                "loss": {
                    "train": [float],
                    "validation: [float]
                },
                "accuracy": {
                    "train": [float],
                    "validation: [float]
                }
            "model2": {
                "loss": {
                    "train": [float],
                    "validation: [float]
                },
                "accuracy": {
                    "train": [float],
                    "validation: [float]
                }
            }
        }
        ```
    """
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    colors = ["b", "g", "r", "y", "m", "c", "k"]
    # colors  = [""]
    for idx, (model_name, metric) in enumerate(metric_dict.items()):
        color = colors[idx % len(colors)]
        for seed_idx, seed in enumerate(metric.keys()):
            current_metric = metric[seed]
            axs[0].plot(current_metric[LOSS][TRAIN], color+"--",
                        label=None if seed_idx >= 1 else (model_name + " TRAIN"))
            axs[0].plot(current_metric[LOSS][VALIDATION], color+"-.",
                        alpha=0.8,
                        label=None if seed_idx >= 1 else (model_name + " VALIDATION"))
            axs[0].plot(current_metric[LOSS][TEST], color+"-", linewidth=2,
                        label=None if seed_idx >= 1 else (model_name + " TEST"))
            axs[1].plot(current_metric[ACCURACY][TRAIN], color+"--",
                        label=None if seed_idx >= 1 else (f"{model_name} TRAIN accuracy"))
            axs[1].plot(current_metric[ACCURACY][VALIDATION], color+"-.",
                        alpha=0.8,
                        label=None if seed_idx >= 1 else (f"{model_name} VALIDATION accuracy"))
            axs[2].plot(current_metric[ACCURACY][TEST], color+"-",
                        # linewidth=2,
                        alpha=0.1,
                        label=None if seed_idx >= 1 else (f"{model_name} TEST accuracy"))
    for idx, (model_name, metric) in enumerate(metric_dict.items()):
        color = colors[idx % len(colors)]
        acc_test = np.array([metric[seed][ACCURACY][TEST]for seed in metric.keys()]).mean(axis=0)
        axs[2].plot(
            acc_test,
            color+"-",
            linewidth=3,
            alpha=1.,
            label=f"{model_name} average TEST accuracy")
    for ax in axs:
        ax.legend()
        ax.grid()
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Binary Cross Entropy Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[2].set_ylabel("Test accuracy")
    axs[0].set_title("Losses")
    axs[1].set_title("Accuracy")
    axs[2].set_title("Test accuracy")

    plt.show()
