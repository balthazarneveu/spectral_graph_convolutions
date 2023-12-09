import matplotlib.pyplot as plt

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
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    colors = ["b", "g", "r", "y", "k"]
    # colors  = [""]
    for idx, (model_name, metric) in enumerate(metric_dict.items()):
        color = colors[idx % len(colors)]
        axs[0].plot(metric[LOSS][TRAIN], color+"--", label=model_name)
        axs[0].plot(metric[LOSS][VALIDATION], color+"-", label=model_name)
        axs[1].plot(metric[ACCURACY][TRAIN], color+"--", label=f"{model_name} TRAIN accuracy")
        axs[1].plot(metric[ACCURACY][VALIDATION], color+"-", label=f"{model_name} VALIDATION accuracy")
    for ax in axs:
        ax.legend()
        ax.grid()
    axs[0].set_title("Training loss (Binary Cross Entropy)")
    axs[1].set_title("Accuracy")

    plt.show()
