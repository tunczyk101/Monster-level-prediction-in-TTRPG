import os

from matplotlib import pyplot as plt
import pandas as pd

from results_analysis_and_plots.constants import OTHER_PLOTS_FOLDER
from training.constants import LOSS_RESULTS_DIR, NEURAL_NETWORK_MODELS, SET_NAME


def plot_learning_curve(history: pd.DataFrame, model_name: str):
    train_loss = history["train_loss"]
    valid_loss = history["valid_loss"]

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="valid")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OTHER_PLOTS_FOLDER, "nn_loss", f"{model_name}_learning_curve.svg")
    )


if __name__ == "__main__":
    for model in NEURAL_NETWORK_MODELS:
        print(model)
        history = pd.read_csv(os.path.join(LOSS_RESULTS_DIR, f"{SET_NAME}_{model}.csv"))
        plot_learning_curve(history, model)
