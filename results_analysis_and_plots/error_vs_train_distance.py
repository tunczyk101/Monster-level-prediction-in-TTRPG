import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_distances

from dataset.splitting_dataset import split_dataframe
from results_analysis_and_plots.constants import OTHER_PLOTS_FOLDER
from results_analysis_and_plots.errors_histogram import load_models_results
from training.constants import CHOSEN_MODEL, PATH_TO_DATASET


def calculate_and_plot_cosine_distance(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray
) -> None:
    abs_error = np.abs(y_test - y_pred)

    distance = cosine_distances(X_test, X_train).min(axis=1)

    plt.figure(figsize=(8, 6))
    plt.scatter(distance, abs_error, alpha=0.6)

    plt.xlabel("Cosine distance to train set")
    plt.ylabel("Absolute error")

    plt.grid(True)
    plt.savefig(
        os.path.join(OTHER_PLOTS_FOLDER, "distance_to_train_vs_absolute_error.svg"),
    )
    plt.savefig(
        os.path.join(OTHER_PLOTS_FOLDER, "distance_to_train_vs_absolute_error.pdf"),
    )


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)

    X_train, X_test, _, y_test = split_dataframe(bestiaries)

    y_test += 1

    y_pred = load_models_results(CHOSEN_MODEL)

    calculate_and_plot_cosine_distance(X_train, X_test, y_test, y_pred)
