import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dataset.splitting_dataset import split_dataframe
from results_analysis_and_plots.constants import OTHER_PLOTS_FOLDER
from training.constants import (
    CHOSEN_MODEL,
    MODELS_RESULTS_DIR,
    PATH_TO_DATASET,
    RESULTS_DIR,
    SET_NAME,
)
from training.rounding import (
    round_single_threshold_results,
)


TEST_RESULTS_FILE = os.path.join(
    MODELS_RESULTS_DIR, f"{SET_NAME}_{CHOSEN_MODEL}_test.csv"
)
TRAIN_RESULTS_FILE = os.path.join(
    MODELS_RESULTS_DIR, f"{SET_NAME}_{CHOSEN_MODEL}_train.csv"
)


def plot_confusion_matrix(
    y_test_pred: np.ndarray,
    y_test: pd.Series,
    model_name: str,
    figsize: tuple[int, int] = (10, 10),
    export: bool = False,
) -> None:
    round_predict = round_single_threshold_results(y_test_pred, 0.5)

    cm = confusion_matrix(y_test, round_predict)

    # min possible level: -1, max possible level: 21
    labels = [i for i in range(-1, 22)]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, colorbar=False)

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(disp.im_, cax=cax)

    disp.ax_.set_xlabel("Predicted level", fontweight="bold", fontsize=20)
    disp.ax_.set_ylabel("True level", fontweight="bold", fontsize=20)

    if export:
        fig.savefig(
            os.path.join(
                OTHER_PLOTS_FOLDER,
                f"confusion_matrix_{model_name}_mathematical.svg",
            ),
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(
                OTHER_PLOTS_FOLDER,
                f"confusion_matrix_{model_name}_mathematical.pdf",
            ),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    model_name = CHOSEN_MODEL
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)
    _, _, y_train, y_test = split_dataframe(bestiaries)

    y_test += 1
    y_train += 1

    y_test_pred = pd.read_csv(
        os.path.join(RESULTS_DIR, "models_predictions", f"full_{model_name}_test.csv"),
        header=None,
    )[0].to_numpy()
    y_train_pred = pd.read_csv(
        os.path.join(RESULTS_DIR, "models_predictions", f"full_{model_name}_train.csv"),
        header=None,
    )[0].to_numpy()

    plot_confusion_matrix(
        y_test_pred=y_test_pred,
        y_test=y_test,
        export=True,
        model_name=model_name,
    )
