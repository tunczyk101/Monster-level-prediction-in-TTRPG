import math
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from results_analysis_and_plots.constants import FEATURES_NAMES_MAP, OTHER_PLOTS_FOLDER
from training.constants import PATH_TO_DATASET

COLOR = "slateblue"


def plot_feature_histograms(df, bins=30, discrete_threshold=30, scale="linear") -> None:
    numeric_df = df.select_dtypes(include="number")
    max_ticks = 10

    n_cols = 6
    n_rows = math.ceil(len(numeric_df.columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_df.columns):
        data = numeric_df[col].dropna()

        unique_vals = np.unique(data)

        if (
            np.issubdtype(data.dtype, np.integer)
            and len(unique_vals) <= discrete_threshold
            and col != "swim"
        ):
            values, counts = np.unique(data, return_counts=True)
            axes[i].bar(values, counts, color=COLOR)

            if len(values) > max_ticks:
                step = max(1, len(values) // max_ticks)
                axes[i].set_xticks(values[::step])
            else:
                axes[i].set_xticks(values)

            axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

        else:
            axes[i].hist(data, bins=bins, color=COLOR)

        axes[i].set_title(FEATURES_NAMES_MAP[col])
        axes[i].set_yscale(scale)

    # remove unused plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(
        os.path.join(OTHER_PLOTS_FOLDER, f"features_histograms_{scale}.svg"),
    )


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)

    bestiaries = bestiaries.drop(columns=["name", "level", "book"])

    for scale in ["linear", "log"]:
        plot_feature_histograms(bestiaries, scale=scale)
