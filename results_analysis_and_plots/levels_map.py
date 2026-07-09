import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset.creating_dataset import min_max_scale_data
from dataset.splitting_dataset import split_levels_column
from results_analysis_and_plots.constants import OTHER_PLOTS_FOLDER
from training.constants import PATH_TO_DATASET, RANDOM_STATE


def reduce_data(bestiaries: pd.DataFrame, reduce_type: str):
    if reduce_type == "UMAP":
        reducer = umap.UMAP(random_state=RANDOM_STATE)

    elif reduce_type == "TSNE":
        reducer = TSNE(
            n_components=2,
            random_state=RANDOM_STATE,
        )

    elif reduce_type == "PCA":
        reducer = PCA(n_components=2)

    else:
        raise ValueError(f"Incorrect reduce type: {reduce_type}")

    return reducer.fit_transform(bestiaries)


def plot_all_methods(
    bestiaries: pd.DataFrame,
    levels: np.ndarray,
    methods: list[str],
) -> None:
    n = len(methods)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(6 * n, 5),
        constrained_layout=True,
    )

    base_cmap = plt.cm.Purples
    colors = base_cmap(np.linspace(0.1, 1, 256))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "no_white",
        colors,
    )

    scatter = None

    for ax, method in zip(axes, methods):
        embedding = reduce_data(bestiaries, method)

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=levels,
            cmap=cmap,
            edgecolors="purple",
            linewidths=0.2,
        )

        ax.set_title(method)

    fig.colorbar(
        scatter,
        ax=axes,
        label="Level",
        shrink=0.9,
    )

    plt.savefig(
        os.path.join(
            OTHER_PLOTS_FOLDER,
            f"levels_all_{'_'.join(methods)}.pdf",
        ),
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    bestiaries = pd.read_csv(
        PATH_TO_DATASET,
        index_col=0,
    )

    bestiaries = min_max_scale_data(bestiaries)

    X, y = split_levels_column(bestiaries)

    plot_all_methods(
        X,
        y,
        ["PCA", "TSNE"],
    )
