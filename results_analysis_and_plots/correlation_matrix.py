import os

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from dataset.splitting_dataset import split_dataframe
from results_analysis_and_plots.constants import FEATURES_NAMES_MAP, OTHER_PLOTS_FOLDER
from training.constants import PATH_TO_DATASET


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)
    X, _, _, _ = split_dataframe(bestiaries, test_size=0.00001)
    X.rename(columns=FEATURES_NAMES_MAP, inplace=True)
    matrix = X.corr()

    plt.figure(figsize=(20, 18))

    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        linewidths=0.25,
        annot_kws={"size": 10},
        cbar_kws={
            "shrink": 0.6,
            "pad": 0.02,
        },
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OTHER_PLOTS_FOLDER,
            "correlation_matrix.pdf",
        ),
        bbox_inches="tight",
    )
