import os

import pandas as pd
from matplotlib import pyplot as plt

from results_analysis_and_plots.constants import OTHER_PLOTS_FOLDER
from training.constants import PATH_TO_DATASET


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)
    ax = bestiaries.level.value_counts().sort_index().plot.bar(color="mediumorchid")

    ax.set_xlabel("Level")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=0)
    plt.savefig(os.path.join(OTHER_PLOTS_FOLDER, "bestiary_levels.svg"), format="svg")
    plt.savefig(os.path.join(OTHER_PLOTS_FOLDER, "bestiary_levels.pdf"), format="pdf")
    plt.close()
