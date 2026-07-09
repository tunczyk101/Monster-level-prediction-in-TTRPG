import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results_analysis_and_plots.constants import MODEL_LABEL, OTHER_PLOTS_FOLDER
from training.constants import RESULTS_DIR, SET_COMPARISION_MODELS


def plot_grouped_bars(
    all_df: dict[str, pd.DataFrame],
    models: list[str],
    rounding="round 0.5",
    metric="mae_macroaveraged",
    bar_width: float = 0.8,
    title: str | None = None,
):
    models_results = defaultdict(list)

    for df in all_df.values():
        result = df[(rounding, metric)]
        for model in models:
            models_results[model].append(result[model])

    n_series = len(models_results)
    n_labels = len(all_df)
    labels = list(all_df.keys())

    # Validate input
    for name, values in models_results.items():
        if len(values) != n_labels:
            raise ValueError(
                f"Series '{name}' has {len(values)} values, expected {n_labels}"
            )

    x = np.arange(n_labels)
    single_bar_width = bar_width / n_series

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.get_cmap("tab10").colors

    for i, (series_name, values) in enumerate(models_results.items()):
        offset = (i - n_series / 2) * single_bar_width + single_bar_width / 2
        ax.bar(
            x + offset,
            values,
            width=single_bar_width,
            label=MODEL_LABEL[series_name],
            color=colors[i % len(colors)],
            # edgecolor="black",               # improves readability
            # linewidth=0.1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.35, 0.5), loc="right")

    ax.set_ylabel("MAE macroaveraged")

    plt.tight_layout()
    fig.savefig(
        os.path.join(OTHER_PLOTS_FOLDER, f"{title.lower().replace(' ', '_')}.svg"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(OTHER_PLOTS_FOLDER, f"{title.lower().replace(' ', '_')}.pdf"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    results_paths = {
        "Basic": os.path.join(RESULTS_DIR, "basic_results_test.csv"),
        "Expanded": os.path.join(RESULTS_DIR, "expanded_results_test.csv"),
        "Full": os.path.join(RESULTS_DIR, "full_results_test.csv"),
    }

    all_df = {}
    for set_name, path_to_file in results_paths.items():
        all_df[set_name] = pd.read_csv(
            path_to_file,
            header=[0, 1],
            index_col=[0],
        )

    plot_grouped_bars(
        all_df,
        title="Set size vs results MAE macroaveraged",
        models=SET_COMPARISION_MODELS,
    )
