import os
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ast

from results_analysis_and_plots.constants import MODEL_LABEL, OTHER_PLOTS_FOLDER
from training.constants import EXPANDING_WINDOW_DIR, ALL_MODELS, ROUNDING


def plot_grouped_bars(
    all_df: dict[str, pd.DataFrame],
    models=list[str],
    metric="mae_macroaveraged",
    bar_width: float = 0.8,
    title: str | None = None,
):
    models_results = defaultdict(list)

    for set_name, df in all_df.items():
        result = df[(ROUNDING, metric)]
        for model in models:
            models_results[set_name].append(result[model])

    n_series = len(models_results)
    n_labels = len(models)
    labels = [MODEL_LABEL[model] for model in models]

    # Validate input
    for name, values in models_results.items():
        if len(values) != n_labels:
            raise ValueError(
                f"Series '{name}' has {len(values)} values, expected {n_labels}"
            )

    x = np.arange(n_labels)
    single_bar_width = bar_width / n_series

    fig, ax = plt.subplots()

    for i, (series_name, values) in enumerate(models_results.items()):
        offset = (i - n_series / 2) * single_bar_width + single_bar_width / 2
        ax.bar(
            x + offset,
            values,
            width=single_bar_width,
            label=series_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend()

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


def _load_expanding_window() -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(
            EXPANDING_WINDOW_DIR,
            "test_results.csv",
        ),
        header=[0, 1],
        index_col=[0],
    )

    result_df = pd.DataFrame()

    for col in df.columns:
        if ROUNDING in col[0] and col[1] == "avg":
            result_df[(ROUNDING, ast.literal_eval(col[0])[1])] = df[col]

    return result_df


if __name__ == "__main__":
    results_paths = {
        "Chronological": os.path.join("training", "results", "full_results_test.csv"),
        "Random": os.path.join("training", "results", "random_full_results_test.csv"),
    }

    all_df = {}
    for set_name, path_to_file in results_paths.items():
        all_df[set_name] = pd.read_csv(
            path_to_file,
            header=[0, 1],
            index_col=[0],
        )

    all_df["Expanding window"] = _load_expanding_window()

    plot_grouped_bars(
        all_df,
        title="Chronological vs random MAE macroaveraged",
        models=ALL_MODELS,
    )
