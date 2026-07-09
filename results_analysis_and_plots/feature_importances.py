import os
from typing import Optional
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dataset.creating_dataset import min_max_scale_data
from dataset.splitting_dataset import split_dataframe
from results_analysis_and_plots.constants import FEATURES_NAMES_MAP, OTHER_PLOTS_FOLDER
from training.constants import CHRONOLOGICAL_SPLIT, PATH_TO_DATASET
from training.create_model import get_fitted_model
from matplotlib import pyplot as plt
import shap
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "random_forest"
FEATURE_IMPORTANCES_FOLDER = os.path.join(OTHER_PLOTS_FOLDER, "feature_importances")


def _save_figure(
    df: pd.DataFrame, x: str, y: str, saving_path: str, title: Optional[str] = None
) -> None:
    ax = df.plot(kind="barh", x=x, y=y, legend=False)
    ax.set(ylabel=None)
    plt.gca().invert_yaxis()

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(saving_path)


def plot_shap_values(model: RandomForestRegressor, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    mean_abs_shap = np.abs(explainer.shap_values(X)).mean(axis=0)

    shap_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs_shap})
    shap_df.rename(columns=FEATURES_NAMES_MAP)
    shap_df = shap_df.sort_values(by="mean_abs_shap", ascending=False)

    saving_path = os.path.join(
        FEATURE_IMPORTANCES_FOLDER, "shap_feature_importance_head.pdf"
    )
    title = "Feature Importance (SHAP) - Random Forest"
    _save_figure(
        shap_df.head(15),
        x="feature",
        y="mean_abs_shap",
        saving_path=saving_path,
        title="",
    )

    saving_path = os.path.join(
        FEATURE_IMPORTANCES_FOLDER, "shap_feature_importance_tail.pdf"
    )
    _save_figure(
        shap_df.tail(15),
        x="feature",
        y="mean_abs_shap",
        saving_path=saving_path,
        title=title,
    )


if __name__ == "__main__":
    bestiaries = pd.read_csv(os.path.join(PATH_TO_DATASET), index_col=0)
    bestiaries = min_max_scale_data(bestiaries)

    X_train, X_test, y_train, y_test = split_dataframe(
        bestiaries, chronological_split=CHRONOLOGICAL_SPLIT
    )

    X_train.columns = [FEATURES_NAMES_MAP[col] for col in X_train.columns]

    # there are models that require the level to be non-negative
    y_train += 1
    y_test += 1

    n_features = X_train.shape[1]

    model = get_fitted_model(MODEL_NAME, X_train, y_train, n_features).best_estimator_

    plot_shap_values(model, X_train)
