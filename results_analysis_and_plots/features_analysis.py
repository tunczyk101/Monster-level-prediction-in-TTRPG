import os

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from results_analysis_and_plots.constants import OTHER_PLOTS_FOLDER
from training.constants import PATH_TO_DATASET, RANDOM_STATE


def feature_analysis(X: pd.DataFrame, y: pd.Series) -> None:
    variance = X.var()
    pearson_corr = X.corrwith(y)
    mi = pd.Series(
        mutual_info_regression(X, y, random_state=RANDOM_STATE), index=X.columns
    )

    feature_stats = pd.DataFrame(
        {
            "variance": variance,
            "pearson_corr_with_target": pearson_corr,
            "mutual_info": mi,
        }
    )

    feature_stats.to_csv(os.path.join(OTHER_PLOTS_FOLDER, "features_stats.csv"))


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)

    level = bestiaries["level"]
    bestiaries = bestiaries.drop(columns=["name", "level", "book"])

    feature_analysis(bestiaries, level)
