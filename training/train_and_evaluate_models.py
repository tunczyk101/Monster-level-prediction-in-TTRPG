import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from metrics import (
    accuracy_at_k,
    calculate_average_and_std,
    mae_macroaveraged,
    mse_macroaveraged,
    rmse_macroaveraged,
    somers_d,
)
from pandas import DataFrame, MultiIndex
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from statsmodels.miscmodels.ordinal_model import OrderedResultsWrapper

from training.constants import (
    EXPANDING_WINDOW_DIR,
    LOSS_RESULTS_DIR,
    MODELS_RESULTS_DIR,
    NEURAL_NETWORK_MODELS,
    RANDOM_PREFIX,
    SET_NAME,
)
from training.create_model import ModelType, get_fitted_model
from training.rounding import (
    round_single_threshold_results,
)


ORDINAL_MODELS = [
    "linear_ordinal_model_probit",
    "linear_ordinal_model_logit",
    "ordered_random_forest",
    "logisticAT",
    "logisticIT",
    "simple_or",
    "coral",
    "corn",
    "clm",
    "nn_rank",
    "condor",
    "or_cnn",
]


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def calculate_results(y_true, y_pred, include_accuracy=True) -> list[float]:
    """
    Calculates evaluation metrics for predicted values compared to true values.

    """
    results = [
        root_mean_squared_error(y_true, y_pred),
        rmse_macroaveraged(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        mae_macroaveraged(y_true, y_pred),
        mse_macroaveraged(y_true, y_pred),
        somers_d(y_true, y_pred),
        None,
        None,
    ]
    if include_accuracy:
        y_pred_rounded = [int(i) for i in y_pred]
        results[-2] = accuracy_score(y_true, y_pred_rounded)
        results[-1] = accuracy_at_k(y_true, y_pred, k=1)
    return results


def get_index():
    iterables = [
        ["no_rounding", "mathematical"],
        [
            "rmse",
            "rmse_macroaveraged",
            "mae",
            "mae_macroaveraged",
            "mse_macroaveraged",
            "somers_d",
            "accuracy",
            "accuracy1",
        ],
    ]
    return pd.MultiIndex.from_product(iterables, names=["metrics", "model"])


def is_ordinal_model(model_name) -> bool:
    if model_name in ORDINAL_MODELS:
        return True
    # in case of expanding window each model_name has also number at the end (e.g. linear_regression_0)
    name = re.match("(.*)_\d+$", model_name)
    if name and name in ORDINAL_MODELS:
        return True
    return False


def calculate_all_results_types(
    y_train, y_pred_train, y_test, y_pred_test, model_name: str
) -> tuple[list, list]:
    """
    Calculates evaluation metrics for predicted values compared to true values
    for continous and rounded results.
    """
    if is_ordinal_model(model_name):
        # rounding not needed, copy paste not rounded results
        n = 2
        train_results = n * calculate_results(y_train, y_pred_train)
        test_results = n * calculate_results(y_test, y_pred_test)

        return train_results, test_results

    train_results = calculate_results(
        y_train, y_pred_train, include_accuracy=False
    ) + calculate_results(
        y_train, round_single_threshold_results(y_pred_train, threshold=0.5)
    )

    test_results = calculate_results(
        y_test, y_pred_test, include_accuracy=False
    ) + calculate_results(
        y_test, round_single_threshold_results(y_pred_test, threshold=0.5)
    )

    return train_results, test_results


def get_model_results(
    model,
    y_train,
    X_train,
    y_test,
    X_test,
    model_name="",
    results_dir: str = MODELS_RESULTS_DIR,
    chronological: bool = True,
    set_name: str = "full",
) -> tuple[list[float], list[float]]:
    """
    Calculates and compares evaluation metrics for different rounding strategies based on a machine learning model.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if isinstance(model, OrderedResultsWrapper):
        y_pred_train = model.predict(X_train).idxmax(axis=1).to_numpy()
        y_pred_test = model.predict(X_test).idxmax(axis=1).to_numpy()
    else:
        y_pred_train = np.array(model.predict(X_train))
        y_pred_test = np.array(model.predict(X_test))

    pd.DataFrame(y_pred_train).to_csv(
        os.path.join(
            results_dir,
            f"{'' if chronological else RANDOM_PREFIX}{set_name}_{model_name}_train.csv",
        ),
        index=False,
        header=False,
    )
    pd.DataFrame(y_pred_test).to_csv(
        os.path.join(
            results_dir,
            f"{'' if chronological else RANDOM_PREFIX}{set_name}_{model_name}_test.csv",
        ),
        index=False,
        header=False,
    )

    return calculate_all_results_types(
        y_train, y_pred_train, y_test, y_pred_test, model_name
    )


def save_loss_curves(
    model_name: str, set_name: str, chronological: bool, model: ModelType
) -> None:
    if model_name in NEURAL_NETWORK_MODELS and isinstance(model, GridSearchCV):
        try:
            history = pd.DataFrame(model.best_estimator_["model"].history)
            history = history.drop(columns=["batches"])
            history.to_csv(
                os.path.join(
                    os.path.join(LOSS_RESULTS_DIR),
                    f"{'' if chronological else RANDOM_PREFIX}{set_name}_{model_name}.csv",
                ),
                index=False,
            )
        except AttributeError as e:
            print(f"Error while getting history: {e}")
            raise


def train_and_evaluate_models(
    models: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_files: tuple[str, str],
    chronological: bool = True,
    set_name: str = "full",
) -> tuple[DataFrame, DataFrame]:
    """
    Trains and evaluates multiple machine learning models.
    """
    all_train_results = []
    all_test_results = []
    train_results_file, test_results_file = save_files
    columns = get_index()
    n_features = X_train.shape[1]

    # there are models that require the level to be non-negative
    y_train += 1
    y_test += 1

    for i, model_name in enumerate(models):
        print(model_name)
        model = get_fitted_model(model_name, X_train, y_train, n_features)
        model_train_results, model_test_results = get_model_results(
            model,
            y_train,
            X_train,
            y_test,
            X_test,
            model_name=model_name,
            chronological=chronological,
            set_name=set_name,
        )

        all_train_results.append(model_train_results)
        all_test_results.append(model_test_results)

        columns = get_index()
        pd.DataFrame(
            data=all_train_results, index=models[: i + 1], columns=columns
        ).to_csv(train_results_file)
        pd.DataFrame(
            data=all_test_results, index=models[: i + 1], columns=columns
        ).to_csv(test_results_file)
        save_loss_curves(model_name, set_name, chronological, model)

    return pd.DataFrame(
        data=all_test_results, index=models, columns=columns
    ), pd.DataFrame(data=all_train_results, index=models, columns=columns)


def calculate_results_from_files(
    models: list[str],
    y_train: pd.Series,
    y_test: pd.Series,
    save_files: tuple[str, str],
) -> tuple[DataFrame, DataFrame]:
    """
    Trains and evaluates multiple machine learning models and compares different rounding strategies.
    """
    all_train_results = []
    all_test_results = []
    train_results_file, test_results_file = save_files
    columns = get_index()

    # there are models that require the level to be non-negative
    y_train += 1
    y_test += 1

    for i, model_name in enumerate(models):
        print(model_name)

        y_pred_train = pd.read_csv(
            os.path.join(MODELS_RESULTS_DIR, f"{SET_NAME}_{model_name}_train.csv"),
            index_col=False,
            header=None,
            names=["predictions"],
        )["predictions"].to_numpy()
        y_pred_test = pd.read_csv(
            os.path.join(MODELS_RESULTS_DIR, f"{SET_NAME}_{model_name}_test.csv"),
            index_col=False,
            header=None,
            names=["predictions"],
        )["predictions"].to_numpy()

        model_train_results, model_test_results = calculate_all_results_types(
            y_train, y_pred_train, y_test, y_pred_test, model_name
        )

        all_train_results.append(model_train_results)
        all_test_results.append(model_test_results)

        columns = get_index()
        pd.DataFrame(
            data=all_train_results, index=models[: i + 1], columns=columns
        ).to_csv(train_results_file)
        pd.DataFrame(
            data=all_test_results, index=models[: i + 1], columns=columns
        ).to_csv(test_results_file)

    return pd.DataFrame(
        data=all_test_results, index=models, columns=columns
    ), pd.DataFrame(data=all_train_results, index=models, columns=columns)


def calculate_and_save_final_results(
    all_test_results: dict[str, list[list[float | None]]],
    all_train_results: dict[str, list[list[float | None]]],
    columns: MultiIndex,
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_results_test = calculate_average_and_std(all_test_results, columns, models)
    final_results_train = calculate_average_and_std(all_train_results, columns, models)
    final_results_test.to_csv(
        os.path.join(
            EXPANDING_WINDOW_DIR,
            "test_results.csv",
        )
    )
    final_results_train.to_csv(
        os.path.join(
            EXPANDING_WINDOW_DIR,
            "train_results.csv",
        )
    )

    return final_results_test, final_results_train


def expanding_window_train_and_evaluate_models(
    models: list[str], dataframes: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_train_results = defaultdict(list)
    all_test_results = defaultdict(list)
    columns = get_index()
    X_train = dataframes[0].copy()
    y_train = X_train.pop("level").to_numpy(copy=True)
    train_books = X_train.pop("book").unique()
    n_features = X_train.shape[1]

    # there are models that require the level to be non-negative
    y_train += 1

    for df_number, test in enumerate(dataframes[1:]):
        X_test = test.copy()
        y_test = X_test.pop("level").to_numpy() + 1
        test_books = X_test.pop("book").unique()

        print(f"TRAIN: {len(y_train)}")
        print(f"TEST: {len(y_test)}")
        for i, model_name in enumerate(models):
            print(model_name)
            model = get_fitted_model(model_name, X_train, y_train, n_features)
            model_train_results, model_test_results = get_model_results(
                model,
                y_train,
                X_train,
                y_test,
                X_test,
                model_name=f"{model_name}_{df_number}",
                results_dir=os.path.join(
                    EXPANDING_WINDOW_DIR, model_name, "models_predictions"
                ),
            )

            all_train_results[model_name].append(model_train_results)
            all_test_results[model_name].append(model_test_results)

            results_path = os.path.join(
                EXPANDING_WINDOW_DIR,
                model_name,
            )
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            pd.DataFrame(data=all_train_results[model_name], columns=columns).to_csv(
                os.path.join(results_path, "train_results.csv")
            )
            pd.DataFrame(data=all_test_results[model_name], columns=columns).to_csv(
                os.path.join(results_path, "test_results.csv")
            )

        X_train = pd.concat([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])
        train_books = np.concatenate([train_books, test_books])

    return calculate_and_save_final_results(
        all_test_results, all_train_results, columns, models
    )


def calculate_final_scores_from_files(
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_test_results = defaultdict(list)
    all_train_results = defaultdict(list)
    columns = None

    for i, model_name in enumerate(models):
        results_path = os.path.join(
            EXPANDING_WINDOW_DIR,
            model_name,
        )
        test_results = pd.read_csv(
            os.path.join(results_path, "test_results.csv"),
            header=[0, 1],
            index_col=[0],
        )
        all_test_results[model_name] = test_results.to_numpy().tolist()
        all_train_results[model_name] = (
            pd.read_csv(
                os.path.join(results_path, "train_results.csv"),
                header=[0, 1],
                index_col=[0],
            )
            .to_numpy()
            .tolist()
        )
        columns = test_results.columns

    if columns is None:
        return
    return calculate_and_save_final_results(
        all_test_results, all_train_results, columns, models
    )
