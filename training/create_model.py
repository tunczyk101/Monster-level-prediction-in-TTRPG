from typing import Any, Callable

import gpflow
import lightgbm as lightgbm
import numpy as np
from optuna import Trial, create_study
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import torch
from mord import LogisticAT, LogisticIT
from orf import OrderedForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    QuantileRegressor,
    RidgeCV,
)
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from skorch import NeuralNet
from spacecutter.callbacks import AscensionCallback
from spacecutter.models import OrdinalLogisticModel
from statsmodels.miscmodels.ordinal_model import OrderedModel
from torch import nn

from training.constants import NUM_CLASSES, RANDOM_STATE
from training.losses import CondorLoss, WeightedBCELoss
from training.models.condor import Condor, CondorNeuralNet
from training.models.coral_corn import (
    CORAL_MLP,
    CORN_MLP,
    DEVICE,
    SkorchCORAL,
    SkorchCORN,
)
from training.models.gpor import GPOR
from training.models.nn_rank import NeuralNetNNRank, NNRank
from training.models.or_cnn import ORCNN, NeuralNetORCNN
from training.models.ordered_models import LinearOrdinalModel
from training.models.simple_ordinal_classification import SimpleOrdinalClassification
from training.models.clm.losses import CumulativeLinkLoss
from training.models.clm.models import (
    SpacecutterGridSearchCV,
    get_spacecutter_predictor,
)
from training.score_functions import (
    orf_mean_absolute_error,
    spacecutter_mean_absolute_error,
)

BASE_LIGHTGBM_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "l2",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
}

ModelType = RidgeCV | GridSearchCV | lightgbm.Booster | OrderedModel
Fold = tuple[np.ndarray, np.ndarray]


def get_fitted_model(
    classifier_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = 53,
) -> ModelType:
    """
    Creates chosen model, performs tuning and fits\n
    :param X_train: train set with features to use during fitting
    :param y_train: train set with values to predict
    :param classifier_name: name of a chosen classifier:
            linear_regression or random_forest
    :return: trained classifier of a chosen type
    """
    if classifier_name == "lightgbm":
        return lightgbm_fit(X_train, y_train)

    model = create_model(classifier_name, n_features, y_train)
    model.fit(X_train, y_train)

    return model


def create_min_max_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("model", model),
        ]
    )


def create_grid_search(
    model: BaseEstimator,
    param_grid: dict[str, Any],
    scoring: str | Callable = "neg_mean_absolute_error",
) -> GridSearchCV:
    pipeline = create_min_max_pipeline(model)

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        verbose=2,
        return_train_score=True,
        n_jobs=-1,
    )


def create_model(classifier_name: str, n_features: int, y_train: np.ndarray):
    """
    Creates chosen model\n
    :param classifier_name: name of a chosen classifier:
            linear_regression or random_forest
    :return: chosen classifier
    """
    match classifier_name:
        case "linear_regression":
            model = create_min_max_pipeline(LinearRegression())
        case "linear_regression_ridge":
            model = create_min_max_pipeline(RidgeCV(alphas=np.linspace(1e-3, 1, 10000)))
        case "lad_regression":
            hyper_params = [{"alpha": np.linspace(0.0, 1e-3, 100)}]

            reg_lad = QuantileRegressor(quantile=0.5, solver="highs")

            model = create_grid_search(reg_lad, hyper_params)
        case "huber_regression":
            huber = HuberRegressor(max_iter=1000)
            hyper_params = {"alpha": np.linspace(1e-3, 1, 1000)}

            model = create_grid_search(huber, hyper_params)
        case "linear_svm":
            clf_linear_svr = LinearSVR(
                loss="epsilon_insensitive", max_iter=10000, random_state=0
            )
            hyper_params = {"C": np.linspace(10, 30, num=20)}

            model = create_grid_search(clf_linear_svr, hyper_params)
        case "kernel_svm":
            svm = SVR(kernel="rbf", max_iter=10000)
            hyper_params = {"C": np.linspace(1, 10, num=100)}

            model = create_grid_search(svm, hyper_params)
        case "knn":
            knn = KNeighborsRegressor()

            hyper_params = {
                "leaf_size": list(range(50, 100, 10)),
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "manhattan", "euclidean"],
                "n_neighbors": [1, 3],
            }

            model = create_grid_search(knn, hyper_params)
        case "random_forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
            hyper_params = {
                "max_features": ["sqrt", 0.3],
                "n_estimators": [100, 200, 500],
                "criterion": ["squared_error", "absolute_error", "friedman_mse"],
            }
            model = create_grid_search(rf, hyper_params)
        case "ordered_random_forest":
            rf = OrderedForest(random_state=RANDOM_STATE, n_jobs=-1)
            hyper_params = {
                "max_features": [0.3],
                "min_samples_leaf": [i for i in range(2, 8)],
                "n_estimators": [100, 200, 500],
                "honesty": [False],
                "replace": [True],
            }
            model = create_grid_search(
                rf,
                hyper_params,
                scoring=make_scorer(orf_mean_absolute_error, greater_is_better=False),
            )
        case "logisticAT":
            hyper_params = [{"alpha": np.linspace(0.0, 1e-3, 100)}]

            logistic_model = LogisticAT()

            model = create_grid_search(logistic_model, hyper_params)
        case "logisticIT":
            hyper_params = [{"alpha": np.linspace(0.0, 1e-3, 100)}]

            logistic_model = LogisticIT()

            model = create_grid_search(logistic_model, hyper_params)
        case "linear_ordinal_model_probit":
            model = create_linear_ordinal_model("probit")
        case "linear_ordinal_model_logit":
            model = create_linear_ordinal_model("logit")
        case "simple_or":
            hyper_params = {
                "max_features": ["sqrt", 0.3],
                "n_estimators": [100, 200, 500],
                "criterion": ["gini", "entropy"],
            }
            model = create_grid_search(
                SimpleOrdinalClassification(),
                hyper_params,
            )
        case "gpor":
            hyper_params = {
                "maxiter": [100],
                "kernel": [gpflow.kernels.ArcCosine()],
            }
            model = create_grid_search(
                GPOR(),
                hyper_params,
            )
        case "coral":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "lr": [1e-3, 1e-2, 1e-1],
            }
            model = create_grid_search(
                SkorchCORAL(
                    module=CORAL_MLP,
                    module__input_size=n_features,
                    module__num_classes=NUM_CLASSES,
                    max_epochs=40,
                    lr=0.05,
                    optimizer=torch.optim.AdamW,
                    iterator_train__shuffle=True,
                    device=DEVICE,
                ),
                hyper_params,
            )
        case "corn":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "lr": [1e-3, 1e-2, 1e-1],
            }
            model = create_grid_search(
                SkorchCORN(
                    module=CORN_MLP,
                    module__input_size=n_features,
                    module__num_classes=NUM_CLASSES,
                    max_epochs=40,
                    lr=0.05,
                    optimizer=torch.optim.AdamW,
                    iterator_train__shuffle=True,
                    device=DEVICE,
                    criterion=nn.CrossEntropyLoss,
                ),
                hyper_params,
            )
        case "clm":
            hyper_params = {
                "lr": [1e-3, 1e-2, 1e-1],
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
            }
            predictor = get_spacecutter_predictor(n_features)

            estimator = NeuralNet(
                module=OrdinalLogisticModel,
                module__predictor=predictor,
                module__num_classes=NUM_CLASSES,
                criterion=CumulativeLinkLoss,
                optimizer=torch.optim.AdamW,
                device=DEVICE,
                max_epochs=40,
                callbacks=[
                    ("ascension", AscensionCallback()),
                ],
            )

            model = SpacecutterGridSearchCV(
                estimator=create_min_max_pipeline(estimator),
                param_grid=hyper_params,
                scoring=make_scorer(
                    spacecutter_mean_absolute_error,
                    greater_is_better=False,
                    needs_proba=True,
                ),
                return_train_score=True,
                n_jobs=-1,
            )
        case "nn_rank":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "optimizer__lr": [1e-3, 1e-2, 1e-1],
            }
            model = create_grid_search(
                NeuralNetNNRank(
                    module=NNRank,
                    module__input_size=n_features,
                    criterion=torch.nn.BCELoss,
                    optimizer=torch.optim.AdamW,
                    device=DEVICE,
                    max_epochs=40,
                ),
                hyper_params,
            )
        case "condor":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "optimizer__lr": [1e-3, 1e-2, 1e-1],
            }
            model = create_grid_search(
                CondorNeuralNet(
                    module=Condor,
                    module__input_size=n_features,
                    criterion=CondorLoss,
                    optimizer=torch.optim.AdamW,
                    device=DEVICE,
                    max_epochs=40,
                ),
                hyper_params,
            )
        case "or_cnn":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "optimizer__lr": [1e-3, 1e-2, 1e-1],
            }
            model = create_grid_search(
                NeuralNetORCNN(
                    module=ORCNN,
                    module__input_size=n_features,
                    criterion__y_train=y_train,
                    criterion=WeightedBCELoss,
                    optimizer=torch.optim.AdamW,
                    device=DEVICE,
                    max_epochs=40,
                ),
                hyper_params,
            )
        case _:
            raise ValueError(f"Classifier {classifier_name} is unsupported")

    return model


def lightgbm_objective(
    trial: Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[Fold],
):
    params = {
        **BASE_LIGHTGBM_PARAMS,
        # Keep the parameters that Optuna tunes.
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-3,
            0.2,
            log=True,
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves",
            10,
            200,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            15,
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            5,
            100,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        ),
    }

    fold_scores = []

    for train_idx, valid_idx in folds:
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        scaler = MinMaxScaler()

        X_fold_train_scaled = scaler.fit_transform(X_fold_train)

        X_fold_valid_scaled = scaler.transform(X_fold_valid)

        lgb_train = lightgbm.Dataset(
            X_fold_train_scaled,
            label=y_fold_train,
        )

        lgb_valid = lightgbm.Dataset(
            X_fold_valid_scaled,
            label=y_fold_valid,
            reference=lgb_train,
        )

        model = lightgbm.train(
            params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_valid],
            callbacks=[
                lightgbm.early_stopping(
                    100,
                    verbose=False,
                ),
            ],
        )

        predictions = model.predict(
            X_fold_valid_scaled,
            num_iteration=model.best_iteration,
        )

        mae = np.mean(np.abs(y_fold_valid.to_numpy() - predictions))

        fold_scores.append(mae)

    return float(np.mean(fold_scores))


def lightgbm_fit(X_train, y_train) -> lightgbm.Booster:
    """
    Performs tuning and fits lightgbm model\n
    :param X_train: train set with features to use during fitting
    :param y_train: train set with values to predict
    :return: trained lightgbm
    """
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    folds = list(
        KFold(
            n_splits=5,
        ).split(X_train)
    )

    study = create_study(
        direction="minimize",
    )
    study.optimize(
        lambda trial: lightgbm_objective(
            trial,
            X_train,
            y_train,
            folds,
        ),
        n_trials=50,
    )

    best_params = study.best_params

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lgb_train = lightgbm.Dataset(
        X_train_scaled,
        label=y_train,
    )

    final_params = {
        **BASE_LIGHTGBM_PARAMS,
        **best_params,
    }

    lgb_tuned = lightgbm.train(
        final_params,
        lgb_train,
        num_boost_round=10000,
    )
    lgb_tuned.scaler = scaler

    return lgb_tuned


def create_linear_ordinal_model(distr: str) -> GridSearchCV:
    model = LinearOrdinalModel(distr=distr)
    hyper_params = {"offset": np.linspace(0.25, 1.25, 11)}
    model = create_grid_search(model, hyper_params)

    return model
