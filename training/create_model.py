from typing import Any, Callable

import gpflow
from lightgbm import Booster, Dataset, early_stopping, log_evaluation
import lightgbm
import numpy as np
import optuna.integration.lightgbm as opt_lgb
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import torch
from mord import LogisticAT, LogisticIT
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
from training.models.ordered_random_forest import OrderedRandomForest
from training.models.scaled_lightgbm import ScaledBooster
from training.models.simple_ordinal_classification import SimpleOrdinalClassification
from training.models.clm.losses import CumulativeLinkLoss
from training.models.clm.models import (
    CLMGridSearchCV,
    get_clm_predictor,
)
from training.score_functions import (
    clm_mean_absolute_error,
)

FoldEntry = dict[str, np.ndarray | MinMaxScaler]
FoldLookup = dict[tuple[int, ...], FoldEntry]

ModelType = RidgeCV | GridSearchCV | lightgbm.Booster | OrderedModel


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
            hyper_params = [{"model__alpha": np.linspace(0.0, 1e-3, 100)}]

            reg_lad = QuantileRegressor(quantile=0.5, solver="highs")

            model = create_grid_search(reg_lad, hyper_params)
        case "huber_regression":
            huber = HuberRegressor(max_iter=1000)
            hyper_params = {"model__alpha": np.linspace(1e-3, 1, 1000)}

            model = create_grid_search(huber, hyper_params)
        case "linear_svm":
            clf_linear_svr = LinearSVR(
                loss="epsilon_insensitive", max_iter=10000, random_state=0
            )
            hyper_params = {"model__C": np.linspace(10, 30, num=20)}

            model = create_grid_search(clf_linear_svr, hyper_params)
        case "kernel_svm":
            svm = SVR(kernel="rbf", max_iter=10000)
            hyper_params = {"model__C": np.linspace(1, 10, num=100)}

            model = create_grid_search(svm, hyper_params)
        case "knn":
            knn = KNeighborsRegressor()

            hyper_params = {
                "model__leaf_size": list(range(50, 100, 10)),
                "model__weights": ["uniform", "distance"],
                "model__metric": ["minkowski", "manhattan", "euclidean"],
                "model__n_neighbors": [1, 3],
            }

            model = create_grid_search(knn, hyper_params)
        case "random_forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
            hyper_params = {
                "model__max_features": ["sqrt", 0.3],
                "model__n_estimators": [100, 200, 500],
                "model__criterion": ["squared_error", "absolute_error", "friedman_mse"],
            }
            model = create_grid_search(rf, hyper_params)
        case "ordered_random_forest":
            rf = OrderedRandomForest(random_state=RANDOM_STATE, n_jobs=-1)
            hyper_params = {
                "model__max_features": [0.3],
                "model__min_samples_leaf": [i for i in range(2, 8)],
                "model__n_estimators": [100, 200, 500],
                "model__honesty": [False],
                "model__replace": [True],
            }
            model = create_grid_search(
                rf,
                hyper_params,
            )
        case "logisticAT":
            hyper_params = [{"model__alpha": np.linspace(0.0, 1e-3, 100)}]

            logistic_model = LogisticAT()

            model = create_grid_search(logistic_model, hyper_params)
        case "logisticIT":
            hyper_params = [{"model__alpha": np.linspace(0.0, 1e-3, 100)}]

            logistic_model = LogisticIT()

            model = create_grid_search(logistic_model, hyper_params)
        case "linear_ordinal_model_probit":
            model = create_linear_ordinal_model("probit")
        case "linear_ordinal_model_logit":
            model = create_linear_ordinal_model("logit")
        case "simple_or":
            hyper_params = {
                "model__max_features": ["sqrt", 0.3],
                "model__n_estimators": [100, 200, 500],
                "model__criterion": ["gini", "entropy"],
            }
            model = create_grid_search(
                SimpleOrdinalClassification(),
                hyper_params,
            )
        case "gpor":
            hyper_params = {
                "model__maxiter": [100],
                "model__kernel": [gpflow.kernels.ArcCosine()],
            }
            model = create_grid_search(
                GPOR(),
                hyper_params,
            )
        case "coral":
            hyper_params = {
                "model__optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "model__lr": [1e-3, 1e-2, 1e-1],
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
                "model__optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "model__lr": [1e-3, 1e-2, 1e-1],
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
                "model__lr": [1e-3, 1e-2, 1e-1],
                "model__optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
            }
            predictor = get_clm_predictor(n_features)

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

            model = CLMGridSearchCV(
                estimator=create_min_max_pipeline(estimator),
                param_grid=hyper_params,
                scoring=make_scorer(
                    clm_mean_absolute_error,
                    greater_is_better=False,
                    needs_proba=True,
                ),
                return_train_score=True,
                n_jobs=-1,
            )
        case "nn_rank":
            hyper_params = {
                "model__optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "model__optimizer__lr": [1e-3, 1e-2, 1e-1],
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
                "model__optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "model__optimizer__lr": [1e-3, 1e-2, 1e-1],
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
                "model__optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "model__optimizer__lr": [1e-3, 1e-2, 1e-1],
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


def _precompute_folds(X: np.ndarray, y: np.ndarray, kf: KFold) -> FoldLookup:
    """
    Fit MinMaxScaler once per fold on the train split, transform both splits.
    """
    fold_lookup: FoldLookup = {}

    for train_idx, val_idx in kf.split(X):
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        # Key by the validation indices — they uniquely identify each fold
        key = tuple(sorted(val_idx))

        fold_lookup[key] = {
            "X_tr": X_tr,
            "y_tr": y[train_idx],
            "X_va": X_va,
            "y_va": y[val_idx],
            "scaler": scaler,
        }
    return fold_lookup


def _make_fpreproc(
    fold_lookup: FoldLookup,
) -> Callable[
    [Dataset, Dataset, dict[str, Any]], tuple[Dataset, Dataset, dict[str, Any]]
]:
    """Return an fpreproc closure that looks up pre-scaled arrays instead of
    recomputing the scaler on every trial.

    fpreproc is called by cv -> _make_n_folds once per fold per trial.
    Without this approach the scaler would be re-fitted ~340 times on identical
    data across all LightGBMTunerCV stages (7+20+10+6+20+5 trials × 5 folds).
    Here it is fitted exactly 5 times — once during _precompute_folds.
    """

    def fpreproc(
        train_set: Dataset, valid_set: Dataset, params: dict[str, Any]
    ) -> tuple[Dataset, Dataset, dict[str, Any]]:
        # Identify which pre-scaled fold this is via validation indices
        val_key = tuple(valid_set.used_indices)
        fold = fold_lookup[val_key]

        new_train = Dataset(fold["X_tr"], label=fold["y_tr"], free_raw_data=False)
        new_valid = Dataset(
            fold["X_va"], label=fold["y_va"], reference=new_train, free_raw_data=False
        )
        return new_train, new_valid, params

    return fpreproc


def lightgbm_fit(X_train, y_train) -> Booster:
    """
    Performs tuning and fits lightgbm model\n
    :param X_train: train set with features to use during fitting
    :param y_train: train set with values to predict
    :return: trained lightgbm
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    kf = KFold(n_splits=5)

    fold_lookup = _precompute_folds(X, y, kf)

    lgb_train = opt_lgb.Dataset(X_train, y_train, free_raw_data=False)
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
    }
    tuner = opt_lgb.LightGBMTunerCV(
        params,
        lgb_train,
        folds=KFold(n_splits=5),
        num_boost_round=10000,
        fpreproc=_make_fpreproc(fold_lookup),
        callbacks=[early_stopping(100), log_evaluation(100)],
    )
    tuner.run()
    best_params = tuner.best_params

    final_scaler = MinMaxScaler()
    X_scaled = final_scaler.fit_transform(X)
    lgb_train_scaled = Dataset(X_scaled, y)

    lgb_tuned = lightgbm.train(
        best_params,
        lgb_train_scaled,
        num_boost_round=10000,
    )
    return ScaledBooster(lgb_tuned, final_scaler)


def create_linear_ordinal_model(distr: str) -> GridSearchCV:
    model = LinearOrdinalModel(distr=distr)
    hyper_params = {"model__offset": np.linspace(0.25, 1.25, 11)}
    model = create_grid_search(model, hyper_params)

    return model
