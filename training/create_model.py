from typing import Any, Callable

import gpflow
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import torch
from mord import LogisticAT, LogisticIT
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    RidgeCV,
)
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from skorch import NeuralNet
from spacecutter.callbacks import AscensionCallback
from spacecutter.models import OrdinalLogisticModel
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

ModelType = Pipeline | GridSearchCV


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
        case "linear_regression_ridge":
            model = create_min_max_pipeline(RidgeCV(alphas=np.linspace(1e-3, 1, 10000)))
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

        case "lightgbm":
            lgbm = LGBMRegressor(
                objective="regression",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )

            hyper_params = {
                "model__n_estimators": [100, 300, 500],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [7, 15, 31],
                "model__min_child_samples": [10, 20, 40],
                "model__max_depth": [-1, 5, 10],
            }

            model = create_grid_search(lgbm, hyper_params)
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


def create_linear_ordinal_model(distr: str) -> GridSearchCV:
    model = LinearOrdinalModel(distr=distr)
    hyper_params = {"model__offset": np.linspace(0.25, 1.25, 11)}
    model = create_grid_search(model, hyper_params)

    return model
