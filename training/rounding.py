from collections import defaultdict
from enum import auto
from queue import PriorityQueue
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from strenum import StrEnum

from dataset.constants import RANDOM_STATE
from training.constants import MAX_LVL, MIN_LVL


class RoundingType(StrEnum):
    mathematical = auto()
    single = auto()
    optuna = auto()
    graph = auto()


def get_rounding_func(
    rounding_type: RoundingType,
    y_pred: np.ndarray | pd.Series,
    y_true: pd.Series,
    thresholds: list[float],
) -> Callable[[np.ndarray | pd.Series], np.ndarray]:
    func: Callable[[np.ndarray | pd.Series], np.ndarray]

    match rounding_type:
        case RoundingType.mathematical:
            func = lambda x: round_single_threshold_results(x, 0.5)  # noqa E731
        case RoundingType.single:
            threshold = find_single_best_threshold(y_pred, y_true, thresholds)
            func = lambda x: round_single_threshold_results(x, threshold)  # noqa E731
        case RoundingType.optuna:
            best_thresholds = find_best_thresholds(
                y_pred, y_true, (min(thresholds), min(thresholds))
            )
            func = lambda x: round_results_multiple_threshold(x, best_thresholds)  # noqa E731
        case RoundingType.graph:
            best_thresholds = find_graph_rounding(y_pred, y_true, thresholds)
            func = lambda x: round_results_multiple_threshold(x, best_thresholds)  # noqa E731

    return func


def round_single_threshold_results(
    y_pred: np.ndarray | pd.Series, threshold: float
) -> np.ndarray:
    """
    Rounds predictions based on a specified threshold with maximum rounded value at 21.

    :param y_pred: Predicted values
    :param threshold: Threshold for rounding
    :return: An array of the rounded predictions
    """
    threshold_predict = np.where(
        (y_pred % 1) >= threshold, np.ceil(y_pred), np.floor(y_pred)
    ).astype("int")
    return np.where(threshold_predict > MAX_LVL, MAX_LVL, threshold_predict)


def find_single_best_threshold(
    y_pred: np.ndarray | pd.Series, y_true: pd.Series, thresholds: list[float]
) -> float:
    """
    Finds the best threshold for rounding predictions to minimize the mean absolute error (MAE).

    :param y_pred: Predicted values
    :param y_true: True values
    :param thresholds:  A list of threshold values to test, each between 0.0 and 1.0
    :return: The best threshold
    """
    best = (thresholds[0], 21)

    for threshold in thresholds:
        if threshold < 0 or threshold >= 1.0:
            raise ValueError(
                f"Incorrect threshold value. Should be between 0.0 and 1.0 but is {threshold}."
            )

        threshold_predict = round_single_threshold_results(y_pred, threshold)

        mae = mean_absolute_error(y_true, threshold_predict)
        if mae < best[1]:
            best = threshold, mae

    return best[0]


def round_prediction(predicted: float, threshold: float) -> int:
    """
    Rounds a single predicted value based on a specified threshold.
    Minimum possible rounded value is -1 and maximum possible rounded value is 21.

    :param predicted: Predicted value
    :param threshold: Threshold for rounding
    :return: Rounded value
    """
    if threshold is None:
        return min(21, max(-1, predicted))

    round_val = predicted // 1

    if predicted % 1 >= threshold:
        round_val += 1
    return round_val


def round_prediction_error(predicted: float, true: float, threshold: float) -> float:
    """
     Calculates the absolute error between the true value and the rounded predicted value based on a specified threshold.

    :param predicted: Predicted value
    :param true: True value
    :param threshold: Threshold for rounding
    :return: The absolute error between the true value and the rounded predicted value.
    """
    return abs(true - round_prediction(predicted, threshold))


def round_results_multiple_threshold(
    y_predicted: np.ndarray, thresholds: dict[int, float]
) -> list[int]:
    """
    Rounds a list of predicted values based on multiple thresholds specified for each integer part of the prediction.

    :param y_predicted: Predicted values to be rounded
    :param thresholds: Dictionary with thresholds
    :return: A list of rounded values.
    """
    return [
        round_prediction(prediction, thresholds.get(prediction // 1))
        for prediction in y_predicted
    ]


def objective(
    trial: optuna.trial.Trial,
    y_true: list[int],
    y_predicted: list[float],
    thresholds: tuple[float, float],
) -> float:
    """
    Objective function for optimizing thresholds to minimize the mean absolute error.

    :param trial: Optimization trial object used to suggest threshold values
    :param y_true: List of true values
    :param y_predicted: List of predicted values
    :param thresholds: Tuple containing the lower and upper bounds for the thresholds
    :return: The mean absolute error (MAE) between the true values and the rounded predicted values
    """
    level_thresholds = {
        i: trial.suggest_float(f"level_{i}", thresholds[0], thresholds[1])
        for i in range(-1, 21)
    }
    n = len(y_true)
    sum_prediction_error = sum(
        [
            round_prediction_error(
                y_predicted[i], y_true[i], level_thresholds.get(y_predicted[i] // 1)
            )
            for i in range(n)
        ]
    )
    mean_prediction_error = sum_prediction_error / n
    return mean_prediction_error


def find_best_thresholds(
    y_predicted: list[float],
    y_true: list[int],
    thresholds: tuple[float, float] = (0, 1),
) -> dict[int, float]:
    """
    Finds the best thresholds for rounding predicted values to minimize the mean absolute error (MAE).

    :param y_predicted: List of predicted values
    :param y_true: List of true values
    :param thresholds: Tuple containing the lower and upper bounds for the thresholds
    :return: Dictionary mapping level to their optimized thresholds.
    """
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, y_true, y_predicted, thresholds), n_trials=100
    )
    return {i - 1: threshold for i, threshold in enumerate(study.best_params.values())}


def get_edges_cost(
    level: int, thresholds: list[float], y_pred: list[float], y_true: list[int]
) -> list[tuple[float, float]]:
    """
    Calculates the cost for each threshold at a given level.

    :param level: Level - integer part of the predicted values to consider
    :param thresholds: Tuple containing the lower and upper bounds for the thresholds
    :param y_pred: List of predicted values
    :param y_true: List of true values
    :return: List of tuples representing each edge (threshold) from this edge and corresponding cost of this edge (MAE).
    """
    lvl_pred = [pred for pred in range(len(y_pred)) if y_pred[pred] // 1 == level]
    n = len(lvl_pred)
    moves = []

    if n == 0:
        # no predictions for a given range, return classic math rounding threshold
        return [(0.5, 0)]

    for threshold in thresholds:
        sum_prediction_error = sum(
            [round_prediction_error(y_pred[i], y_true[i], threshold) for i in lvl_pred]
        )
        mean_prediction_error = sum_prediction_error / n
        moves.append((threshold, mean_prediction_error))

    return moves


def find_graph_rounding(
    y_pred: list[float], y_true: list[int], thresholds: list[float]
) -> dict[int, float]:
    print("Graph thresholds")
    """
     Finds the best thresholds for rounding using a graph-based approach.

    :param y_pred: List of predicted values
    :param y_true: List of true values
    :param thresholds: List of threshold values to consider for rounding
    :return: Dictionary mapping level to their optimized thresholds
    """
    q = PriorityQueue()
    final_thresholds = {}
    q.put((0, 0))

    while not q.empty():
        cost, level = q.get()

        if level == 22:
            return final_thresholds

        edges_cost = get_edges_cost(level, thresholds, y_pred, y_true)
        threshold, next_edge_cost = min(edges_cost, key=lambda x: x[1])
        final_thresholds[level] = threshold
        q.put((cost + next_edge_cost, level + 1))


def create_graph(
    y_test: np.array,
    y_pred: np.array,
    thresholds: list[float],
    min_level: int = MIN_LVL,
    max_level: int = MAX_LVL,
) -> dict[tuple[int, int], list]:
    # V_i_r -> i - level, r -> threshold
    G = defaultdict(list)
    G[0, -np.inf] = []

    for level in range(min_level, max_level + 1):
        print(f"LEVEL {level}")
        if level == min_level:
            start_thresholds = [-np.inf]
        else:
            start_thresholds = [t + (level - 1) for t in thresholds]

        for start_threshold in start_thresholds:
            if level == max_level:
                end_thresholds = [np.inf]
            else:
                end_thresholds = [t + level for t in thresholds]

            for end_threshold in end_thresholds:
                v = (level + 1, end_threshold)
                ids = np.logical_and(y_pred > start_threshold, y_pred < end_threshold)
                cost = 0
                if any(ids):
                    rounded_values = np.array([level for _ in range(len(y_test[ids]))])
                    cost = np.sum(np.abs(np.subtract(y_test[ids], rounded_values)))

                G[(level, start_threshold)].append((v, cost))

    G[(max_level + 1, np.inf)] = []
    return G


def get_dijkstry_path(parent: dict) -> list[float]:
    current_v = (MAX_LVL + 1, np.inf)  # last v
    thresholds = {}
    while current_v[0] != MIN_LVL + 1:
        current_v = parent[current_v]
        thresholds[current_v[0] - 1] = current_v[1]

    return thresholds


def fill_dict_with_value(keys: tuple[int, int], value: int) -> dict:
    result_dict = dict()
    for k in keys:
        result_dict[k] = value
    return result_dict


def dijkstry(
    G, s
) -> tuple[dict[tuple[int, int], int | tuple[int, int]], dict[tuple[int, int], float]]:
    parent = fill_dict_with_value(G.keys(), -1)
    value = fill_dict_with_value(G.keys(), np.inf)
    value[s] = 0
    q = PriorityQueue()
    q.put((0, s))

    while not q.empty():
        v, p = q.get()
        for i in G[p]:
            if value[i[0]] > v + i[1]:
                value[i[0]] = v + i[1]
                parent[i[0]] = p
                q.put((value[i[0]], i[0]))

    return parent, value


def get_all_possible_thresholds(thresholds: list[float]) -> list[float]:
    result_thresholds = []

    for i in range(MIN_LVL, MAX_LVL):
        for threshold in thresholds:
            result_thresholds.append(i + threshold)

    return result_thresholds
