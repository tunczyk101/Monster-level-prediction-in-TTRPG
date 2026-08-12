import numpy as np
import pandas as pd

from training.constants import MAX_LVL


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
