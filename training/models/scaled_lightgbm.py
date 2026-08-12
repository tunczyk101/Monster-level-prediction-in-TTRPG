from typing import Any

from lightgbm import Booster
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ScaledBooster(Booster):
    def __init__(self, booster: Booster, scaler: MinMaxScaler) -> None:
        # Bypass Booster.__init__ — copy the trained model's internal state
        # directly by re-loading from its string representation.
        model_str = booster.model_to_string()
        super().__init__(model_str=model_str)
        self._scaler = scaler

    @property
    def scaler(self) -> MinMaxScaler:
        """The MinMaxScaler fitted on the full training set."""
        return self._scaler

    def predict(
        self,
        data: Any,
        *args,
        **kwargs,
    ):
        """Scale data then delegate to the standard lgb.Booster.predict."""
        X_scaled = self._scaler.transform(np.asarray(data))
        return super().predict(X_scaled, *args, **kwargs)
