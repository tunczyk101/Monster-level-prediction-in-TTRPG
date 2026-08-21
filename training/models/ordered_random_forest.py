import numpy as np
from orf import OrderedForest


class OrderedRandomForest(OrderedForest):
    def predict(self, X, *args, **kwargs):
        predictions = super().predict(X)["predictions"]

        return np.argmax(predictions, axis=1)
