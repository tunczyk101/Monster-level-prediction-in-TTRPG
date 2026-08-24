import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier

from training.constants import RANDOM_STATE


class SimpleOrdinalClassification(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.default_params = {
            "max_features": 0.3,
            "n_estimators": 100,
            "criterion": "gini",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        self.default_params.update(kwargs)

        for param, value in self.default_params.items():
            setattr(self, param, value)

        self.base_model_ = RandomForestClassifier(**self.default_params)
        self.model_ = OrdinalClassifier(self.base_model_)

    def fit(self, X, y):
        self.model_.fit(X, y)

    def predict(self, X):
        return self.model_.predict(X)

    def get_params(self, deep=True):
        return self.base_model_.get_params(deep)

    def set_params(self, **params):
        self.base_model_.set_params(**params)

        self.model_ = OrdinalClassifier(self.base_model_)

        return self


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    #  https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
    #  by https://github.com/mosh98
    """
    A classifier that can be trained on a range of classes.
    @param classifier: A scikit-learn classifier.
    """

    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}
        self.uniques_class = None

    def fit(self, X, y):
        self.uniques_class = np.sort(np.unique(y))
        assert self.uniques_class.shape[0] >= 3, (
            f"OrdinalClassifier needs at least 3 classes, only {self.uniques_class.shape[0]} found"
        )

        for i in range(self.uniques_class.shape[0] - 1):
            binary_y = (y > self.uniques_class[i]).astype(np.uint8)

            clf = clone(self.clf)
            clf.fit(X, binary_y)
            self.clfs[i] = clf

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        predicted = [
            self.clfs[k].predict_proba(X)[:, 1].reshape(-1, 1) for k in self.clfs
        ]

        p_x_first = 1 - predicted[0]
        p_x_last = predicted[-1]
        p_x_middle = [
            predicted[i] - predicted[i + 1] for i in range(len(predicted) - 1)
        ]

        probs = np.hstack([p_x_first, *p_x_middle, p_x_last])

        return probs

    def set_params(self, **params):
        self.clf.set_params(**params)
        for _, clf in self.clfs.items():
            clf.set_params(**params)
