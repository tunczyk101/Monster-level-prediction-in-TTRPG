from sklearn.metrics import mean_absolute_error


def clm_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred.argmax(axis=1))
