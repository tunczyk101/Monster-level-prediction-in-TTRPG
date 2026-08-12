import os


RANDOM_STATE = 42
THRESHOLD = 0.33
NUM_CLASSES = 23  # number of possible levels
MIN_LVL = 0
MAX_LVL = 22

SET_NAME = "full"
CHRONOLOGICAL_SPLIT = True

ROUNDING = "mathematical"
RESULTS_DIR = os.path.join("training", "results")
PATH_TO_DATASET_PATTERN = os.path.join(
    "preprocessed_bestiaries", "bestiaries_{set_name}.csv"
)
PATH_TO_DATASET = PATH_TO_DATASET_PATTERN.format(set_name=SET_NAME)

RANDOM_PREFIX = "random_"
PREFIX = "" if CHRONOLOGICAL_SPLIT else RANDOM_PREFIX
RESULT_PATTERN = os.path.join(
    RESULTS_DIR, "{prefix}{set_name}_results_{result_type}.csv"
)
TEST_RESULT_FILE = RESULT_PATTERN.format(
    prefix=PREFIX, set_name=SET_NAME, result_type="test"
)
TRAIN_RESULT_FILE = RESULT_PATTERN.format(
    prefix=PREFIX, set_name=SET_NAME, result_type="train"
)

LOSS_RESULTS_DIR = os.path.join(RESULTS_DIR, "nn_loss")

MODELS_RESULTS_DIR = os.path.join(RESULTS_DIR, "models_predictions")

EXPANDING_WINDOW_DIR = os.path.join("training", "results", "ts")

ALL_MODELS = [
    "linear_regression_ridge",
    "kernel_svm",
    "knn",
    "random_forest",
    "lightgbm",
    "linear_ordinal_model_probit",
    "linear_ordinal_model_logit",
    "ordered_random_forest",
    "logisticAT",
    "logisticIT",
    "simple_or",
    "gpor",
    "coral",
    "corn",
    "clm",
    "nn_rank",
    "condor",
    "or_cnn",
]
SET_COMPARISION_MODELS = [
    "knn",
    "linear_regression_ridge",
    "random_forest",
    "lightgbm",
    "simple_or",
    "ordered_random_forest",
    "or_cnn",
]

CHOSEN_MODEL = "random_forest"

NEURAL_NETWORK_MODELS = {
    "coral",
    "corn",
    "clm",
    "nn_rank",
    "condor",
    "or_cnn",
}
