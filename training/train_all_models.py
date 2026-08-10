import argparse
import warnings

from training.train_and_evaluate_models import train_and_evaluate_models

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd  # noqa: E402

from dataset.creating_dataset import min_max_scale_data  # noqa: E402
from dataset.splitting_dataset import split_dataframe  # noqa: E402
from training.constants import (  # noqa: E402
    ALL_MODELS,
    CHRONOLOGICAL_SPLIT,
    PATH_TO_DATASET_PATTERN,
    RANDOM_PREFIX,
    RESULT_PATTERN,
    SET_NAME,
)  # noqa: E402
# from training.train_and_evaluate_models import train_and_evaluate_models  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate models")

    parser.add_argument(
        "--sets",
        nargs="+",
        help="List of dataset sets (e.g. basic expanded)",
        default={},
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random split",
    )

    parser.add_argument(
        "--sets_comparision",
        action="store_true",
        help="Use comparison models set",
        default=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_names = set(args.sets) or {SET_NAME}
    models_to_use = ALL_MODELS

    chronological_split = False if args.random else CHRONOLOGICAL_SPLIT

    for set_name in set_names:
        print(set_name)
        path_to_dataset = PATH_TO_DATASET_PATTERN.format(set_name=set_name)
        bestiaries = pd.read_csv(path_to_dataset, index_col=0)
        bestiaries = min_max_scale_data(bestiaries)

        X_train, X_test, y_train, y_test = split_dataframe(
            bestiaries, chronological_split=chronological_split
        )
        train_result_file = RESULT_PATTERN.format(
            prefix="" if chronological_split else RANDOM_PREFIX,
            set_name=set_name,
            result_type="train",
        )
        test_result_file = RESULT_PATTERN.format(
            prefix="" if chronological_split else RANDOM_PREFIX,
            set_name=set_name,
            result_type="test",
        )

        results_test, results_train = train_and_evaluate_models(
            models_to_use,
            X_train,
            y_train,
            X_test,
            y_test,
            thresholds=[
                [0.05 * i for i in range(1, 20)],
                [0.05 * i for i in range(5, 16)],
            ],
            save_files=(train_result_file, test_result_file),
            chronological=chronological_split,
            set_name=set_name,
        )
