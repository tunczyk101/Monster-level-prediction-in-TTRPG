import os

import pandas as pd

from dataset.constants import (
    FEATURES,
    ORDERED_CHARACTERISTICS_BASIC,
    ORDERED_CHARACTERISTICS_EXPANDED,
    ORDERED_CHARACTERISTICS_FULL,
    PROCESSED_BESTIARIES_FOLDER,
    SAVE_DIR,
)
from dataset.creating_dataset import load_and_preprocess_data


BASIC_COLUMNS = ["book", "level", "name"]


if __name__ == "__main__":
    dataset_books = pd.read_csv("dataset/dataset_books.csv")["book"].to_list()

    all_features_df = load_and_preprocess_data(
        folder=SAVE_DIR, characteristics=FEATURES, chosen_books=dataset_books
    )

    df = all_features_df[ORDERED_CHARACTERISTICS_FULL + BASIC_COLUMNS]
    df.to_csv(os.path.join(PROCESSED_BESTIARIES_FOLDER, "bestiaries_full.csv"))

    df = all_features_df[ORDERED_CHARACTERISTICS_EXPANDED + BASIC_COLUMNS]
    df.to_csv(os.path.join(PROCESSED_BESTIARIES_FOLDER, "bestiaries_expanded.csv"))

    df = all_features_df[ORDERED_CHARACTERISTICS_BASIC + BASIC_COLUMNS]
    df.to_csv(os.path.join(PROCESSED_BESTIARIES_FOLDER, "bestiaries_basic.csv"))
