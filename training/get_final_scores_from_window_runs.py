from training.constants import ALL_MODELS
from training.train_and_evaluate_models import calculate_final_scores_from_files


if __name__ == "__main__":
    models = ALL_MODELS

    calculate_final_scores_from_files(models, "expanded")
