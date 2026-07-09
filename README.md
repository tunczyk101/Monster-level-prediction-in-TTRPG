# Application of machine learning to monster level prediction in tabletop RPG game design

Code for *"Application of machine learning to monster level prediction in tabletop RPG game design"*.


## Set up environment

To install all dependencies, use the provided `uv.lock` file by running `uv sync`. The project requires a Python 3.11 virtual environment.

## Data source

The original data was cloned from the [FoundryVTT pf2e](https://github.com/foundryvtt/pf2e) repository and filtered using `dataset/save_monsters_to_books_files.py`.
The script creates the `dataset/pathfinder_2e_remaster_data_2026` folder containing JSON Lines (`.jsonl`) files with all monsters from PF2e NPC FoundryVTT packs. After the initial filtering step, all processed NPC creatures are transformed into three datasets with different numbers of features: Basic, Expanded, Full. The datasets are generated using `make save-bestiaries` command. The final datasets are stored in the `preprocessed_bestiaries` folder.

## Models

Implementations of models that are not directly imported from external packages are stored in the `training/models` folder. The implementation of the SAOC model is adapted from [Ordinal_Classifier](https://github.com/mosh98/Ordinal_Classifier).

## Reproducing results
The following commands reproduce the complete experimental pipeline, including data loading, dataset splitting, model training, prediction generation, and evaluation:

* **Chronological split** - `make train`
* **Random split** (evaluation of selected models) - `make train -random`
* **Expanding window** - `make train-window`
* **Chronological split: Basic & Expanded** (evaluation of selected models) - `make train-comparision-sets`

Final evaluation results are stored as CSV files inside:`training/results`. Predictions from individual models are stored inside `training/results\models_predictions`. Expanding window results are stored inside `training/results/ts`. Detailed results for individual models are stored in `training/results/ts/{model_name}` folders.

The repository contains implementations and evaluations of: LR Ridge, Kernel SVM, KNN, RF, LightGBM, ORD [probit], ORD [logit], ORF, Logistic AT, Logistic IT, SAOC, GPOR, CORAL, CORN, CLM, NNRank, CONDOR, OR-CNN, 

Models used for Random, Basic, and Expanded dataset experiments: KNN, Ridge regression, RF, LightGBM, SAOC, ORF, OR-CNN.

## Additional ploting scripts

The `results_analysis_and_plots` folder contains scripts for plotting results and analyzing dataset characteristics.

Some scripts require selecting a specific model for detailed analysis. The default analyzed model is Random Forest (RF).

Available commands::

* Correlation matrix - `make plot-correlation-matrix`
* Distance to training samples vs. absolute error (RF) - `make plot-error-vs-train-distance`
* Error histogram (RF) - `make plot-errors`
* Feature importance analysis (RF) - `make plot-feature-importances`
* Feature statistics (variance, Pearson correlation with target, mutual information) - `make features-analysis`
* Feature distribution histograms - `make plot-histograms`
* PCA and t-SNE visualization - `make plot-levels-map`
* Confusion matrix - `make plot-confusion-matrix`
* Target level distribution - `make plot-levels`
* Comparison of Basic, Expanded, and Full datasets - `make plot-sets`
* Comparison of evaluation strategies (chronological split, random split, expanding window) - `make plot-splits`




