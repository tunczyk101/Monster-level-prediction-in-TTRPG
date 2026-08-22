RUNNER = uv run
FILES = .

setup:
	$(RUNNER) pre-commit install

lint:
	$(RUNNER) ruff check $(FILES)
	$(RUNNER) ruff format $(FILES) --diff

save-bestiaries:
	$(RUNNER) python dataset/save_preprocessed_bestiaries.py

plot-correlation-matrix:
	$(RUNNER) python results_analysis_and_plots/correlation_matrix.py

plot-error-vs-train-distance:
	$(RUNNER) python results_analysis_and_plots/error_vs_train_distance.py

plot-errors:
	$(RUNNER) python results_analysis_and_plots/errors_histogram.py

plot-feature-importances:
	$(RUNNER) python results_analysis_and_plots/feature_importances.py

features-analysis:
	$(RUNNER) python results_analysis_and_plots/features_analysis.py

plot-histograms:
	$(RUNNER) python results_analysis_and_plots/features_histograms.py

plot-levels-map:
	$(RUNNER) python results_analysis_and_plots/levels_map.py

plot-confusion-matrix:
	$(RUNNER) python results_analysis_and_plots/plot_confusion_matrix.py

plot-levels:
	$(RUNNER) python results_analysis_and_plots/plot_levels.py

plot-sets:
	$(RUNNER) python results_analysis_and_plots/plot_sets_comparison.py

plot-splits:
	$(RUNNER) python results_analysis_and_plots/plot_splits.py

plot-all: plot-correlation-matrix \
          plot-error-vs-train-distance \
          plot-errors \
          features-analysis \
          plot-histograms \
          plot-levels-map \
          plot-confusion-matrix \
          plot-levels \
          plot-sets \
          plot-splits \
		  plot-feature-importances

train:
	$(RUNNER) python training/train_all_models.py

train-comparison-sets:
	$(RUNNER) python training/train_all_models.py --sets basic full

train-random:
	$(RUNNER) python training/train_all_models.py --random --sets_comparison

train-window:
	$(RUNNER) python training/train_all_models_expanding_window.py

recalculate-final-results:
	$(RUNNER) python training/calculate_metrics_from_files.py

recalculate-final-window-results:
	$(RUNNER) python training/get_final_scores_from_window_runs.py

