## Codebase from the paper

### Structure of the code

This repository combines R and python. The structure is as follows:

1. Files starting by `1_xxx` are used to generate the data (X,Y,M) and computes the bayes probabilities of Y. All the relevant information is stored into an separate folder for each experiement (e.g. `data/SimMCAR/*`).
2. The files `2_train_methods.py` & `3_train_methods_in_R.R` takes as input a list of methods, a training size and a simulation and runs these methods in python & R, respectively.
3. `4_adjust_results.py` process the results files after using once or several times the `train_methods` files.
4. `5_param_estimation_scores.py` computes the inference and running time scores of each method.
5. `6A_build_score_matrix.py`, `6B_build_calibration_score.R` & `6C_score_per_pattern.py` compile inference and running time, computes the prediction scores, computes the calibration scores and computes a specific score conditioning on patterns in the test set.

### Reproducing the results

In order to reproduce our results, you should first delete/move our results folders (`data/SimXXX`). Then, for the MCAR simulation, do:

1. Create the python environment: `python -m pip install -r requirements.txt`
2. Create the R environment:
   1. Open `logistic_with_NAs.Rproj`
   3. Open R: `R`
   4. Install renv: `install.packages("renv")`
   5. Restore environment: `renv::restore()`
3. Create the simulation data: `python 1_SimMCAR_generate_data.py`
4. Train the methods in R: `Rscript 3_train_methods_in_R.R configs/config_MCAR.txt`
5. Process the results: `python 4_adjust_results.py && python 5_param_estimation_scores.py`
6. Create the score matrix: `python 6A_build_score_matrix.py && Rscript 6B_build_calibration_score.R`
7. Reproduce the figures and table in the `/tables_and_figures/` folder.