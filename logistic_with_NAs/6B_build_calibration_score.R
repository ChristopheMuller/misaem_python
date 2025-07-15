
library(dplyr)
library(stringr)
library(reliabilitydiag)

library(reticulate)
np <- import("numpy")

####
# Configuration
####

exp <- "SimMCAR"
df_set_up <- read.csv(file.path("data", exp, "set_up.csv"))
simulation_file <- file.path("data", exp, "simulation.csv")
df_simulations <- read.csv(simulation_file)
matrix_scores <- read.csv(file.path("data", exp, "score_matrix.csv"))

list_of_methods <- c(
"MICE.1.IMP","MICE.1.Y.IMP","MICE.1.M.IMP","MICE.1.Y.M.IMP",
"MICE.1.IMP.M","MICE.1.Y.IMP.M","MICE.1.M.IMP.M","MICE.1.Y.M.IMP.M",
"MICE.10.IMP","MICE.10.Y.IMP","MICE.10.M.IMP","MICE.10.Y.M.IMP",
"MICE.10.IMP.M","MICE.10.Y.IMP.M","MICE.10.M.IMP.M","MICE.10.Y.M.IMP.M",
"SAEM",
"Mean.IMP","Mean.IMP.M","05.IMP","05.IMP.M",
"PbP",
#### "CC",
"MICE.RF.10.IMP","MICE.RF.10.Y.IMP","MICE.RF.10.M.IMP","MICE.RF.10.Y.M.IMP",
"MICE.RF.10.IMP.M","MICE.RF.10.Y.IMP.M","MICE.RF.10.M.IMP.M","MICE.RF.10.Y.M.IMP.M",
"MICE.100.IMP","MICE.100.Y.IMP","MICE.100.M.IMP","MICE.100.Y.M.IMP",
"MICE.100.IMP.M","MICE.100.Y.IMP.M","MICE.100.M.IMP.M","MICE.100.Y.M.IMP.M",
)


####
# Loop over all simulations
####

for (i in (1:dim(df_simulations)[1])){
  
  cat("Simulation ", i, " of ", dim(df_simulations)[1], "\n")

  ntrain <- df_simulations[i, "n_train"]
  ntrain <- as.character(as.integer(ntrain))
  method <- df_simulations[i, "method"]
  setup <- df_simulations[i, "set_up"]
  
  if (method %in% list_of_methods){

    test_y_file <- file.path("data", exp, "test_data", paste0(setup, ".npz"))
    test <- np$load(test_y_file)
    y_test <- test$f[["y"]]


    preds_y_file <- file.path("data", exp, "pred_data", paste0(setup, "_", method, "_", ntrain, ".npz"))
    pred <- np$load(preds_y_file)
    pred_y <- pred$f[["y_probs_pred"]]
    if (length(dim(pred_y))>1){
      pred_y <- pred_y[,1]
    }

    bayes_y_file <- file.path("data", exp, "bayes_data", paste0(setup, ".npz"))
    bayes <- np$load(bayes_y_file)
    bayes_y <- bayes$f[["y_probs_bayes"]]

    diag <- reliabilitydiag(preds = pred_y, bayes=bayes_y, y = y_test)
    cali <- summary(diag)$miscalibration
    
    pred_score <- cali[1]
    pred_bayes <- cali[2]

    # add score to matrix
    new_row <- c(exp, setup, method, ntrain, FALSE, "calibration", pred_score, "all")
    new_row_bayes <- c(exp, setup, method, ntrain, TRUE, "calibration", pred_score-pred_bayes, "all")
    matrix_scores <- rbind(matrix_scores, new_row)
    matrix_scores <- rbind(matrix_scores, new_row_bayes)

  }
    
}


# save as csv
write.csv(matrix_scores, file.path("data", exp, "score_matrix.csv"), row.names = FALSE)

