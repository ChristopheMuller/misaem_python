#################################################
# Capture the arguments
#################################################

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  stop("Usage: Rscript run_simulation_from_config.R <path_to_config_file.txt>")
}

config_file_path <- args[1]

# Check if config file exists
if (!file.exists(config_file_path)) {
  stop(paste0("Configuration file not found: ", config_file_path))
}

# Read and parse the configuration file
config_lines <- readLines(config_file_path)
config_params <- list()
for (line in config_lines) {
  if (grepl("=", line)) {
    parts <- strsplit(line, "=")[[1]]
    key <- trimws(parts[1])
    value <- trimws(parts[2])
    config_params[[key]] <- value
  }
}

exp <- config_params$sim
if (is.null(exp)) stop("Config file must contain 'sim' parameter.")

method_keys_str <- config_params$methods
if (is.null(method_keys_str)) stop("Config file must contain 'methods' parameter.")
method_keys <- unlist(strsplit(method_keys_str, ","))

training_sizes_str <- config_params$training
if (is.null(training_sizes_str)) stop("Config file must contain 'training' parameter.")
training_sizes <- as.numeric(unlist(strsplit(training_sizes_str, ",")))


#################################################
# Load the packages
#################################################


library(tidyr, quietly=TRUE)
library(dplyr, quietly=TRUE)
library(furrr, quietly=TRUE)
library(future, quietly=TRUE)
library(reticulate, quietly=TRUE)
library(stringr, quietly=TRUE)

source("methods_in_R.R")

#################################################
# Start the simulation
#################################################

plan(multisession)
cat("\n")
cat("Number of parallel sessions configured:", future::nbrOfWorkers(), "\n")

# Experiment configuration
test_size <- 15000

create_method_object <- function(key) {
  switch(key,
    "MICE.1.IMP" = MICELogisticRegression$new(name="MICE.1.IMP", n_imputations=1, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
    "MICE.1.Y.IMP" = MICELogisticRegression$new(name="MICE.1.Y.IMP", n_imputations=1, add.y=TRUE, mask.after=FALSE, mask.before=FALSE),
    "MICE.1.M.IMP" = MICELogisticRegression$new(name="MICE.1.M.IMP", n_imputations=1, add.y=FALSE, mask.after=FALSE, mask.before=TRUE),
    "MICE.1.Y.M.IMP" = MICELogisticRegression$new(name="MICE.1.Y.M.IMP", n_imputations=1, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),

    "MICE.10.IMP" = MICELogisticRegression$new(name="MICE.10.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
    "MICE.10.Y.IMP" = MICELogisticRegression$new(name="MICE.10.Y.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=FALSE),
    "MICE.10.M.IMP" = MICELogisticRegression$new(name="MICE.10.M.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=TRUE),
    "MICE.10.Y.M.IMP" = MICELogisticRegression$new(name="MICE.10.Y.M.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),

    "MICE.100.IMP" = MICELogisticRegression$new(name="MICE.100.IMP", n_imputations=100, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
    "MICE.100.Y.IMP" = MICELogisticRegression$new(name="MICE.100.Y.IMP", n_imputations=100, add.y=TRUE, mask.after=FALSE, mask.before=FALSE),
    "MICE.100.M.IMP" = MICELogisticRegression$new(name="MICE.100.M.IMP", n_imputations=100, add.y=FALSE, mask.after=FALSE, mask.before=TRUE),
    "MICE.100.Y.M.IMP" = MICELogisticRegression$new(name="MICE.100.Y.M.IMP", n_imputations=100, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),
 
    "MICE.1.IMP.M" = MICELogisticRegression$new(name="MICE.1.IMP.M", n_imputations=1, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
    "MICE.1.Y.IMP.M" = MICELogisticRegression$new(name="MICE.1.Y.IMP.M", n_imputations=1, add.y=TRUE, mask.after=TRUE, mask.before=FALSE),
    "MICE.1.M.IMP.M" = MICELogisticRegression$new(name="MICE.1.M.IMP.M", n_imputations=1, add.y=FALSE, mask.after=TRUE, mask.before=TRUE),
    "MICE.1.Y.M.IMP.M" = MICELogisticRegression$new(name="MICE.1.Y.M.IMP.M", n_imputations=1, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),

    "MICE.10.IMP.M" = MICELogisticRegression$new(name="MICE.10.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
    "MICE.10.Y.IMP.M" = MICELogisticRegression$new(name="MICE.10.Y.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=FALSE),
    "MICE.10.M.IMP.M" = MICELogisticRegression$new(name="MICE.10.M.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=TRUE),
    "MICE.10.Y.M.IMP.M" = MICELogisticRegression$new(name="MICE.10.Y.M.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),

    "MICE.100.IMP.M" = MICELogisticRegression$new(name="MICE.100.IMP.M", n_imputations=100, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
    "MICE.100.Y.IMP.M" = MICELogisticRegression$new(name="MICE.100.Y.IMP.M", n_imputations=100, add.y=TRUE, mask.after=TRUE, mask.before=FALSE),
    "MICE.100.M.IMP.M" = MICELogisticRegression$new(name="MICE.100.M.IMP.M", n_imputations=100, add.y=FALSE, mask.after=TRUE, mask.before=TRUE),
    "MICE.100.Y.M.IMP.M" = MICELogisticRegression$new(name="MICE.100.Y.M.IMP.M", n_imputations=100, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),

    "MICE.RF.10.IMP" = MICERFLogisticRegression$new(name="MICE.RF.10.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
    "MICE.RF.10.Y.IMP" = MICERFLogisticRegression$new(name="MICE.RF.10.Y.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=FALSE),
    "MICE.RF.10.M.IMP" = MICERFLogisticRegression$new(name="MICE.RF.10.M.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=TRUE),
    "MICE.RF.10.Y.M.IMP" = MICERFLogisticRegression$new(name="MICE.RF.10.Y.M.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),

    "MICE.RF.10.IMP.M" = MICERFLogisticRegression$new(name="MICE.RF.10.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
    "MICE.RF.10.Y.IMP.M" = MICERFLogisticRegression$new(name="MICE.RF.10.Y.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=FALSE),
    "MICE.RF.10.M.IMP.M" = MICERFLogisticRegression$new(name="MICE.RF.10.M.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=TRUE),
    "MICE.RF.10.Y.M.IMP.M" = MICERFLogisticRegression$new(name="MICE.RF.10.Y.M.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),

    "SAEM" = SAEMLogisticRegression$new(name="SAEM"),

    "Mean.IMP" = MeanImputationLogisticRegression$new(name="Mean.IMP", mask=FALSE),
    "Mean.IMP.M" = MeanImputationLogisticRegression$new(name="Mean.IMP.M", mask=TRUE),

    "05.IMP" = ConstantImputationLogisticRegression$new(name="05.IMP", fill_value=0.5, mask=FALSE),
    "05.IMP.M" = ConstantImputationLogisticRegression$new(name="05.IMP.M", fill_value=0.5, mask=TRUE),

    "PbP" = RegLogPatByPat$new(name="PbP"),
    "PbP.MinObs" = RegLogPatByPatMinObservation$new(name="PbP.MinObs"),
    
    "CC" = CompleteCase$new(name="CC"),

    stop("Unknown method key: ", key)
  )
}

# Create the list of method objects based on keys from the config
methods_list <- lapply(method_keys, create_method_object)

cat("\n")
cat("Methods:\n")
for (method in methods_list) cat(method$name, "\n")
cat("Training sizes:\n")
print(training_sizes)

# Load setup
df_set_up <- read.csv(file.path("data", exp, "set_up.csv"))


# Grid of tasks
task_grid <- expand.grid(
  set_up = df_set_up$set_up,
  n_train = training_sizes,
  method_idx = seq_along(methods_list),
  stringsAsFactors = FALSE
)

# Correctly defined run_task function
run_task <- function(set_up, n_train, method_idx) {

  cat("\n")
  cat("Running task for set_up:", set_up, "n_train:", n_train, "method_idx:", method_idx, "\n")
  
  source("methods_in_R.R")
  np <- import("numpy")
  
  time.start <- Sys.time()
  
  method <- methods_list[[method_idx]]
  
  data <- np$load(file.path("data", exp, "original_data", paste0(set_up, ".npz")))
  X_obs <- data$f[["X_obs"]]
  M <- data$f[["M"]]
  y <- data$f[["y"]]
  
  data_test <- np$load(file.path("data", exp, "test_data", paste0(set_up, ".npz")))
  X_test <- data_test$f[["X_obs"]]
  M_test <- data_test$f[["M"]]
  y_probs_test <- data_test$f[["y_probs"]]
  y_test <- data_test$f[["y"]]
  
  X_train <- X_obs[1:n_train, ]
  M_train <- M[1:n_train, ]
  y_train <- y[1:n_train]
  
  fit_success <- tryCatch({
    tic <- Sys.time()
    method$fit(X_train, M_train, y_train, X_test, M_test)
    toc <- Sys.time()
    running_time <- as.numeric(difftime(toc, tic, units = "secs"))
    TRUE
  }, error = function(e) {
    message(sprintf("Error in fit for method %s: %s", method$name, e$message))
    return(NULL)
  })
  
  y_probs_pred <- NA
  running_time_pred <- NA
  save_name <- NA
  estimated_beta <- NA
  
  if (isTRUE(fit_success) && method$can_predict) {
    pred_success <- tryCatch({
      tic.pred <- Sys.time()
      y_probs_pred <- method$predict_probs(X_test, M_test)
      toc.pred <- Sys.time()
      running_time_pred <- as.numeric(difftime(toc.pred, tic.pred, units = "secs"))
      
      save_name <- paste0(set_up, "_", method$name, "_", n_train)
      np$savez(file.path("data", exp, "pred_data", paste0(save_name, ".npz")), y_probs_pred = y_probs_pred)
      
      estimated_beta <- if (method$return_beta) toString(method$return_params()) else NA
      
      TRUE 
    }, error = function(e) {
      message(sprintf("Error in prediction for method %s: %s", method$name, e$message))
      return(FALSE) # Indicate prediction failed
    })
    
    if (!isTRUE(pred_success)) { # If prediction failed, set relevant fields to NA
        y_probs_pred <- NA
        running_time_pred <- NA
        save_name <- NA
        estimated_beta <- NA # If prediction failed, params might not be trustworthy
    }
  } else if (isTRUE(fit_success) && !method$can_predict) {
    estimated_beta <- if (method$return_beta) toString(method$return_params()) else NA
  } else {
    return(NULL)
  }

  data.frame(
    set_up = set_up,
    method = method$name,
    n_train = n_train,
    estimated_beta = estimated_beta,
    file_name = save_name,
    running_time_train = running_time,
    running_time_pred = running_time_pred,
    running_datetime = as.character(time.start),
    stringsAsFactors = FALSE
  )
}

# Run tasks in parallel
results_df <- future_pmap_dfr(task_grid, run_task, .progress = FALSE,   .options = furrr_options(seed = TRUE))

# Load or initialize results
simulation_file <- file.path("data", exp, "simulation.csv")
if (file.exists(simulation_file)) {
  simulations_df <- read.csv(simulation_file)
} else {
  simulations_df <- data.frame(
    set_up = character(),
    method = character(),
    n_train = numeric(),
    estimated_beta = character(),
    file_name = character(),
    running_time_train = numeric(),
    running_time_pred = numeric(),
    running_datetime = character(),
    stringsAsFactors = FALSE
  )
}

simulations_df <- bind_rows(simulations_df, results_df)

write.csv(simulations_df, simulation_file, row.names = FALSE)