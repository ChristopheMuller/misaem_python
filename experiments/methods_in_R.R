library(mice, quietly=TRUE)
library(dplyr, quietly=TRUE)
library(stringr, quietly=TRUE)
library(misaem, quietly=TRUE)

# Base class for imputation methods
ImputationMethod <- R6::R6Class("ImputationMethod",
  public = list(
      name = NULL,
      can_predict = TRUE,
      return_beta = TRUE,
      model = NULL,
    imputation_model = NULL,
      
      initialize = function(name) {
        self$name <- name
      },
      
      fit = function(X, M, y) {
        stop("Method not implemented")
      },
      
      predict_probs = function(X_new, M_new) {
        stop("Method not implemented")
      },
      
      return_params = function() {
        if (!self$return_beta) return(NULL)
        return(coef(self$model))
      }
    )
)

MICELogisticRegression <- R6::R6Class("MICELogisticRegression",
  inherit = ImputationMethod,
  public = list(
    n_imputations = 5,
    maxit = 5,
    mask.after = FALSE,
    add.y = FALSE,
    mask.before = FALSE,
    
    initialize = function(name, n_imputations = 5, maxit = 5, mask.before = FALSE, add.y = FALSE, mask.after = FALSE) {
      super$initialize(name)
      self$n_imputations <- n_imputations
      self$maxit <- maxit
      self$add.y <- add.y
      self$mask.before <- mask.before
      self$mask.after <- mask.after
    },
    
    fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
      
      # change the col names of M-train: M1, .., Md
      colnames(M_train) <- paste0("M", seq_len(ncol(M_train)))
      M_train <- as.data.frame(M_train)

      # Combine training data for imputation
      data_train <- as.data.frame(X_train)
      
      # Add y to data if add.y is TRUE
      if (self$add.y) {
        data_train$y <- y_train
      }
      
      # Add mask before imputation if mask.before is TRUE
      if (self$mask.before) {
        data_train <- cbind(data_train, as.data.frame(M_train))
      }
      
      # Create ignore vector for MICE
      ignore_vec <- rep(FALSE, nrow(data_train))
      
      # If test set is provided
      if (!is.null(X_test)) {
        colnames(M_test) <- paste0("M", seq_len(ncol(M_test)))
        data_test <- as.data.frame(X_test)
        
        # Add y (as NA) to test data if add.y is TRUE
        if (self$add.y) {
          data_test$y <- NA
        }

        # Add mask before imputation if mask.before is TRUE
        if (self$mask.before) {
          data_test <- cbind(data_test, as.data.frame(M_test))
        }
        
        # Combine train and test data
        data_full <- rbind(data_train, data_test)
        
        # Update ignore vector
        ignore_vec <- c(ignore_vec, rep(TRUE, nrow(data_test)))
        
        # Run MICE on full dataset
        self$imputation_model <- mice(data_full, m = self$n_imputations, 
                                      maxit = self$maxit, 
                                      printFlag = FALSE, 
                                      ignore = ignore_vec)
      } else {
        # Run MICE only on training data
        self$imputation_model <- mice(data_train, m = self$n_imputations, 
                                      maxit = self$maxit, 
                                      printFlag = FALSE)
      }
      
      # print(self$imputation_model$loggedEvents)
      
      # Fit logistic regression on each imputed training dataset
      models <- list()
      for(i in 1:self$n_imputations) {
        # Complete only the training data
        imp_train_data <- complete(self$imputation_model, i)[!ignore_vec, ]

        # Remove y if it was added during imputation
        if (self$add.y) {
          imp_train_data <- imp_train_data[, !names(imp_train_data) %in% "y"]
        }

        if (self$mask.before) {
          imp_train_data <- imp_train_data[, !(names(imp_train_data) %in% names(M_train))]
        }

        # Add mask before logistic regression if mask is TRUE
        if (self$mask.after) {
          imp_train_data <- cbind(imp_train_data, as.data.frame(M_train))
        }

        # Fit logistic regression
        formula <- as.formula(paste("y ~", paste(names(imp_train_data)[names(imp_train_data) != "y"], 
                                                 collapse = " + ")))
        models[[i]] <- glm(formula, family = binomial(), data = cbind(imp_train_data, y = y_train))
      }
      
      # Store models
      self$model <- models
      TRUE
    },
    
    predict_probs = function(X_new, M_new) {

      # change the col names of M_new: M1, .., Md
      colnames(M_new) <- paste0("M", seq_len(ncol(M_new)))

      # Predict for each imputed model
      pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
      
      for(i in 1:self$n_imputations) {
        # Complete the test data using the same imputation model
        imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
        
        # Remove y if it was added during imputation
        if (self$add.y) {
          imp_test <- imp_test[, !names(imp_test) %in% "y"]
        }

        if (self$mask.before) {
          imp_test <- imp_test[, !names(imp_test) %in% names(M_new)]
        }
        
        # Add mask before prediction if mask is TRUE
        if (self$mask.after) {
          imp_test <- cbind(imp_test, as.data.frame(M_new))
        }
        
        # Predict probabilities
        pred_probs[,i] <- predict(self$model[[i]], newdata = imp_test, type = "response")
      }
      
      # Average predictions across imputations
      return(rowMeans(pred_probs))
    },
                                        
    return_params = function() {
      if (!self$return_beta) return(NULL)
      
      # Pool coefficients using Rubin's rules
      coef_list <- lapply(self$model, coef)
      pooled_coef <- Reduce('+', coef_list) / length(coef_list)
      
      # Separate intercept and coefficients
      intercept <- pooled_coef[1]  # First coefficient is intercept in R
      coefficients <- pooled_coef[-1]  # All other coefficients
      
      # Remove names from the vectors
      names(intercept) <- NULL
      names(coefficients) <- NULL
      
      # Create the exact string format to match Python output
      coef_str <- paste(coefficients, collapse = ", ")
      int_str <- as.character(intercept)
      
      return(sprintf("[[%s], [%s]]", coef_str, int_str))
    }
  )                     
)


MICERFLogisticRegression <- R6::R6Class("MICERFLogisticRegression",
                                          inherit = ImputationMethod,
                                          public = list(
                                            n_imputations = 5,
                                            maxit = 5,
                                            mask.after = FALSE,
                                            add.y = FALSE,
                                            mask.before = FALSE,
                                            
                                            initialize = function(name, n_imputations = 5, maxit = 5, mask.before = FALSE, add.y = FALSE, mask.after = FALSE) {
                                              super$initialize(name)
                                              self$n_imputations <- n_imputations
                                              self$maxit <- maxit
                                              self$add.y <- add.y
                                              self$mask.before <- mask.before
                                              self$mask.after <- mask.after
                                            },
                                            
                                            fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
                                              
                                              # change the col names of M-train: M1, .., Md
                                              colnames(M_train) <- paste0("M", seq_len(ncol(M_train)))
                                              M_train <- as.data.frame(M_train)

                                              # Combine training data for imputation
                                              data_train <- as.data.frame(X_train)
                                              
                                              # Add y to data if add.y is TRUE
                                              if (self$add.y) {
                                                data_train$y <- y_train
                                              }
                                              
                                              # Add mask before imputation if mask.before is TRUE
                                              if (self$mask.before) {
                                                data_train <- cbind(data_train, as.data.frame(M_train))
                                              }
                                              
                                              # Create ignore vector for MICE
                                              ignore_vec <- rep(FALSE, nrow(data_train))
                                              
                                              # If test set is provided
                                              if (!is.null(X_test)) {
                                                colnames(M_test) <- paste0("M", seq_len(ncol(M_test)))
                                                data_test <- as.data.frame(X_test)
                                                
                                                # Add y (as NA) to test data if add.y is TRUE
                                                if (self$add.y) {
                                                  data_test$y <- NA
                                                }

                                                # Add mask before imputation if mask.before is TRUE
                                                if (self$mask.before) {
                                                  data_test <- cbind(data_test, as.data.frame(M_test))
                                                }
                                                
                                                # Combine train and test data
                                                data_full <- rbind(data_train, data_test)
                                                
                                                # Update ignore vector
                                                ignore_vec <- c(ignore_vec, rep(TRUE, nrow(data_test)))
                                                
                                                # Run MICE on full dataset
                                                self$imputation_model <- mice(data_full, m = self$n_imputations, 
                                                                              maxit = self$maxit, 
                                                                              printFlag = FALSE, 
                                                                              ignore = ignore_vec, method="rf")
                                              } else {
                                                # Run MICE only on training data
                                                self$imputation_model <- mice(data_train, m = self$n_imputations, 
                                                                              maxit = self$maxit, 
                                                                              printFlag = FALSE, method="rf")
                                              }
                                              
                                              # Fit logistic regression on each imputed training dataset
                                              models <- list()
                                              for(i in 1:self$n_imputations) {
                                                # Complete only the training data
                                                imp_train_data <- complete(self$imputation_model, i)[!ignore_vec, ]

                                                # Remove y if it was added during imputation
                                                if (self$add.y) {
                                                  imp_train_data <- imp_train_data[, !names(imp_train_data) %in% "y"]
                                                }

                                                if (self$mask.before) {
                                                  imp_train_data <- imp_train_data[, !(names(imp_train_data) %in% names(M_train))]
                                                }

                                                # Add mask before logistic regression if mask is TRUE
                                                if (self$mask.after) {
                                                  imp_train_data <- cbind(imp_train_data, as.data.frame(M_train))
                                                }

                                                # Fit logistic regression
                                                formula <- as.formula(paste("y ~", paste(names(imp_train_data)[names(imp_train_data) != "y"], 
                                                                                        collapse = " + ")))
                                                models[[i]] <- glm(formula, family = binomial(), data = cbind(imp_train_data, y = y_train))
                                              }
                                              
                                              # Store models
                                              self$model <- models
                                              TRUE
                                            },
                                            
                                            predict_probs = function(X_new, M_new) {

                                              # change the col names of M_new: M1, .., Md
                                              colnames(M_new) <- paste0("M", seq_len(ncol(M_new)))

                                              # Predict for each imputed model
                                              pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
                                              
                                              for(i in 1:self$n_imputations) {
                                                # Complete the test data using the same imputation model
                                                imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
                                                
                                                # Remove y if it was added during imputation
                                                if (self$add.y) {
                                                  imp_test <- imp_test[, !names(imp_test) %in% "y"]
                                                }

                                                if (self$mask.before) {
                                                  imp_test <- imp_test[, !names(imp_test) %in% names(M_new)]
                                                }
                                                
                                                # Add mask before prediction if mask is TRUE
                                                if (self$mask.after) {
                                                  imp_test <- cbind(imp_test, as.data.frame(M_new))
                                                }
                                                
                                                # Predict probabilities
                                                pred_probs[,i] <- predict(self$model[[i]], newdata = imp_test, type = "response")
                                              }
                                              
                                              # Average predictions across imputations
                                              return(rowMeans(pred_probs))
                                            },
                                                                                
                                            return_params = function() {
                                              if (!self$return_beta) return(NULL)
                                              
                                              # Pool coefficients using Rubin's rules
                                              coef_list <- lapply(self$model, coef)
                                              pooled_coef <- Reduce('+', coef_list) / length(coef_list)
                                              
                                              # Separate intercept and coefficients
                                              intercept <- pooled_coef[1]  # First coefficient is intercept in R
                                              coefficients <- pooled_coef[-1]  # All other coefficients
                                              
                                              # Remove names from the vectors
                                              names(intercept) <- NULL
                                              names(coefficients) <- NULL
                                              
                                              # Create the exact string format to match Python output
                                              coef_str <- paste(coefficients, collapse = ", ")
                                              int_str <- as.character(intercept)
                                              
                                              return(sprintf("[[%s], [%s]]", coef_str, int_str))
                                            }
                                          )                     
)


SAEMLogisticRegression <- R6::R6Class("SAEMLogisticRegression",
                                      inherit = ImputationMethod,
                                      public = list(
                                        initialize = function(name) {
                                          super$initialize(name)
                                        },
                                        
                                        fit = function(X, M, y, X_test = NULL, M_test = NULL) {
                                          # Convert data to required format
                                          data <- as.data.frame(X)
                                          colnames(data) <- paste0("X", 1:ncol(X))
                                          data$y <- y
                                          
                                          # Fit SAEM model
                                          formula <- as.formula(paste("y ~", paste(colnames(data)[1:(ncol(data)-1)], collapse = " + ")))
                                          self$model <- miss.glm(formula, data = data, print_iter = FALSE)
                                          
                                          TRUE
                                        },
                                        
                                        predict_probs = function(X_new, M_new) {
                                          # Prepare test data
                                          X_test <- as.data.frame(X_new)
                                          colnames(X_test) <- paste0("X", 1:ncol(X_new))
                                          
                                          # Get predictions
                                          pred_probs <- predict(self$model, newdata = X_test, type = "response", mcmc_map=500)

                                          return(pred_probs)
                                        },
                                        
                                        return_params = function() {
                                          if (!self$return_beta) return(NULL)
                                          
                                          # Extract coefficients
                                          coef_summary <- summary(self$model)$coef
                                          coefficients <- coef_summary[-1, "Estimate"]  # All except intercept
                                          intercept <- coef_summary[1, "Estimate"]     # Intercept only
                                          
                                          # Remove names from the vectors
                                          names(coefficients) <- NULL
                                          names(intercept) <- NULL
                                          
                                          # Create the exact string format to match Python output
                                          coef_str <- paste(coefficients, collapse = ", ")
                                          int_str <- as.character(intercept)
                                          
                                          return(sprintf("[[%s], [%s]]", coef_str, int_str))
                                        }
                                      )
)

MeanImputationLogisticRegression <- R6::R6Class("MeanImputationLogisticRegression",
  inherit = ImputationMethod,
  public = list(
    column_means = NULL,
    mask = FALSE,

    initialize = function(name = "Mean", mask = FALSE) {
      super$initialize(name)
      self$mask <- mask
    },

    fit = function(X, M, y, X_test = NULL, M_test = NULL) {
      data_train <- as.data.frame(X)

      self$column_means <- list()
      for (col_name in names(data_train)) {
        self$column_means[[col_name]] <- mean(data_train[[col_name]], na.rm = TRUE)
        if (any(is.na(data_train[[col_name]]))) {
          data_train[[col_name]][is.na(data_train[[col_name]])] <- self$column_means[[col_name]]
        }
      }

      if (self$mask) {
        colnames(M) <- paste0("M", seq_len(ncol(M)))
        data_train <- cbind(data_train, as.data.frame(M))
      }

      formula <- as.formula(paste("y ~", paste(names(data_train), collapse = " + ")))
      self$model <- glm(formula, family = binomial(), data = cbind(data_train, y = y))
      TRUE
    },

    predict_probs = function(X_new, M_new) {
      data_new <- as.data.frame(X_new)

      for (col_name in names(data_new)) {
        if (any(is.na(data_new[[col_name]]))) {
          if (!is.null(self$column_means[[col_name]])) {
            data_new[[col_name]][is.na(data_new[[col_name]])] <- self$column_means[[col_name]]
          } else {
            data_new[[col_name]][is.na(data_new[[col_name]])] <- mean(data_new[[col_name]], na.rm = TRUE)
            cat("Warning: No mean stored for column", col_name, "- using mean of new data.\n")
          }
        }
      }

      if (self$mask) {
        colnames(M_new) <- paste0("M", seq_len(ncol(M_new)))
        data_new <- cbind(data_new, as.data.frame(M_new))
      }

      return(predict(self$model, newdata = data_new, type = "response"))
    },

    return_params = function() {
      if (!self$return_beta) return(NULL)

      # Get coefficients from the fitted glm model
      # For glm, coef(self$model) directly gives the coefficients
      # The first element is the intercept, followed by other coefficients
      model_coef <- coef(self$model)

      # Separate intercept and coefficients
      intercept <- model_coef[1]  # First coefficient is intercept in R
      coefficients <- model_coef[-1] # All other coefficients

      # Remove names from the vectors
      names(intercept) <- NULL
      names(coefficients) <- NULL

      # Create the exact string format to match Python output
      coef_str <- paste(coefficients, collapse = ", ")
      int_str <- as.character(intercept)

      return(sprintf("[[%s], [%s]]", coef_str, int_str))
    }
  )
)

ConstantImputationLogisticRegression <- R6::R6Class("ConstantImputationLogisticRegression",
  inherit = ImputationMethod,
  public = list(
    fill_value = 0, 
    mask = FALSE,

    initialize = function(name = "0.IMP", fill_value = 0, mask = FALSE) {
      super$initialize(name)
      self$fill_value <- fill_value
      self$mask <- mask
    },

    fit = function(X, M, y, X_test = NULL, M_test = NULL) {
      data_train <- as.data.frame(X)

      # Impute missing values with the constant fill_value
      for (col_name in names(data_train)) {
        if (any(is.na(data_train[[col_name]]))) {
          data_train[[col_name]][is.na(data_train[[col_name]])] <- self$fill_value
        }
      }

      # Add mask M to covariates if 'mask' is TRUE
      if (self$mask) {
        colnames(M) <- paste0("M", seq_len(ncol(M)))
        data_train <- cbind(data_train, as.data.frame(M))
      }

      formula <- as.formula(paste("y ~", paste(names(data_train), collapse = " + ")))
      self$model <- glm(formula, family = binomial(), data = cbind(data_train, y = y))
      TRUE
    },

    predict_probs = function(X_new, M_new) {
      data_new <- as.data.frame(X_new)

      for (col_name in names(data_new)) {
        if (any(is.na(data_new[[col_name]]))) {
          data_new[[col_name]][is.na(data_new[[col_name]])] <- self$fill_value
        }
      }

      if (self$mask) {
        colnames(M_new) <- paste0("M", seq_len(ncol(M_new)))
        data_new <- cbind(data_new, as.data.frame(M_new))
      }

      return(predict(self$model, newdata = data_new, type = "response"))
    },

    return_params = function() {
      if (!self$return_beta) return(NULL)

      model_coef <- coef(self$model)
      intercept <- model_coef[1]
      coefficients <- model_coef[-1]

      names(intercept) <- NULL
      names(coefficients) <- NULL

      coef_str <- paste(coefficients, collapse = ", ")
      int_str <- as.character(intercept)

      return(sprintf("[[%s], [%s]]", coef_str, int_str))
    }
  )
)


RegLogPatByPat <- R6::R6Class("RegLogPatByPat",
  inherit = ImputationMethod,
  public = list(
    
    models_by_pattern = NULL,
    default_prob = NULL,

    initialize = function(name = "PbP") {
      super$initialize(name)
      self$can_predict = TRUE  
      self$return_beta = FALSE 
      self$models_by_pattern = list()
    },

    fit = function(X, M, y, X_test = NULL, M_test = NULL) {

      X_df <- as.data.frame(X)
      M_matrix <- as.matrix(M)
      y_vec <- as.numeric(y)

      self$default_prob <- mean(y_vec, na.rm = TRUE)

      all_pattern_keys <- apply(M_matrix, 1, paste, collapse = "_")
      unique_pattern_keys <- unique(all_pattern_keys)
      
      for (pattern_str in unique_pattern_keys) {

        current_pattern_mask_vec <- M_matrix[match(pattern_str, all_pattern_keys),]

        has_observed_vars <- any(current_pattern_mask_vec == 0)

        if (has_observed_vars) {

          S_indices <- which(apply(M_matrix, 1, function(row) all(row == current_pattern_mask_vec)))

          Xp <- X_df[S_indices, current_pattern_mask_vec == 0, drop = FALSE]
          yp <- y_vec[S_indices]

          if (nrow(Xp) > 0 && length(unique(yp)) == 2) {

            model_data <- Xp
            model_data$y_outcome <- yp 
            
            if (ncol(Xp) > 0) {
              formula_str <- paste("y_outcome ~", paste(names(Xp), collapse = " + "))
            } else {
              formula_str <- "y_outcome ~ 1"
              cat("Warning: No observed variables for pattern '", pattern_str, "'. Using intercept-only model.\n")
            }

            reg_model <- tryCatch({
              glm(as.formula(formula_str), family = binomial(), data = model_data)
            }, error = function(e) {
              message(paste("  Warning: GLM failed for pattern '", pattern_str, "'. Error: ", e$message, sep=""))
              return(NULL)
            })
            
            self$models_by_pattern[[pattern_str]] <- reg_model
          }
        }
      }
      TRUE
    },

    predict_probs = function(X_new, M_new) {
      X_new_df <- as.data.frame(X_new)
      M_new_matrix <- as.matrix(M_new)

      n_new_obs <- nrow(X_new_df)
      predictions <- numeric(n_new_obs)

      all_pattern_keys_new <- apply(M_new_matrix, 1, paste, collapse = "_")
      unique_pattern_keys_new <- unique(all_pattern_keys_new)
      
      for (pattern_str in unique_pattern_keys_new) {
        
        pattern_indices <- which(all_pattern_keys_new == pattern_str)
        
        current_m_mask_vec <- M_new_matrix[pattern_indices[1], ] # Mask is same for all rows with this pattern

        all_vars_missing <- all(current_m_mask_vec == 1)

        if (all_vars_missing || is.null(self$models_by_pattern[[pattern_str]])) {
          predictions[pattern_indices] <- self$default_prob
        } else {
          reg_model <- self$models_by_pattern[[pattern_str]]

          X_current_pattern <- X_new_df[pattern_indices, current_m_mask_vec == 0, drop = FALSE]
          pattern_predictions <- predict(reg_model, newdata = X_current_pattern, type = "response")
          
          predictions[pattern_indices] <- pattern_predictions
        }
      }
      return(predictions)
    },

    return_params = function() {
      if (!self$return_beta) return(NULL)
    }
  )
)


RegLogPatByPatMinObservation <- R6::R6Class("RegLogPatByPat",
  inherit = ImputationMethod,
  public = list(

    models_by_pattern = NULL,
    default_prob = NULL,

    initialize = function(name = "PbP") {
      super$initialize(name)
      self$can_predict = TRUE
      self$return_beta = FALSE
      self$models_by_pattern = list()
    },

    fit = function(X, M, y, X_test = NULL, M_test = NULL) {

      X_df <- as.data.frame(X)
      M_matrix <- as.matrix(M)
      y_vec <- as.numeric(y)

      self$default_prob <- mean(y_vec, na.rm = TRUE)

      all_pattern_keys <- apply(M_matrix, 1, paste, collapse = "_")
      unique_pattern_keys <- unique(all_pattern_keys)

      for (pattern_str in unique_pattern_keys) {

        current_pattern_mask_vec <- M_matrix[match(pattern_str, all_pattern_keys),]

        has_observed_vars <- any(current_pattern_mask_vec == 0)
        
        # Determine the number of observed variables for this pattern
        num_observed_vars <- sum(current_pattern_mask_vec == 0)

        if (has_observed_vars) {

          S_indices <- which(apply(M_matrix, 1, function(row) all(row == current_pattern_mask_vec)))

          Xp <- X_df[S_indices, current_pattern_mask_vec == 0, drop = FALSE]
          yp <- y_vec[S_indices]

          if (nrow(Xp) >= num_observed_vars && length(unique(yp)) == 2) {

            model_data <- Xp
            model_data$y_outcome <- yp

            if (ncol(Xp) > 0) {
              formula_str <- paste("y_outcome ~", paste(names(Xp), collapse = " + "))
            } else {
              formula_str <- "y_outcome ~ 1"
            }

            reg_model <- tryCatch({
              glm(as.formula(formula_str), family = binomial(), data = model_data)
            }, error = function(e) {
              message(paste("    Warning: GLM failed for pattern '", pattern_str, "'. Error: ", e$message, ". Model not trained for this pattern.", sep=""))
              return(NULL) # Return NULL if GLM fails
            })

            self$models_by_pattern[[pattern_str]] <- reg_model
          } else {
            message(paste("    Not enough data points (", nrow(Xp), ") or not enough unique outcomes (", length(unique(yp)), ") for pattern '", pattern_str, "'. Model not trained for this pattern.", sep=""))
            self$models_by_pattern[[pattern_str]] <- NULL # Explicitly set to NULL
          }
        }
      }
      TRUE
    },

    predict_probs = function(X_new, M_new) {
      X_new_df <- as.data.frame(X_new)
      M_new_matrix <- as.matrix(M_new)

      n_new_obs <- nrow(X_new_df)
      predictions <- numeric(n_new_obs)

      all_pattern_keys_new <- apply(M_new_matrix, 1, paste, collapse = "_")
      unique_pattern_keys_new <- unique(all_pattern_keys_new)
      
      for (pattern_str in unique_pattern_keys_new) {
        
        pattern_indices <- which(all_pattern_keys_new == pattern_str)
        
        current_m_mask_vec <- M_new_matrix[pattern_indices[1], ] # Mask is same for all rows with this pattern

        all_vars_missing <- all(current_m_mask_vec == 1)

        if (all_vars_missing || is.null(self$models_by_pattern[[pattern_str]])) {
          predictions[pattern_indices] <- self$default_prob
        } else {
          reg_model <- self$models_by_pattern[[pattern_str]]

          X_current_pattern <- X_new_df[pattern_indices, current_m_mask_vec == 0, drop = FALSE]
          pattern_predictions <- predict(reg_model, newdata = X_current_pattern, type = "response")
          
          predictions[pattern_indices] <- pattern_predictions
        }
      }
      return(predictions)
    },

    return_params = function() {
      if (!self$return_beta) return(NULL)
    }
  )
)


CompleteCase <- R6::R6Class("CompleteCase",
  inherit = ImputationMethod,
  public = list(
    reg = NULL,                  
    default_prob = NULL,         

    initialize = function(name = "CC") {
      super$initialize(name)
      self$can_predict = FALSE
      self$return_beta = TRUE
      self$reg = NULL
      self$default_prob = 0.5 
    },

    fit = function(X, M, y, X_test = NULL, M_test = NULL) {
      self$can_predict = FALSE
      self$return_beta = TRUE
      self$reg = NULL

      X_df <- as.data.frame(X)
      M_matrix <- as.matrix(M)
      y_vec <- as.numeric(y)

      self$default_prob <- mean(y_vec, na.rm = TRUE)

      complete_case_indices <- apply(M_matrix == 0, 1, all)
      
      X_cc <- X_df[complete_case_indices, , drop = FALSE]
      y_cc <- y_vec[complete_case_indices]

      num_complete_cases <- nrow(X_cc)


      if (num_complete_cases > 0 && sum(y_cc == 0) > 0 && sum(y_cc == 1) > 0) {
        data_for_glm <- X_cc
        data_for_glm$y_outcome <- y_cc

        if (ncol(X_cc) > 0) {
          formula_str <- paste("y_outcome ~", paste(names(X_cc), collapse = " + "))
        } else {
          formula_str <- "y_outcome ~ 1"
          cat("  No features in complete cases. Training intercept-only model.\n")
        }

        self$reg <- tryCatch({
          glm(as.formula(formula_str), family = binomial(), data = data_for_glm)
        }, error = function(e) {
          message(paste("  Warning: GLM failed for complete cases. Error: ", e$message, sep=""))
          self$reg <- NULL
          return(NULL)
        })
      } else {
        cat("  Insufficient complete cases or only one class in complete cases. No model trained.\n")
        self$reg <- NULL
      }
      TRUE
    },

    predict = function(X, M) {
      return(NULL)
    },

    predict_probs = function(X, M) {
      return(NULL)
    },

    return_params = function() {

      model_coef <- coef(self$reg)
      intercept <- model_coef[1]
      coefficients <- model_coef[-1]

      names(intercept) <- NULL
      names(coefficients) <- NULL

      coef_str <- paste(coefficients, collapse = ", ")
      int_str <- as.character(intercept)

      return(sprintf("[[%s], [%s]]", coef_str, int_str))
    }
  )
)
