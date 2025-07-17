library(readr)
library(dplyr)
library(tidyr)
library(tools)

build_presentation_table <- function(combined_scores, methods_map, sim_names, sizes, metric_name) {
  scores_data <- combined_scores %>%
    filter(metric == metric_name)
  
  times_data <- combined_scores %>%
    filter(metric == "running_time_train") %>%
    group_by(method, n_train) %>%
    summarise(
      sd_score = sd(mean_score, na.rm = TRUE),
      mean_score = mean(mean_score, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    mutate(metric = "Time")
  
  scores_wide <- scores_data %>%
    pivot_wider(id_cols = method, 
                names_from = c(exp_name, n_train), 
                names_sep = "_", 
                values_from = c(mean_score, sd_score))
  
  times_wide <- times_data %>%
    pivot_wider(id_cols = method, 
                names_from = c(metric, n_train), 
                names_sep = "_", 
                values_from = c(mean_score, sd_score))
  
  table_df <- full_join(scores_wide, times_wide, by = "method")
  
  mean_col_order <- c()
  sd_col_order <- c()
  for (n_size in sizes) {
    for (sim_name in sim_names) {
      mean_col_order <- c(mean_col_order, paste("mean_score", sim_name, n_size, sep = "_"))
      sd_col_order <- c(sd_col_order, paste("sd_score", sim_name, n_size, sep = "_"))
    }
    mean_col_order <- c(mean_col_order, paste("mean_score_Time", n_size, sep = "_"))
    sd_col_order <- c(sd_col_order, paste("sd_score_Time", n_size, sep = "_"))
  }
  
  col_order <- c("method", unlist(mapply(c, mean_col_order, sd_col_order, SIMPLIFY = FALSE)))
  
  missing_cols <- setdiff(col_order, names(table_df))
  if (length(missing_cols) > 0) {
    for (col in missing_cols) {
      table_df[[col]] <- NA_real_
    }
  }
  
  # Ensure dataframe only contains columns that are actually needed and in the correct order
  table_df <- table_df[, intersect(col_order, names(table_df))]
  
  table_df <- table_df %>%
    slice(match(names(methods_map), method))
  
  return(table_df)
}

apply_latex_formatting <- function(df, methods_map) {
  formatted_df <- data.frame(method = paste0("\\texttt{", unlist(methods_map[df$method]), "}"))
  
  mean_cols <- names(df)[grepl("^mean_score", names(df))]
  
  for (mean_col_name in mean_cols) {
    sd_col_name <- sub("mean_score", "sd_score", mean_col_name)
    
    if (!sd_col_name %in% names(df)) next
    
    mean_values <- as.numeric(df[[mean_col_name]])
    sd_values <- as.numeric(df[[sd_col_name]])
    
    formatted_values <- rep("---", nrow(df))
    
    non_na_indices <- which(!is.na(mean_values))
    
    if (length(non_na_indices) > 0) {
      # Determine digits based on the minimum value in the current column
      min_in_column <- min(mean_values[non_na_indices], na.rm = TRUE)
      
      # If the minimum value is smaller than 0.01, use 3 digits, otherwise use 2
      # Handle potential Inf or NA from min_in_column if all values are NA/Inf
      if (is.finite(min_in_column) && min_in_column < 0.01) {
        num_digits <- 3
      } else {
        num_digits <- 2
      }
      
      # Apply numeric formatting first
      temp_formatted_values <- formatC(mean_values[non_na_indices], format = "f", digits = num_digits, preserve.width = "individual")
      
      # Now, apply the "+100" rule based on the original numeric values
      over_100_indices <- which(mean_values[non_na_indices] > 100)
      if (length(over_100_indices) > 0) {
        temp_formatted_values[over_100_indices] <- "+100"
      }
      
      # For columns where lower is better (all scores and time)
      # This part should only apply if the value is NOT "+100"
      # We need to make sure we don't try to find min/max for "+100" strings.
      # The min/max logic should operate on original numeric values
      
      # Re-filter non_na_indices to exclude those that became "+100"
      # Or, more simply, perform bold/underline logic *before* replacing with "+100"
      # Let's re-arrange the logic to apply bold/underline *then* replace with "+100*
      
      # Reset temp_formatted_values for re-application after bold/underline
      temp_formatted_values_for_markup <- formatC(mean_values[non_na_indices], format = "f", digits = num_digits, preserve.width = "individual")
      
      min_val_index <- which.min(mean_values[non_na_indices])
      min_val <- mean_values[non_na_indices][min_val_index]
      std_of_min <- sd_values[non_na_indices][min_val_index]
      
      if (!is.na(min_val) && !is.na(std_of_min)) {
        bold_threshold <- min_val + std_of_min
        underline_threshold <- min_val + 2 * std_of_min
        
        scores_to_format <- mean_values[non_na_indices]
        
        bold_indices <- which(scores_to_format <= bold_threshold)
        underline_indices <- which(scores_to_format > bold_threshold & scores_to_format <= underline_threshold)
        
        if(length(bold_indices) > 0) temp_formatted_values_for_markup[bold_indices] <- paste0("\\textbf{", temp_formatted_values_for_markup[bold_indices], "}")
        if(length(underline_indices) > 0) temp_formatted_values_for_markup[underline_indices] <- paste0("\\underline{", temp_formatted_values_for_markup[underline_indices], "}")
      }
      
      # Now, re-apply the "+100" rule on the potentially bold/underlined strings
      over_100_indices <- which(mean_values[non_na_indices] > 1000)
      if (length(over_100_indices) > 0) {
        temp_formatted_values_for_markup[over_100_indices] <- "+1000"
      }
      
      formatted_values[non_na_indices] <- temp_formatted_values_for_markup
    }
    
    output_col_name <- sub("mean_score_", "", mean_col_name)
    formatted_df[[output_col_name]] <- formatted_values
  }
  return(formatted_df)
}


generate_latex_from_df <- function(formatted_df, metric_name) {
  metric_title <- toTitleCase(gsub("_", " ", metric_name))
  caption <- paste("Summary of", metric_title, "Results")
  label <- paste0("tab:", metric_name, "_summary")
  
  col_spec <- "l cccc c cccc c"
  
  header_main <- "& \\multicolumn{5}{c}{Low sample size ($N=100$)} & \\multicolumn{5}{c}{High sample size ($N=50,000$)} \\\\"
  sim_headers <- c("MCAR", "MAR", "MNAR", "NL")
  header_sub <- paste0("& ", paste(sim_headers, collapse = " & "), " & Time & ", paste(sim_headers, collapse = " & "), " & Time \\\\")
  
  latex_output_lines <- c(
    "\\begin{table}[h!]",
    "\\centering",
    paste0("\\begin{tabular}{", col_spec, "}"),
    "\\toprule",
    "\\multirow{2}{*}{Methods}",
    header_main,
    "\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}",
    header_sub,
    "\\midrule",
    apply(formatted_df, 1, function(row) paste0(paste(row, collapse = " & "), " \\\\")),
    "\\bottomrule",
    "\\end{tabular}",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\end{table}"
  )
  
  return(paste(latex_output_lines, collapse = "\n"))
}

generate_metric_table <- function(metric_name, bayes.diff=TRUE, multiplier=1) {
  Sims <- c("SimMCAR", "SimMAR", "SimMNAR", "SimNL")
  Sims.names <- c("MCAR (gaussian)", "GPMM-MAR", "GPMM-MNAR", "MCAR (non-linear)")
  train.sizes <- c(100, 50000)
  
  method_map_csv_to_latex <- list(
    "PbP.Fixed" = "PbP",
    # "CC" = "CC",
    "SAEM" = "SAEM",
    "Mean.IMP" = "Mean.IMP",
    "Mean.IMP.M" = "Mean.IMP.M",
    "MICE.1.IMP" = "MICE.1.IMP",
    "MICE.1.Y.IMP" = "MICE.1.Y.IMP",
    "MICE.1.Y.M.IMP.M" = "MICE.1.Y.M.IMP.M",
    "MICE.100.IMP" = "MICE.100.IMP",
    "MICE.100.Y.IMP" = "MICE.100.Y.IMP",
    "MICE.100.Y.M.IMP.M" = "MICE.100.Y.M.IMP.M",
    "MICE.RF.10.IMP" = "MICE.RF.10.IMP",
    "MICE.RF.10.Y.IMP" = "MICE.RF.10.Y.IMP",
    "MICE.RF.10.Y.M.IMP.M" = "MICE.RF.10.Y.M.IMP.M"
  )
  
  all_scores_data <- list()
  
  for (i in seq_along(Sims)) {
    sim_exp <- Sims[i]
    sim_name <- Sims.names[i]
    file_path <- file.path("data", sim_exp, "score_matrix.csv")
    
    if (file.exists(file_path)) {
      raw_file <- read_csv(file_path, show_col_types = FALSE)
      
      score_matrix_df <- raw_file %>%
        filter(filter == "all",
               method %in% names(method_map_csv_to_latex),
               metric == metric_name,
               bayes_adj == bayes.diff,
               n_train %in% train.sizes
        ) %>%
        mutate(score = multiplier * as.numeric(score)) %>%
        select(method, n_train, metric, score) %>%
        group_by(method, n_train, metric) %>%
        summarise(mean_score = mean(score, na.rm = TRUE), sd_score = sd(score, na.rm = TRUE), .groups = "drop") %>%
        mutate(exp_name = sim_name)
      
      running_time_matrix_df <- raw_file %>%
        filter(filter == "all",
               method %in% names(method_map_csv_to_latex),
               metric == "running_time_train",
               n_train %in% train.sizes
        ) %>%
        mutate(score = as.numeric(score)) %>%
        select(method, n_train, metric, score) %>%
        group_by(method, n_train, metric) %>%
        summarise(mean_score = mean(score, na.rm = TRUE), sd_score = sd(score, na.rm = TRUE), .groups = "drop") %>%
        mutate(exp_name = sim_name)
      
      all_scores_data[[sim_exp]] <- rbind(score_matrix_df, running_time_matrix_df)
    } else {
      warning(paste("File not found for simulation:", sim_exp, ". Skipping."))
    }
  }
  
  if (length(all_scores_data) == 0) {
    stop("No data found for any simulation. Stopping.")
  }
  
  combined_scores_df <- bind_rows(all_scores_data)
  
  numeric_table <- build_presentation_table(combined_scores_df, method_map_csv_to_latex, Sims.names, train.sizes, metric_name)
  formatted_table <- apply_latex_formatting(numeric_table, method_map_csv_to_latex)
  final_latex_code <- generate_latex_from_df(formatted_table, metric_name)
  
  output_filename <- paste0("plots_scripts/tables/", metric_name, "_summary.tex")
  writeLines(final_latex_code, output_filename)
  
  cat(paste("Successfully generated table for metric '", metric_name, "' and saved to '", output_filename, "'.\n", sep = ""))
}

# --- SCRIPT EXECUTION ---
# Simply call the main function with the desired metric name.
generate_metric_table("calibration", bayes.diff=TRUE, multiplier=1)
