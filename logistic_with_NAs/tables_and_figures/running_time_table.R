library(readr)
library(dplyr)
library(tidyr)
library(tools)

exp <- "SimMCAR"
n_train_selected <- c(100,500,1000,5000,10000,50000)
methods_selected <- NULL # This remains unchanged as per previous instruction

df <- read.csv(file.path("data", exp, "simulation.csv")) %>% 
  select(method, n_train, running_time_train, running_time_pred) %>% 
  filter(n_train %in% n_train_selected)

# DO NOT CHANGE: methods_selected filtering logic remains as is
if (!is.null(methods_selected)){
  df <- df %>% filter(method %in% methods_selected)
}
if (!is.null(methods_selected)){
  df <- df %>% filter(method %in% methods_selected)
}

train_time_agg <- df %>%
  group_by(method, n_train) %>%
  summarise(mean_running_time_train = mean(running_time_train, na.rm = TRUE), .groups = 'drop') %>%
  pivot_wider(names_from = n_train, values_from = mean_running_time_train, names_prefix = "")

pred_time_agg <- df %>%
  group_by(method) %>%
  summarise(mean_running_time_pred = mean(running_time_pred, na.rm = TRUE), .groups = 'drop')

combined_agg_df <- left_join(train_time_agg, pred_time_agg, by = "method")

# Removed method_map_csv_to_latex_runtime
# Removed: ordered_methods <- names(method_map_csv_to_latex_runtime)[names(method_map_csv_to_latex_runtime) %in% combined_agg_df$method]
# Removed: combined_agg_df <- combined_agg_df %>% slice(match(ordered_methods, method))
# The methods in combined_agg_df will now be used as-is, in their default order.

generate_latex_table <- function(data_frame, caption_text, label_text) { # Removed methods_map parameter
  column_names <- colnames(data_frame)
  
  # Exclude 'method' and 'mean_running_time_pred' to get training time columns
  train_time_columns_numeric <- column_names[!column_names %in% c("method", "mean_running_time_pred")]
  
  num_train_cols <- length(train_time_columns_numeric)
  
  # The last column is for prediction time
  pred_col_header <- "15000" # As per your original table, this is the prediction sample size
  
  # Column specification: 'l' for method, 'c' for all numeric columns
  col_spec <- paste0("l ", paste(rep("c", num_train_cols), collapse = " "), " c")
  
  # Format table header
  header <- paste0(
    "\\begin{table}[h!]\n",
    "\\centering\n",
    "\\begin{tabular}{", col_spec, "}\n",
    "\\toprule\n",
    "\\textbf{Algorithms} & \\multicolumn{", num_train_cols, "}{c}{\\textbf{Training}} & \\multicolumn{1}{c}{\\textbf{Prediction}} \\\\\n",
    "\\cmidrule(lr){2-", num_train_cols + 1, "} \\cmidrule(lr){", num_train_cols + 2, "-", num_train_cols + 2, "}\n",
    "& ", paste(train_time_columns_numeric, collapse = " & "), " & ", pred_col_header, " \\\\\n",
    "\\midrule\n"
  )
  
  # Format table rows
  rows <- apply(data_frame, 1, function(row) {
    method_original <- row["method"]
    # Directly use the method_original for LaTeX \texttt{}
    method_latex <- paste0("\\texttt{", method_original, "}")
    
    train_times <- as.numeric(row[train_time_columns_numeric])
    
    # Format training times: fixed 3 digits
    formatted_train_times <- formatC(train_times, format = "f", digits = 3, preserve.width = "individual")
    
    pred_time <- as.numeric(row["mean_running_time_pred"])
    
    # Format prediction time: fixed 3 digits, or "---" for NA
    if (!is.na(pred_time)){
      formatted_pred_time <- formatC(pred_time, format = "f", digits = 3, preserve.width = "individual")
    } else {
      formatted_pred_time <- "---"
    }
    
    # Combine and ensure each row ends with \\
    paste0(method_latex, " & ", paste(formatted_train_times, collapse = " & "), " & ", formatted_pred_time, " \\\\\n")
  })
  
  # Format table footer
  footer <- paste0(
    "\\bottomrule\n",
    "\\end{tabular}\n",
    "\\caption{", caption_text, "}\n",
    "\\label{", label_text, "}\n",
    "\\end{table}\n"
  )
  
  # Combine header, rows, and footer
  paste0(header, paste(rows, collapse = ""), footer)
}

# Generate the LaTeX table
latex_output <- generate_latex_table(
  combined_agg_df,
  "Average training and prediction time, in seconds, of the procedures for different training sample sizes, for the experiment described in \\ref{sec:methodo_SimA}.",
  "tab:runtimeSimA"
) # Removed methods_map argument from the call

# Print the LaTeX output
cat(latex_output)

# Save the R script to a file
dir.create("plots_scripts/tables", recursive = TRUE, showWarnings = FALSE)
writeLines(latex_output, "plots_scripts/tables/runtime_table.tex")