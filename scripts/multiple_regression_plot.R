# Clear environment
rm(list = ls())

# Load required packages
library(tidyverse)
library(readxl)
library(car)
library(broom)
library(kableExtra)

# Settings
min_n_for_model <- 10     # Minimum observations for regression
epsilon <- 1e-6           # Small value for log2 transformation
out_dir <- "regression_outputs"
dir.create(out_dir, showWarnings = FALSE)

# Load dataset
sidero_dataset <- read_excel("Downloads/Sidero-dataset.xlsx")
cat("\nDataset columns:\n")
print(colnames(sidero_dataset))
cat("\nTotal rows in sidero_dataset:", nrow(sidero_dataset), "\n")
cat("\nMissing values in raw dataset:\n")
print(colSums(is.na(sidero_dataset)))

# Define organism groups
enterobacterales <- c(
  "Escherichia coli", "Klebsiella pneumoniae", "Klebsiella oxytoca", "Klebsiella aerogenes", 
  "Citrobacter koseri", "Citrobacter freundii", "Citrobacter braakii", "Citrobacter amalonaticus",
  "Citrobacter farmeri", "Citrobacter gillenii", "Citrobacter murliniae", "Citrobacter sakazakii", 
  "Citrobacter sedlakii", "Citrobacter youngae", "Enterobacter cloacae", "Enterobacter cloacae complex",
  "Enterobacter asburiae", "Enterobacter bugandensis", "Enterobacter cancerogenus", 
  "Enterobacter kobei", "Enterobacter ludwigii", "Enterobacter xiangfangensis",
  "Morganella morganii", "Proteus mirabilis", "Proteus vulgaris", "Proteus hauseri", "Proteus penneri",
  "Raoultella ornithinolytica", "Raoultella planticola", "Serratia marcescens", "Serratia liquefaciens",
  "Serratia ureilytica", "Serratia odorifera", "Serratia rubidaea", "Serratia fonticola", 
  "Serratia grimesii", "Serratia ficaria", "Providencia rettgeri", "Providencia stuartii", 
  "Lelliottia amnigena", "Kluyvera, non-speciated", "Pantoea calida", "Pantoea septica", 
  "Pantoea dispersa", "Pluralibacter gergoviae", "Cronobacter, non-speciated"
)
acinetobacter_species <- c(
  "Acinetobacter baumannii", "Acinetobacter baumannii complex", "Acinetobacter pittii", 
  "Acinetobacter nosocomialis", "Acinetobacter calcoaceticus", "Acinetobacter ursingii", 
  "Acinetobacter radioresistens", "Acinetobacter bereziniae", "Acinetobacter haemolyticus", 
  "Acinetobacter baylyi", "Acinetobacter seifertii", "Acinetobacter guillouiae", 
  "Acinetobacter lwoffii", "Acinetobacter johnsonii", "Acinetobacter parvus", 
  "Acinetobacter proteolyticus", "Acinetobacter sp.", "Acinetobacter junii", 
  "Acinetobacter courvalinii", "Acinetobacter schindleri", "Acinetobacter dispersus", 
  "Acinetobacter dijkshoorniae"
)
pseudomonas_species <- c(
  "Pseudomonas aeruginosa", "Pseudomonas mendocina", "Pseudomonas mosselii", 
  "Pseudomonas otitidis", "Pseudomonas putida", "Pseudomonas, non-speciated"
)
klebsiella_species <- c(
  "Klebsiella pneumoniae", "Klebsiella oxytoca", "Klebsiella aerogenes"
)

# Apply CLSI interpretation for cefiderocol susceptibility
sidero_dataset <- sidero_dataset %>%
  mutate(
    cefiderocol_category = case_when(
      `Organism Name` == "Stenotrophomonas maltophilia" & !is.na(Cefiderocol) & Cefiderocol <= 1 ~ "Susceptible",
      `Organism Name` == "Stenotrophomonas maltophilia" & !is.na(Cefiderocol) & Cefiderocol > 1 ~ "Nonsusceptible",
      `Organism Name` %in% acinetobacter_species & !is.na(Cefiderocol) & Cefiderocol <= 4 ~ "Susceptible",
      `Organism Name` %in% acinetobacter_species & Cefiderocol == 8 ~ "Intermediate",
      `Organism Name` %in% acinetobacter_species & !is.na(Cefiderocol) & Cefiderocol >= 16 ~ "Resistant",
      `Organism Name` %in% pseudomonas_species & !is.na(Cefiderocol) & Cefiderocol <= 4 ~ "Susceptible",
      `Organism Name` %in% pseudomonas_species & Cefiderocol == 8 ~ "Intermediate",
      `Organism Name` %in% pseudomonas_species & !is.na(Cefiderocol) & Cefiderocol >= 16 ~ "Resistant",
      `Organism Name` %in% enterobacterales & !is.na(Cefiderocol) & Cefiderocol <= 4 ~ "Susceptible",
      `Organism Name` %in% enterobacterales & Cefiderocol == 8 ~ "Intermediate",
      `Organism Name` %in% enterobacterales & !is.na(Cefiderocol) & Cefiderocol >= 16 ~ "Resistant",
      TRUE ~ NA_character_
    )
  )

# Define predictors
abx_list <- c("Cefepime", "Ceftolozane/ Tazobactam", "Ceftazidime/ Avibactam", "Meropenem", "Ciprofloxacin", "Colistin")

# Validate required columns
req_cols <- c("Organism Name", "Cefiderocol", abx_list)
missing_cols <- setdiff(req_cols, colnames(sidero_dataset))
if (length(missing_cols) > 0) stop("Missing required columns: ", paste(missing_cols, collapse = ", "))

# Clean and transform MIC data (log2 scale)
sidero_dataset <- sidero_dataset %>%
  mutate(
    across(
      c(Cefiderocol, all_of(abx_list)),
      ~ suppressWarnings(as.numeric(gsub("[><=]", "", as.character(.)))),
      .names = "{.col}_raw"
    ),
    across(
      c(Cefiderocol_raw, all_of(paste0(abx_list, "_raw"))),
      ~ log2(. + epsilon),
      .names = "log2_{sub('_.+$', '', .col)}"
    )
  )

# Debugging: Check transformation
cat("\nColumns after MIC transformation:\n")
print(colnames(sidero_dataset))
cat("\nMissing values after MIC transformation:\n")
print(colSums(is.na(sidero_dataset)))
cat("\nSample of log2_Cefiderocol:\n")
print(head(sidero_dataset$log2_Cefiderocol))

# Create organism group for regression
sidero_dataset <- sidero_dataset %>%
  mutate(
    organism_group_regression = case_when(
      `Organism Name` == "Escherichia coli" ~ "E. coli",
      `Organism Name` %in% acinetobacter_species ~ "Acinetobacter spp.",
      `Organism Name` %in% pseudomonas_species ~ "Pseudomonas spp.",
      `Organism Name` %in% klebsiella_species ~ "Klebsiella spp.",
      TRUE ~ NA_character_
    )
  )

# Prepare regression data
mic_data <- sidero_dataset %>%
  filter(!is.na(organism_group_regression)) %>%
  select(organism_group_regression, starts_with("log2_"), cefiderocol_category) %>%
  rename(
    Cefiderocol = log2_Cefiderocol,
    Cefepime = log2_Cefepime,
    `Ceftolozane/ Tazobactam` = `log2_Ceftolozane/ Tazobactam`,
    `Ceftazidime/ Avibactam` = `log2_Ceftazidime/ Avibactam`,
    Meropenem = log2_Meropenem,
    Ciprofloxacin = log2_Ciprofloxacin,
    Colistin = log2_Colistin
  ) %>%
  drop_na(Cefiderocol)

# Debugging: Check mic_data
cat("\nRows in mic_data:", nrow(mic_data), "\n")
cat("\nSample sizes per organism group:\n")
print(table(mic_data$organism_group_regression, useNA = "always"))
cat("\nMissing values in mic_data:\n")
print(colSums(is.na(mic_data)))

# Function to fit regression model
fit_regression <- function(df, group_name) {
  n_obs <- sum(complete.cases(select(df, all_of(c("Cefiderocol", abx_list)))))
  if (n_obs < min_n_for_model) {
    cat("Skipping regression for", group_name, "- insufficient data (n =", n_obs, ")\n")
    return(tibble(
      Organism_Group = group_name,
      Antibiotic = abx_list,
      Coefficient = NA_real_,
      Std_Error = NA_real_,
      P_Value = NA_real_,
      Significance = "NA",
      Sample_Size = n_obs,
      R_Squared = NA_real_,
      Adj_R_Squared = NA_real_,
      VIF = NA_real_
    ))
  }
  
  model <- tryCatch({
    lm(Cefiderocol ~ Cefepime + `Ceftolozane/ Tazobactam` + `Ceftazidime/ Avibactam` + 
         Meropenem + Ciprofloxacin + Colistin, data = df, na.action = na.omit)
  }, error = function(e) {
    cat("Regression failed for", group_name, ":", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(model)) {
    return(tibble(
      Organism_Group = group_name,
      Antibiotic = abx_list,
      Coefficient = NA_real_,
      Std_Error = NA_real_,
      P_Value = NA_real_,
      Significance = "NA",
      Sample_Size = n_obs,
      R_Squared = NA_real_,
      Adj_R_Squared = NA_real_,
      VIF = NA_real_
    ))
  }
  
  model_summary <- summary(model)
  tidy_coefs <- tidy(model) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      Antibiotic = term,
      Coefficient = round(estimate, 3),
      Std_Error = round(std.error, 3),
      P_Value = round(p.value, 3),
      Significance = case_when(
        P_Value < 0.001 ~ "***",
        P_Value < 0.01 ~ "**",
        P_Value < 0.05 ~ "*",
        P_Value < 0.1 ~ ".",
        TRUE ~ ""
      )
    )
  
  vif_vals <- tryCatch({
    vifs <- vif(model)
    if (is.numeric(vifs)) {
      tibble(Antibiotic = names(vifs), VIF = round(as.numeric(vifs), 2))
    } else {
      tibble(Antibiotic = abx_list, VIF = NA_real_)
    }
  }, error = function(e) {
    tibble(Antibiotic = abx_list, VIF = NA_real_)
  })
  
  tibble(
    Organism_Group = group_name,
    Antibiotic = tidy_coefs$Antibiotic,
    Coefficient = tidy_coefs$Coefficient,
    Std_Error = tidy_coefs$Std_Error,
    P_Value = tidy_coefs$P_Value,
    Significance = tidy_coefs$Significance,
    Sample_Size = n_obs,
    R_Squared = round(model_summary$r.squared, 3),
    Adj_R_Squared = round(model_summary$adj.r.squared, 3),
    VIF = vif_vals$VIF[match(tidy_coefs$Antibiotic, vif_vals$Antibiotic)]
  )
}

# Run regressions
groups <- unique(na.omit(mic_data$organism_group_regression))
regression_results <- map_dfr(groups, ~ fit_regression(mic_data %>% filter(organism_group_regression == .x), .x))

# Save and print results
write_csv(regression_results, file.path(out_dir, "regression_results.csv"))
cat("\n=== REGRESSION RESULTS ===\n")
print(kable(regression_results, format = "markdown", digits = 3, 
            caption = "Multiple Regression Results for Cefiderocol MIC Prediction"))

# Partial regression plots for resistance signals
for (group in groups) {
  group_data <- mic_data %>% filter(organism_group_regression == group)
  n_obs <- sum(complete.cases(select(group_data, all_of(c("Cefiderocol", abx_list)))))
  if (n_obs >= min_n_for_model) {
    model <- tryCatch({
      lm(Cefiderocol ~ Cefepime + `Ceftolozane/ Tazobactam` + `Ceftazidime/ Avibactam` + 
           Meropenem + Ciprofloxacin + Colistin, data = group_data, na.action = na.omit)
    }, error = function(e) NULL)
    
    if (!is.null(model)) {
      out_file <- file.path(out_dir, paste0("partial_regression_", gsub("[^A-Za-z0-9]", "_", group), ".png"))
      png(out_file, width = 800, height = 600)
      avPlots(model, main = paste("Partial Regression Plots for", group), 
              col.points = case_when(
                group_data$cefiderocol_category == "Susceptible" ~ "#1E88E5",
                group_data$cefiderocol_category == "Intermediate" ~ "#FFB300",
                group_data$cefiderocol_category == "Resistant" ~ "#D32F2F",
                TRUE ~ "gray"
              ),
              col.lines = "#D55E00")
      dev.off()
      cat("Saved partial regression plot for", group, "to", out_file, "\n")
    }
  }
}

cat("\nOutputs saved to:", normalizePath(out_dir), "\n")
cat("Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")


################################################################################


# Install packages if needed
install.packages(c("flextable", "officer"))

library(flextable)
library(officer)

# Assuming regression_results is already in your environment
# Create a clean table
ft <- flextable(regression_results) |>
  autofit() |> 
  theme_booktabs() |> 
  fontsize(size = 11, part = "all") |> 
  bold(part = "header") |> 
  align(align = "center", part = "all")

# Save as Word document
save_as_docx(ft, path = "regression_results_table.docx")

flextable(regression_results) |> 
  autofit() |> 
  theme_booktabs()
