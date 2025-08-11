rm(list = ls())
library(lubridate)
library(tidyverse)
library(dplyr)
library(readxl)
library(DT)
library(tidyr)
library(readr)
library(httr)
library(scales)  # for percent_format
library(RColorBrewer) # for prettier palettes
library(hrbrthemes)
library(ggpmisc)
library(glue)
library(patchwork)
library(car)  # for partial regression plots
library(broom)  # for tidy regression output

################################################################################

sidero_dataset <- read_excel("Downloads/Sidero-dataset.xlsx")
colnames(sidero_dataset)
unique(sidero_dataset$`Organism Name`)

################################################################################

# Define groups of organisms
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

################################################################################
# Apply CLSI interpretation logic
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

################################################################################
# Define organism groups for bar plots
sidero_dataset <- sidero_dataset %>%
  mutate(
    organism_group = case_when(
      `Organism Name` == "Stenotrophomonas maltophilia" ~ "Stenotrophomonas",
      `Organism Name` %in% acinetobacter_species ~ "Acinetobacter",
      `Organism Name` %in% pseudomonas_species ~ "Pseudomonas",
      `Organism Name` %in% enterobacterales ~ "Enterobacterales",
      TRUE ~ "Other"
    )
  )

################################################################################
# Count isolates by country
country_counts <- sidero_dataset %>%
  count(Country, sort = TRUE)

# Plot
ggplot(country_counts, aes(x = reorder(Country, n), y = n)) +
  geom_bar(stat = "identity", fill = "#1f78b4") +
  coord_flip() +
  labs(
    title = "Number of Isolates per Country",
    x = "Country",
    y = "Number of Isolates"
  ) +
  theme_minimal(base_size = 14)

################################################################################
# Count isolates by region
region_counts <- sidero_dataset %>%
  count(Region, sort = TRUE)

# Plot
ggplot(region_counts, aes(x = reorder(Region, n), y = n)) +
  geom_bar(stat = "identity", fill = "#33a02c") +
  coord_flip() +
  labs(
    title = "Number of Isolates per Region",
    x = "Region",
    y = "Number of Isolates"
  ) +
  theme_minimal(base_size = 14)

################################################################################
# Count top 20 organisms
top_organisms <- sidero_dataset %>%
  count(`Organism Name`, sort = TRUE) %>%
  slice_max(n, n = 20)

# Plot
ggplot(top_organisms, aes(x = reorder(`Organism Name`, n), y = n)) +
  geom_bar(stat = "identity", fill = "#e31a1c") +
  coord_flip() +
  labs(
    title = "Top 20 Most Frequent Organisms",
    x = "Organism Name",
    y = "Number of Isolates"
  ) +
  theme_minimal(base_size = 14)

################################################################################
# Count isolates by organism group
group_counts <- sidero_dataset %>%
  count(organism_group, sort = TRUE)

# Plot
ggplot(group_counts, aes(x = reorder(organism_group, n), y = n)) +
  geom_bar(stat = "identity", fill = "#6a3d9a") +
  coord_flip() +
  labs(
    title = "Number of Isolates by Organism Group",
    x = "Organism Group",
    y = "Number of Isolates"
  ) +
  theme_minimal(base_size = 14)

################################################################################
# Plot Cefiderocol susceptibility by organism group
sidero_dataset %>%
  filter(!is.na(cefiderocol_category)) %>%
  ggplot(aes(x = organism_group, fill = cefiderocol_category)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(
    values = c(
      "Susceptible" = "#1E88E5",   # Vivid blue
      "Intermediate" = "#FFB300", # Bright orange
      "Resistant" = "#D32F2F"     # Strong red
    )
  ) +
  labs(
    title = "Cefiderocol Susceptibility by Organism Group",
    x = "Organism Group",
    y = "Proportion",
    fill = "Cefiderocol Category"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

################################################################################
# Non-susceptible isolates by region
non_susceptible_data <- sidero_dataset %>%
  filter(cefiderocol_category %in% c("Intermediate", "Resistant"))

# Pre-calculate counts for each Region and Category
cef_count <- non_susceptible_data %>%
  count(Region, cefiderocol_category)

# Plot
ggplot(cef_count, aes(x = Region, y = n, fill = cefiderocol_category)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.6), width = 0.6, colour = "black") +
  geom_text(
    aes(label = n),
    position = position_dodge(width = 0.6),
    vjust = -0.4,
    size = 4
  ) +
  scale_fill_manual(
    values = c(
      "Intermediate" = "#FFA000",  # sharp orange
      "Resistant" = "#D32F2F"      # vivid red
    )
  ) +
  labs(
    title = "Non-Susceptible Cefiderocol Isolates by Region",
    subtitle = "Categorized by Intermediate and Resistant",
    x = "Region",
    y = "Number of Isolates",
    fill = "Cefiderocol Category"
  ) +
  theme_ipsum(base_size = 14, axis_title_size = 14) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 13)
  )

################################################################################
# Regression Analysis for Correlation and Multiple Regression with Cefiderocol MIC

# Define antibiotics
abx_list <- c(
  "Cefepime",
  "Ceftolozane/ Tazobactam",
  "Ceftazidime/ Avibactam",
  "Meropenem",
  "Ciprofloxacin",
  "Colistin"
)

# Define Klebsiella species
klebsiella_species <- c(
  "Klebsiella pneumoniae", "Klebsiella oxytoca", "Klebsiella aerogenes"
)

# Clean MIC data to handle non-numeric values
sidero_dataset <- sidero_dataset %>%
  mutate(
    across(
      c(Cefiderocol, all_of(abx_list)),
      ~ as.numeric(gsub("[><=]", "", as.character(.)))  # Remove >, <, = and convert to numeric
    )
  )

# Create organism group column for regression analysis
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

# Prepare MIC data
mic_data <- sidero_dataset %>%
  filter(!is.na(organism_group_regression)) %>%
  select(organism_group_regression, Cefiderocol, all_of(abx_list), cefiderocol_category) %>%
  drop_na()

# Function to safely compute correlation
safe_cor_test <- function(x, y) {
  valid_pairs <- sum(!is.na(x) & !is.na(y))
  if (valid_pairs < 3 || var(x, na.rm = TRUE) == 0 || var(y, na.rm = TRUE) == 0) {
    return(list(correlation = NA_real_, p_value = NA_real_, n = valid_pairs))
  } else {
    result <- tryCatch(
      {
        cor_test <- cor.test(x, y, method = "pearson")
        list(correlation = as.numeric(cor_test$estimate), p_value = cor_test$p.value, n = valid_pairs)
      },
      error = function(e) {
        message("Correlation failed for pair: ", e$message)
        list(correlation = NA_real_, p_value = NA_real_, n = valid_pairs)
      }
    )
    return(result)
  }
}

# Calculate correlations and p-values for each antibiotic and organism group
correlation_results <- mic_data %>%
  group_by(organism_group_regression) %>%
  reframe(
    across(
      all_of(abx_list),
      ~ list(safe_cor_test(Cefiderocol, .x)),
      .names = "{.col}"
    )
  ) %>%
  pivot_longer(
    cols = all_of(abx_list),
    names_to = "Antibiotic",
    values_to = "Stats"
  ) %>%
  unnest_wider(Stats, names_sep = "_") %>%
  mutate(
    Correlation = round(Stats_correlation, 3),
    P_Value = round(Stats_p_value, 4),
    Significance = case_when(
      is.na(P_Value) ~ "NA",
      P_Value < 0.001 ~ "***",
      P_Value < 0.01 ~ "**",
      P_Value < 0.05 ~ "*",
      TRUE ~ ""
    ),
    Sample_Size = as.integer(Stats_n)
  ) %>%
  rename(Organism_Group = organism_group_regression) %>%
  select(-Stats_correlation, -Stats_p_value, -Stats_n)

# Print correlation table
print("Correlation of Cefiderocol MIC with Other Antibiotics by Organism Group:")
print(correlation_results)

# Multiple regression analysis
multiple_regression_results <- mic_data %>%
  group_by(organism_group_regression) %>%
  do({
    model <- tryCatch(
      {
        lm(Cefiderocol ~ Cefepime + `Ceftolozane/ Tazobactam` + `Ceftazidime/ Avibactam` + Meropenem + Ciprofloxacin + Colistin, data = .)
      },
      error = function(e) {
        message("Multiple regression failed for group: ", .$organism_group_regression[1], " - ", e$message)
        NULL
      }
    )
    if (is.null(model)) {
      tibble(term = character(), estimate = numeric(), std.error = numeric(), statistic = numeric(), p.value = numeric())
    } else {
      tidy(model) %>%
        mutate(p.value = round(p.value, 4)) %>%
        filter(term != "(Intercept)")
    }
  }) %>%
  rename(Organism_Group = organism_group_regression)

# Print multiple regression results
print("Multiple Regression Coefficients for Predicting Cefiderocol MIC:")
print(multiple_regression_results)

# Generate partial regression plots for each organism group
for (group in unique(mic_data$organism_group_regression)) {
  group_data <- mic_data %>% filter(organism_group_regression == group)
  if (nrow(group_data) >= 3) {  # Ensure sufficient data
    model <- tryCatch(
      {
        lm(Cefiderocol ~ Cefepime + `Ceftolozane/ Tazobactam` + `Ceftazidime/ Avibactam` + Meropenem + Ciprofloxacin + Colistin, data = group_data)
      },
      error = function(e) NULL
    )
    if (!is.null(model)) {
      cat("Generating partial regression plots for", group, "\n")
      avPlots(model, main = paste("Partial Regression Plots for", group), 
              col.points = case_when(group_data$cefiderocol_category == "Susceptible" ~ "#1E88E5",
                                     group_data$cefiderocol_category == "Intermediate" ~ "#FFB300",
                                     group_data$cefiderocol_category == "Resistant" ~ "#D32F2F"),
              col.lines = "#D55E00")
    }
  }
}

# Create single-variable regression plots with equation, R², and p-value
plots <- map(abx_list, function(abx) {
  ggplot(mic_data, aes(x = Cefiderocol, y = .data[[abx]], color = cefiderocol_category)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", se = TRUE, color = "#D55E00") +
    stat_poly_eq(
      aes(label = paste(..eq.label.., ..rr.label.., ..p.value.label.., sep = "~~~")),
      formula = y ~ x,
      parse = TRUE,
      size = 4,
      label.x.npc = "right",
      label.y.npc = 0.95,
      color = "black"
    ) +
    facet_wrap(~ organism_group_regression, scales = "free") +
    scale_color_manual(
      values = c(
        "Susceptible" = "#1E88E5",
        "Intermediate" = "#FFB300",
        "Resistant" = "#D32F2F"
      )
    ) +
    labs(
      x = "Cefiderocol MIC (µg/mL)",
      y = glue("{abx} MIC (µg/mL)"),
      title = glue("Cefiderocol vs {abx} by Organism Group"),
      color = "Cefiderocol Category"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      strip.text = element_text(size = 12, face = "bold"),
      legend.position = "top"
    )
})

# Combine single-variable regression plots
wrap_plots(plots, ncol = 2) +
  plot_annotation(
    title = "Single-Variable Regression of Cefiderocol MIC vs Other Antibiotics",
    subtitle = "Stratified by E. coli, Acinetobacter spp., Pseudomonas spp., and Klebsiella spp.",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 13)
    )
  )



################################################################################


# Multiple Regression Analysis with Visualization and Table
# Add this code after your existing analysis

library(broom)
library(kableExtra)
library(gridExtra)
library(grid)

# Enhanced Multiple Regression Analysis with Model Metrics
enhanced_regression_results <- mic_data %>%
  group_by(organism_group_regression) %>%
  do({
    group_name <- .$organism_group_regression[1]
    n_obs <- nrow(.)
    
    if(n_obs < 10) {
      # Return empty results for small sample sizes
      tibble(
        term = character(0),
        estimate = numeric(0),
        std.error = numeric(0),
        statistic = numeric(0),
        p.value = numeric(0),
        r.squared = NA_real_,
        adj.r.squared = NA_real_,
        f.statistic = NA_real_,
        f.p.value = NA_real_,
        sample_size = n_obs
      )
    } else {
      model <- tryCatch({
        lm(Cefiderocol ~ Cefepime + `Ceftolozane/ Tazobactam` + `Ceftazidime/ Avibactam` + 
             Meropenem + Ciprofloxacin + Colistin, data = .)
      }, error = function(e) NULL)
      
      if(!is.null(model)) {
        model_summary <- summary(model)
        model_tidy <- tidy(model) %>% filter(term != "(Intercept)")
        
        model_tidy %>%
          mutate(
            r.squared = model_summary$r.squared,
            adj.r.squared = model_summary$adj.r.squared,
            f.statistic = model_summary$fstatistic[1],
            f.p.value = pf(model_summary$fstatistic[1], 
                           model_summary$fstatistic[2], 
                           model_summary$fstatistic[3], 
                           lower.tail = FALSE),
            sample_size = n_obs
          )
      } else {
        tibble(
          term = abx_list,
          estimate = NA_real_,
          std.error = NA_real_,
          statistic = NA_real_,
          p.value = NA_real_,
          r.squared = NA_real_,
          adj.r.squared = NA_real_,
          f.statistic = NA_real_,
          f.p.value = NA_real_,
          sample_size = n_obs
        )
      }
    }
  }) %>%
  ungroup() %>%
  mutate(
    significance = case_when(
      is.na(p.value) ~ "",
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      p.value < 0.1 ~ ".",
      TRUE ~ ""
    ),
    estimate_formatted = ifelse(is.na(estimate), "NA", sprintf("%.3f", estimate)),
    std.error_formatted = ifelse(is.na(std.error), "NA", sprintf("%.3f", std.error)),
    p.value_formatted = ifelse(is.na(p.value), "NA", 
                               ifelse(p.value < 0.001, "<0.001", sprintf("%.3f", p.value)))
  )

# Create comprehensive regression table
regression_table <- enhanced_regression_results %>%
  select(organism_group_regression, term, estimate_formatted, std.error_formatted, 
         p.value_formatted, significance, r.squared, adj.r.squared, sample_size) %>%
  rename(
    "Organism Group" = organism_group_regression,
    "Antibiotic" = term,
    "Coefficient" = estimate_formatted,
    "Std Error" = std.error_formatted,
    "P-value" = p.value_formatted,
    "Sig." = significance,
    "R²" = r.squared,
    "Adj R²" = adj.r.squared,
    "N" = sample_size
  ) %>%
  distinct(`Organism Group`, .keep_all = TRUE) %>%
  select(`Organism Group`, `R²`, `Adj R²`, `N`)

# Create detailed coefficient table
coeff_table <- enhanced_regression_results %>%
  select(organism_group_regression, term, estimate_formatted, std.error_formatted, 
         p.value_formatted, significance) %>%
  rename(
    "Organism Group" = organism_group_regression,
    "Antibiotic" = term,
    "Coefficient" = estimate_formatted,
    "Std Error" = std.error_formatted,
    "P-value" = p.value_formatted,
    "Sig." = significance
  )

# Print formatted tables
cat("\n=== MULTIPLE REGRESSION MODEL SUMMARY ===\n")
print(kable(regression_table, format = "markdown", digits = 3, 
            caption = "Multiple Regression Model Summary by Organism Group"))

cat("\n=== REGRESSION COEFFICIENTS ===\n")
print(kable(coeff_table, format = "markdown", 
            caption = "Multiple Regression Coefficients for Predicting Cefiderocol MIC"))

# Create visualization of regression coefficients
coeff_plot_data <- enhanced_regression_results %>%
  filter(!is.na(estimate), !is.na(std.error)) %>%
  mutate(
    ci_lower = estimate - 1.96 * std.error,
    ci_upper = estimate + 1.96 * std.error,
    term_clean = case_when(
      term == "Ceftolozane/ Tazobactam" ~ "C/T",
      term == "Ceftazidime/ Avibactam" ~ "CZA",
      TRUE ~ str_replace(term, "Cef", "C")
    )
  )

# Coefficient plot
coeff_plot <- ggplot(coeff_plot_data, aes(x = term_clean, y = estimate, color = organism_group_regression)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                width = 0.2, position = position_dodge(width = 0.5)) +
  facet_wrap(~ organism_group_regression, scales = "free_x", ncol = 2) +
  scale_color_brewer(type = "qual", palette = "Set1") +
  labs(
    title = "Multiple Regression Coefficients with 95% Confidence Intervals",
    subtitle = "Predicting Cefiderocol MIC from Other Antibiotics",
    x = "Antibiotic Predictor",
    y = "Regression Coefficient (log2 scale)",
    color = "Organism Group"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

print(coeff_plot)

# Create predicted vs observed plot for each organism group
prediction_plots <- mic_data %>%
  group_by(organism_group_regression) %>%
  filter(n() >= 10) %>%
  do({
    group_data <- .
    model <- lm(Cefiderocol ~ Cefepime + `Ceftolozane/ Tazobactam` + `Ceftazidime/ Avibactam` + 
                  Meropenem + Ciprofloxacin + Colistin, data = group_data)
    
    group_data$predicted <- predict(model)
    group_data$residuals <- residuals(model)
    
    # Predicted vs Observed
    p1 <- ggplot(group_data, aes(x = predicted, y = Cefiderocol)) +
      geom_point(aes(color = cefiderocol_category), alpha = 0.7) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      geom_smooth(method = "lm", se = TRUE, color = "blue", alpha = 0.3) +
      scale_color_manual(values = c("Susceptible" = "#1E88E5", "Intermediate" = "#FFB300", "Resistant" = "#D32F2F")) +
      labs(
        title = paste("Predicted vs Observed:", unique(group_data$organism_group_regression)),
        x = "Predicted Cefiderocol MIC (log2)",
        y = "Observed Cefiderocol MIC (log2)",
        color = "Susceptibility"
      ) +
      theme_minimal() +
      theme(legend.position = "bottom")
    
    print(p1)
    
    group_data
  })

# Model comparison table
model_comparison <- enhanced_regression_results %>%
  filter(!is.na(r.squared)) %>%
  distinct(organism_group_regression, .keep_all = TRUE) %>%
  select(organism_group_regression, r.squared, adj.r.squared, f.statistic, f.p.value, sample_size) %>%
  mutate(
    r.squared = round(r.squared, 3),
    adj.r.squared = round(adj.r.squared, 3),
    f.statistic = round(f.statistic, 2),
    f.p.value_formatted = ifelse(f.p.value < 0.001, "<0.001", sprintf("%.3f", f.p.value))
  ) %>%
  rename(
    "Organism Group" = organism_group_regression,
    "R²" = r.squared,
    "Adj R²" = adj.r.squared,
    "F-statistic" = f.statistic,
    "F p-value" = f.p.value_formatted,
    "Sample Size" = sample_size
  )

cat("\n=== MODEL COMPARISON ===\n")
print(kable(model_comparison, format = "markdown", 
            caption = "Model Performance Comparison Across Organism Groups"))

# Significance codes
cat("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")