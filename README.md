# Anticipating Cefiderocol Resistance in MDR Pathogens

## Overview

This project aims to identify early warning signs of cefiderocol resistance in multidrug-resistant (MDR) Gram-negative bacteria, a critical last-resort antibiotic. By analyzing phenotypic patterns in antibiotic susceptibility, we seek to pinpoint geographic and pathogen-specific risks to guide antimicrobial stewardship.

## Datasets

- SIDERO-WT: Contains Minimum Inhibitory Concentration (MIC) data for cefiderocol and comparator antibiotics (e.g., meropenem, ciprofloxacin) across North America and Europe (2014–2019).

- ATLAS: Global MIC data for comparator antibiotics, used to generalize findings from SIDERO-WT to other regions.

## Objectives

- Explore and standardize MIC data across datasets.

- Develop machine learning models (Logistic Regression, Random Forest, XGBoost) to predict cefiderocol resistance.

- Cluster resistance profiles to define phenotypic "signatures."

- Generalize findings to the ATLAS dataset to identify high-risk regions and species.

- Build an interactive dashboard and deliver a technical report.

# Methodology

Phase 1 (Setup & EDA): Standardize MIC units, perform exploratory data analysis (EDA) to study trends by year, region, and species.
Phase 2 (Model Development): Train classification models on SIDERO-WT (2014–2018) and test on 2019 data, evaluating with AUC, precision, and recall.
Phase 3 (Phenotypic Signatures): Use clustering (k-means, hierarchical) and PCA to identify resistance patterns.
Phase 4 (Generalization): Apply models to ATLAS to map resistance risks globally.
Phase 5 (Deliverables): Create a Streamlit dashboard, technical report, and presentation.
