# Vivli - Anticipating Resistance Risks to Cefiderocol in MDR Pathogens

## Overview

This project combines multiple prediction models to provide evidence-based antibiotic recommendations. The system uses machine learning to recommend antibiotics in optimal order of efficacy, with cefiderocol as the last resort option.

## 🏗️ Architecture

The system follows a structured 4-step methodology:

1. **Step 1**: Data Preparation and Exploration
2. **Step 2**: Antibiotic Decision Tree Model Development
3. **Step 3**: Phenotypic Signature Analysis and Clustering
4. **Step 4**: Cefiderocol Use Prediction Model

## 📁 Project Structure

```
Vivli/
├── scripts/                    # Python and R scripts
│   ├── antibiotic_decision_tree.py
│   ├── step4_prediction.py
│   ├── generate_english_antibiotic_report.py
│   ├── convert_md_to_html.py
│   ├── multiple_regression_plot.R
│   ├── univariate_analysis_script.R
│   └── ...
├── docs/                       # Documentation
│   ├── vivli_complete_methodology.md
│   ├── vivli_complete_methodology.html
│   ├── antibiotic_recommendation_system_details_english.md
│   ├── last_prediction_model_details.md
│   └── ...
├── outputs/                    # Generated outputs
│   ├── reports/               # PDF and HTML reports
│   ├── plots/                 # Visualizations
│   └── models/                # Trained models
├── data/                      # Data files (not included in repo)
│   ├── 1.xlsx                # SIDERO-WT Database
│   └── 2.xlsx                # ATLAS Database
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap reportlab markdown
```

### Running the System

1. **Antibiotic Decision Tree Model**:
```bash
cd scripts
python antibiotic_decision_tree.py
```

2. **Cefiderocol Prediction Model**:
```bash
cd scripts
python step4_prediction.py
```

3. **Generate English Report**:
```bash
cd scripts
python generate_english_antibiotic_report.py
```

## 📊 Models

### 1. Antibiotic Decision Tree Model
- **Algorithm**: Decision Tree Classifier
- **Features**: 7 primary features (species, country, year, resistance patterns)
- **Performance**: 100% accuracy (reported)
- **Output**: Complete antibiotic sequence with cefiderocol as last resort

### 2. Cefiderocol Use Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: 20+ features including MIC values, resistance patterns, ratios
- **Performance**: AUC 1.000, Precision 1.000, Recall 1.000
- **Output**: Binary decision for cefiderocol use

### 3. Phenotypic Signature Analysis
- **Method**: Clustering analysis with PCA
- **Purpose**: Identify resistance patterns and signatures
- **Output**: Cluster assignments and phenotypic signatures

## 📈 Performance Metrics

### Antibiotic Decision Tree
- **Accuracy**: 100%
- **Target**: First antibiotic recommendation
- **Coverage**: 40+ antibiotics analyzed


## 🏥 Clinical Applications

### Decision Framework
1. **First-Line Treatment**: Most effective antibiotic based on species, region, resistance
2. **Sequential Alternatives**: Complete sequence of alternatives
3. **Phenotypic Analysis**: Resistance patterns and clusters
4. **Last Resort Decision**: Cefiderocol use determination

### Example Usage
```python
# Get antibiotic recommendations
recommendations = model.recommend_antibiotics(
    species="Escherichia coli",
    country="France", 
    year=2023,
    resistance_profile={'beta_lactam': 0.3, 'quinolone': 0.7}
)
```

## 📋 Data Sources

- **ATLAS Database** (2.xlsx): Global antimicrobial susceptibility data
  - 966,805 isolates, 134 variables
  - Multiple countries and species
  - Temporal coverage

- **SIDERO-WT Database** (1.xlsx): Cefiderocol-specific susceptibility data
  - MIC values and resistance patterns
  - Species and geographic information

## ⚠️ Important Limitations

### Critical Considerations
- **No real treatment failure data** available
- **Targets based on theoretical resistance patterns**
- **High performance likely reflects simplified target definitions**
- **Geographic and temporal biases possible**

### Clinical Implementation
- **Do NOT use for clinical decisions** without validation
- **Validate on real treatment failure data**
- **Include clinical factors** (comorbidities, previous exposure)
- **Prospective validation** required before implementation
- **Use as research tool** only

## 🔬 Technical Details

### Feature Engineering
- MIC value standardization
- Resistance threshold application
- Categorical encoding
- Composite resistance scores
- MIC ratios for comparative analysis

### Model Training
- Train/Test Split: 80/20
- Cross-validation: 5-fold StratifiedKFold
- Feature scaling: StandardScaler
- Random State: 42

### Dependencies
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting
- **SHAP**: Feature importance analysis
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **ReportLab**: PDF generation

## 📚 Documentation

### Complete Documentation
- **Methodology**: `docs/vivli_complete_methodology.html`
- **Antibiotic System**: `docs/antibiotic_recommendation_system_details_english.html`
- **Cefiderocol Model**: `docs/last_prediction_model_details.html`

### Reports
- **English PDF Report**: `outputs/reports/antibiotic_recommendation_report_english.pdf`
- **Methodology HTML**: `docs/vivli_complete_methodology.html`

## 🚀 Future Directions

### 1. Clinical Integration
- Include patient factors (age, comorbidities, allergies)
- Add pharmacokinetic considerations
- Integrate with local resistance patterns

### 2. Model Improvements
- Include genomic resistance markers
- Add temporal resistance trends
- Develop species-specific models

### 3. Clinical Validation
- Prospective clinical studies
- Real-world implementation
- Outcome assessment

### 4. Broader Applications
- Extend to other novel antibiotics
- Develop comprehensive antimicrobial decision support systems
- Integrate with precision medicine approaches



### Authors
Adekemi Adepeju, Christian Ako, Abeeb Adeniyi, Oluwatobiloba Kazeem, Oluwadamilare Olatunbosun using the Pfizer's ATLAS dataset and Sidero datatset as part of the 2025 Vivli AMR Data Challenge.
