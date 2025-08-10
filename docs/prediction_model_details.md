# Prediction Model Details - Vivli Antibiotic System

## Overview

The Vivli system contains multiple prediction models designed to support antibiotic decision-making. This document provides detailed information about each model's architecture, features, performance, and clinical applications.

## 1. Antibiotic Decision Tree Model

### Purpose
Recommends antibiotics in optimal order of efficacy, with cefiderocol as last resort.

### Model Architecture
- **Algorithm**: Decision Tree Classifier
- **Max Depth**: 10
- **Random State**: 42
- **Train/Test Split**: 80/20

### Features Used
1. **Species encoded** (LabelEncoder)
2. **Country encoded** (LabelEncoder) 
3. **Year** (temporal data)
4. **Beta-lactam resistance** (mean resistance score)
5. **Aminoglycoside resistance** (mean resistance score)
6. **Quinolone resistance** (mean resistance score)
7. **Other resistance** (mean resistance score)

### Training Process
1. Load ATLAS data (2.xlsx)
2. Clean MIC values for 40+ antibiotics
3. Calculate resistance scores using clinical breakpoints
4. Determine optimal antibiotic order based on global efficacy
5. Create decision features from resistance patterns
6. Train decision tree to predict first-choice antibiotic

### Performance
- **Accuracy**: 100% (reported)
- **Target**: First antibiotic recommendation
- **Output**: Ordered list of antibiotics with cefiderocol last

### Clinical Application
- Provides antibiotic sequence recommendations
- Considers species, geographic region, and resistance patterns
- Prioritizes antibiotics by efficacy while preserving cefiderocol

---

## 2. Cefiderocol Treatment Failure Prediction Model

### Purpose
Predicts risk of cefiderocol treatment failure based on resistance patterns.

### Model Architecture
- **Algorithms Tested**: 
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- **Best Model**: Logistic Regression
- **Cross-validation**: StratifiedKFold

### Target Definition
Treatment failure risk defined by complex resistance patterns:
- Pattern 1: Carbapenem resistance + ≥2 other resistances
- Pattern 2: Resistance to ≥3 different antibiotics  
- Pattern 3: Carbapenem + fluoroquinolone resistance

### Features Used
1. **MIC values**: meropenem, ciprofloxacin, colistin
2. **Resistance status**: Binary resistance indicators
3. **Species and region**: Geographic/epidemiological factors
4. **Multiple resistance scores**: Composite resistance metrics
5. **MIC ratios**: Comparative susceptibility patterns
6. **Complex resistance patterns**: Multi-drug resistance combinations

### Performance Metrics
- **AUC**: 1.000 (all models)
- **Precision**: 1.000
- **Recall**: 1.000
- **F1-score**: 1.000

### Top Predictive Features
1. `carbapenem_fluoroquinolone_combo` (importance: 1.843)
2. `meropenem_mic_resistant` (importance: 1.391)
3. `carbapenem_resistance` (importance: 1.391)
4. `resistance_pattern_score` (importance: 1.107)
5. `other_resistance_count` (importance: 1.061)

### Limitations
- **Critical limitation**: No real treatment failure data available
- Target based on theoretical resistance patterns
- High performance likely reflects simplified target definition
- **Not recommended for clinical use** without validation

---

## 3. Cefiderocol Use Prediction Model (Step 4)

### Purpose
Predicts when cefiderocol should be used based on susceptibility patterns.

### Model Architecture
- **Best Model**: Random Forest
- **Algorithms Tested**: Random Forest, XGBoost, Logistic Regression
- **Cross-validation**: StratifiedKFold

### Target Definition
Use cefiderocol when:
- Cefiderocol is susceptible (MIC < 4 mg/L)
- AND resistance to other antibiotics is present

### Features Used
1. **Cefiderocol MIC**: Direct susceptibility measure
2. **Other antibiotic MICs**: meropenem, ciprofloxacin, colistin
3. **Resistance patterns**: Binary resistance indicators
4. **MIC ratios**: Comparative susceptibility analysis
5. **Log-transformed MICs**: Normalized MIC values
6. **Total resistance count**: Multi-drug resistance score

### Performance Metrics
- **AUC**: 1.000
- **Precision**: 1.000
- **Recall**: 1.000
- **Sensitivity**: 1.000
- **Specificity**: 1.000

### Top Predictive Features
1. `cefiderocol_only_susceptible` (importance: 0.267)
2. `total_resistance` (importance: 0.239)
3. `CIPROFLOXACIN_mic` (importance: 0.091)
4. `colistin_mic_resistant` (importance: 0.063)
5. `log_CIPROFLOXACIN_mic` (importance: 0.062)

### Clinical Decision Rules
1. **MIC Threshold**: Use if MIC < 4 mg/L
2. **Resistance Pattern**: Use if susceptible + other antibiotics resistant
3. **Comparative Analysis**: Use if lower MIC than alternatives
4. **Epidemiological Factors**: Consider regional patterns

---

## 4. Data Sources and Preprocessing

### Primary Data Sources
1. **ATLAS Database** (2.xlsx): Global antimicrobial susceptibility data
2. **SIDERO-WT Database** (1.xlsx): Cefiderocol-specific susceptibility data

### Data Cleaning Process
1. **MIC Value Cleaning**: Remove non-numeric characters, convert to float
2. **Missing Value Handling**: Drop rows with missing MIC values
3. **Resistance Thresholds**: Apply clinical breakpoints
4. **Feature Engineering**: Create composite resistance scores

### Resistance Breakpoints Used
- **Cefiderocol**: 4 mg/L
- **Meropenem**: 8 mg/L  
- **Ciprofloxacin**: 1 mg/L
- **Colistin**: 4 mg/L

---

## 5. Model Validation and Limitations

### Validation Approach
- **Train/Test Split**: 80/20 random split
- **Cross-validation**: StratifiedKFold (5-fold)
- **Performance Metrics**: AUC, precision, recall, F1-score

### Critical Limitations
1. **No Real Treatment Failure Data**: Models based on theoretical patterns
2. **Simplified Target Definitions**: May not reflect clinical reality
3. **Potential Data Leakage**: Features directly related to target definition
4. **Geographic Bias**: Data may not represent global patterns
5. **Temporal Bias**: Historical data may not reflect current resistance

### Recommendations for Clinical Use
1. **Do NOT use for clinical decisions** without validation
2. **Validate on real treatment failure data**
3. **Include clinical factors** (comorbidities, previous exposure)
4. **Prospective validation** required before implementation
5. **Use as research tool** only

---

## 6. Clinical Applications and Decision Support

### Current Capabilities
1. **Antibiotic Sequence Recommendation**: Optimal order based on efficacy
2. **Resistance Pattern Analysis**: Multi-drug resistance identification
3. **Cefiderocol Use Guidance**: When to consider cefiderocol
4. **Risk Stratification**: Treatment failure risk assessment

### Implementation Considerations
1. **Clinical Validation Required**: Before any clinical use
2. **Integration with Clinical Systems**: Decision support integration
3. **Education and Training**: Clinician training on model interpretation
4. **Monitoring and Updates**: Regular model performance monitoring

### Future Directions
1. **Real Treatment Failure Data**: Collect actual clinical outcomes
2. **Additional Clinical Variables**: Include patient factors
3. **Species-Specific Models**: Develop targeted prediction models
4. **Genomic Integration**: Include resistance gene data
5. **Prospective Clinical Trials**: Validate in real clinical settings

---

## 7. Technical Implementation

### Model Files
- `antibiotic_decision_tree.py`: Main decision tree model
- `cefiderocol_treatment_failure_model_final.py`: Treatment failure prediction
- `step4_prediction.py`: Cefiderocol use prediction
- `generate_pdf_report.py`: Report generation

### Dependencies
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting
- **SHAP**: Feature importance analysis
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

### Output Files
- **Reports**: PDF and Markdown reports
- **Visualizations**: Performance plots and feature importance
- **Model Artifacts**: Trained models and encoders

---

## Summary

The Vivli system contains sophisticated prediction models for antibiotic decision-making, but they have critical limitations that prevent clinical use without validation. The models demonstrate excellent performance metrics, but these likely reflect simplified target definitions rather than real predictive ability.

**Key Recommendation**: These models should be used for research and development purposes only, with comprehensive clinical validation required before any clinical implementation.
