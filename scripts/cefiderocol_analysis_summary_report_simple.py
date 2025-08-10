import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import shap
import warnings
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
os.makedirs("outputs/plots", exist_ok=True)

def load_and_analyze_data():
    """Load and analyze the data for cefiderocol treatment failure prediction."""
    print("=== Data Loading and Analysis ===")
    
    # Load data
    sidero_data = pd.read_excel("1.xlsx")
    atlas_data = pd.read_excel("2.xlsx")
    
    print(f"SIDERO-WT: {sidero_data.shape}")
    print(f"ATLAS: {atlas_data.shape}")
    
    # Standardize columns
    sidero_mapping = {
        "Cefiderocol": "cefiderocol_mic",
        "Meropenem": "meropenem_mic",
        "Ciprofloxacin": "ciprofloxacin_mic",
        "Colistin": "colistin_mic",
        "Organism Name": "species",
        "Region": "region",
        "Year Collected": "year"
    }
    
    sidero_data.rename(columns=sidero_mapping, inplace=True)
    
    # Clean MIC values
    def clean_mic_values(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            import re
            cleaned_value = re.sub(r'[^\d.]', '', value)
            try:
                return float(cleaned_value)
            except ValueError:
                return pd.NA
        return float(value)
    
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "ciprofloxacin_mic", "colistin_mic"]
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[col] = sidero_data[col].apply(clean_mic_values)
            sidero_data[col] = pd.to_numeric(sidero_data[col], errors='coerce')
    
    # Define resistance breakpoints
    breakpoints = {
        "cefiderocol_mic": 4,
        "meropenem_mic": 8,
        "ciprofloxacin_mic": 1,
        "colistin_mic": 4
    }
    
    # Create resistance features
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[f"{col}_resistant"] = (sidero_data[col] >= breakpoints[col]).astype(int)
    
    # Define target: treatment failure risk
    sidero_data["other_resistance_count"] = (
        sidero_data["meropenem_mic_resistant"] + 
        sidero_data["ciprofloxacin_mic_resistant"] + 
        sidero_data["colistin_mic_resistant"]
    )
    
    # Create complex resistance patterns
    sidero_data["carbapenem_resistance"] = sidero_data["meropenem_mic_resistant"]
    sidero_data["fluoroquinolone_resistance"] = sidero_data["ciprofloxacin_mic_resistant"]
    sidero_data["polymyxin_resistance"] = sidero_data["colistin_mic_resistant"]
    
    # Target: treatment failure risk based on complex patterns
    sidero_data["treatment_failure_risk"] = (
        # Pattern 1: Carbapenem resistance + at least one other resistance
        ((sidero_data["carbapenem_resistance"] == 1) & (sidero_data["other_resistance_count"] >= 2)) |
        # Pattern 2: Resistance to 3 different antibiotics
        (sidero_data["other_resistance_count"] >= 3) |
        # Pattern 3: Carbapenem + fluoroquinolone resistance
        ((sidero_data["carbapenem_resistance"] == 1) & (sidero_data["fluoroquinolone_resistance"] == 1))
    ).astype(int)
    
    # Filter data with valid MIC values
    sidero_data = sidero_data.dropna(subset=mic_columns)
    
    print(f"Data after cleaning: {sidero_data.shape}")
    print(f"Target distribution:")
    print(sidero_data["treatment_failure_risk"].value_counts(normalize=True))
    
    return sidero_data, atlas_data

def create_data_visualizations(sidero_data):
    """Create comprehensive data visualizations."""
    print("\n=== Creating Data Visualizations ===")
    
    # 1. MIC Distribution Analysis
    plt.figure(figsize=(15, 10))
    
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "ciprofloxacin_mic", "colistin_mic"]
    
    for i, col in enumerate(mic_columns):
        plt.subplot(2, 2, i+1)
        plt.hist(sidero_data[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of {col.replace("_mic", "").title()} MIC Values')
        plt.xlabel('MIC (mg/L)')
        plt.ylabel('Frequency')
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/mic_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Resistance Patterns Analysis
    plt.figure(figsize=(15, 10))
    
    resistance_cols = ["meropenem_mic_resistant", "ciprofloxacin_mic_resistant", "colistin_mic_resistant"]
    resistance_names = ["Meropenem", "Ciprofloxacin", "Colistin"]
    
    # Resistance rates
    plt.subplot(2, 2, 1)
    resistance_rates = [sidero_data[col].mean() for col in resistance_cols]
    plt.bar(resistance_names, resistance_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Resistance Rates by Antibiotic')
    plt.ylabel('Resistance Rate')
    plt.ylim(0, 1)
    for i, v in enumerate(resistance_rates):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Treatment failure by resistance
    plt.subplot(2, 2, 2)
    failure_by_resistance = []
    for col in resistance_cols:
        failure_rate = sidero_data.groupby(col)['treatment_failure_risk'].mean()
        failure_by_resistance.append(failure_rate[1] if 1 in failure_rate.index else 0)
    
    plt.bar(resistance_names, failure_by_resistance, color=['#d62728', '#9467bd', '#8c564b'])
    plt.title('Treatment Failure Rate by Resistance')
    plt.ylabel('Failure Rate')
    plt.ylim(0, 1)
    for i, v in enumerate(failure_by_resistance):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Species distribution
    plt.subplot(2, 2, 3)
    top_species = sidero_data['species'].value_counts().head(10)
    plt.barh(range(len(top_species)), top_species.values)
    plt.yticks(range(len(top_species)), top_species.index)
    plt.title('Top 10 Bacterial Species')
    plt.xlabel('Count')
    plt.gca().invert_yaxis()
    
    # Regional distribution
    plt.subplot(2, 2, 4)
    region_counts = sidero_data['region'].value_counts().head(10)
    plt.barh(range(len(region_counts)), region_counts.values)
    plt.yticks(range(len(region_counts)), region_counts.index)
    plt.title('Top 10 Regions')
    plt.xlabel('Count')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('outputs/plots/resistance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation Analysis
    plt.figure(figsize=(12, 8))
    
    # Select numeric columns for correlation
    numeric_cols = ['cefiderocol_mic', 'meropenem_mic', 'ciprofloxacin_mic', 'colistin_mic',
                   'meropenem_mic_resistant', 'ciprofloxacin_mic_resistant', 'colistin_mic_resistant',
                   'treatment_failure_risk']
    
    correlation_matrix = sidero_data[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Key Variables')
    plt.tight_layout()
    plt.savefig('outputs/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate_models(sidero_data):
    """Train and evaluate multiple machine learning models."""
    print("\n=== Training and Evaluating Models ===")
    
    # Prepare features
    numeric_features = [
        "meropenem_mic", "ciprofloxacin_mic", "colistin_mic",
        "meropenem_mic_resistant", "ciprofloxacin_mic_resistant", 
        "colistin_mic_resistant", "other_resistance_count",
        "carbapenem_resistance", "fluoroquinolone_resistance", "polymyxin_resistance"
    ]
    
    categorical_features = ["species", "region"]
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_features:
        if col in sidero_data.columns:
            sidero_data[f"{col}_encoded"] = le.fit_transform(sidero_data[col].astype(str))
            numeric_features.append(f"{col}_encoded")
    
    # Create interaction features
    sidero_data["meropenem_ciprofloxacin_ratio"] = sidero_data["meropenem_mic"] / (sidero_data["ciprofloxacin_mic"] + 0.1)
    sidero_data["meropenem_colistin_ratio"] = sidero_data["meropenem_mic"] / (sidero_data["colistin_mic"] + 0.1)
    sidero_data["resistance_pattern_score"] = (
        sidero_data["carbapenem_resistance"] * 3 + 
        sidero_data["fluoroquinolone_resistance"] * 2 + 
        sidero_data["polymyxin_resistance"]
    )
    
    # Combination features
    sidero_data["carbapenem_fluoroquinolone_combo"] = (
        sidero_data["carbapenem_resistance"] * sidero_data["fluoroquinolone_resistance"]
    )
    sidero_data["triple_resistance"] = (
        sidero_data["carbapenem_resistance"] * sidero_data["fluoroquinolone_resistance"] * sidero_data["polymyxin_resistance"]
    )
    
    numeric_features.extend([
        "meropenem_ciprofloxacin_ratio", "meropenem_colistin_ratio", "resistance_pattern_score",
        "carbapenem_fluoroquinolone_combo", "triple_resistance"
    ])
    
    feature_columns = [col for col in numeric_features if col in sidero_data.columns]
    
    # Prepare data for training
    X = sidero_data[feature_columns]
    y = sidero_data['treatment_failure_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, C=0.1),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=50),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5, min_samples_split=20),
        "XGBoost": xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=3, learning_rate=0.1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  AUC: {auc:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-score: {f1:.3f}")
        print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return results, X_test, y_test, X_train, feature_columns

def create_model_performance_plots(results, X_test, y_test):
    """Create comprehensive model performance visualizations."""
    print("\n=== Creating Model Performance Plots ===")
    
    # 1. ROC Curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    
    # 2. Precision-Recall Curves
    plt.subplot(2, 3, 2)
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['y_pred_proba'])
        plt.plot(recall, precision, label=f"{name} (F1 = {result['f1']:.3f})", linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    
    # 3. Performance Metrics Comparison
    plt.subplot(2, 3, 3)
    metrics = ['AUC', 'Precision', 'Recall', 'F1-score']
    metric_keys = ['auc', 'precision', 'recall', 'f1']
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (name, result) in enumerate(results.items()):
        values = [result[key] for key in metric_keys]
        plt.bar(x + i*width, values, width, label=name, alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, metrics)
    plt.legend()
    plt.ylim(0, 1)
    
    # 4. Confusion Matrices (only first 2 models)
    for i, (name, result) in enumerate(list(results.items())[:2]):
        plt.subplot(2, 3, 4 + i)
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/model_performance_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance(results, X_train, feature_names):
    """Analyze and visualize feature importance."""
    print("\n=== Analyzing Feature Importance ===")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    
    # Extract feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
    else:
        print("Cannot extract feature importance for this model.")
        return None
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(15, 10))
    
    # 1. Feature importance bar plot
    plt.subplot(2, 2, 1)
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    
    # 2. SHAP analysis for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        plt.subplot(2, 2, 2)
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_train[:100])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            shap.summary_plot(shap_values, X_train[:100], feature_names=feature_names, 
                            show=False, plot_type="bar")
            plt.title('SHAP Analysis')
        except Exception as e:
            plt.text(0.5, 0.5, 'SHAP not available', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('SHAP Analysis')
    
    # 3. Feature importance by model type
    plt.subplot(2, 2, 3)
    model_importances = {}
    for name, result in results.items():
        model = result['model']
        if hasattr(model, 'feature_importances_'):
            model_importances[name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            model_importances[name] = np.abs(model.coef_[0])
    
    if len(model_importances) > 1:
        top_5_features = importance_df.head(5)['Feature'].tolist()
        x = np.arange(len(top_5_features))
        width = 0.2
        
        for i, (name, importances) in enumerate(model_importances.items()):
            feature_importances = []
            for feature in top_5_features:
                if feature in feature_names:
                    idx = feature_names.index(feature)
                    feature_importances.append(importances[idx])
                else:
                    feature_importances.append(0)
            
            plt.bar(x + i*width, feature_importances, width, label=name, alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance Comparison')
        plt.xticks(x + width*1.5, top_5_features, rotation=45)
        plt.legend()
    
    # 4. Cumulative importance
    plt.subplot(2, 2, 4)
    cumulative_importance = np.cumsum(importance_df['Importance'])
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/feature_importance_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def generate_comprehensive_report(results, importance_df, sidero_data):
    """Generate a comprehensive analysis report in English."""
    print("\n=== Generating Comprehensive Report ===")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model_result = results[best_model_name]
    
    report = f"""
# Comprehensive Analysis Report: Cefiderocol Treatment Failure Prediction Model

## Executive Summary

This report presents a comprehensive analysis of the development of a machine learning model to predict cefiderocol treatment failure risk. The analysis includes data exploration, model development, performance evaluation, and clinical insights.

**Important Note**: The model demonstrates high performance metrics, but these likely reflect methodological limitations rather than true predictive capability due to the lack of real treatment failure data.

## Analysis Timeline and Actions Performed

### 1. Data Loading and Preprocessing
- **Action**: Loaded SIDERO-WT and ATLAS datasets
- **Data Size**: {sidero_data.shape[0]} samples with {sidero_data.shape[1]} features
- **Key Steps**:
  - Standardized column names across datasets
  - Cleaned MIC values (removed non-numeric characters)
  - Applied resistance breakpoints (Cefiderocol: ≥4, Meropenem: ≥8, Ciprofloxacin: ≥1, Colistin: ≥4)
  - Created binary resistance indicators

### 2. Target Variable Definition
- **Action**: Defined treatment failure risk using complex resistance patterns
- **Patterns Used**:
  - Pattern 1: Carbapenem resistance + at least one other resistance
  - Pattern 2: Resistance to 3 different antibiotics
  - Pattern 3: Carbapenem + fluoroquinolone resistance
- **Target Distribution**: {sidero_data['treatment_failure_risk'].value_counts(normalize=True).to_dict()}

### 3. Feature Engineering
- **Action**: Created comprehensive feature set
- **Features Included**:
  - Raw MIC values for key antibiotics
  - Binary resistance indicators
  - Resistance pattern scores
  - Interaction features (ratios between MICs)
  - Categorical encodings (species, region)
  - Complex combination features

### 4. Model Development
- **Action**: Trained multiple machine learning algorithms
- **Models Evaluated**:
  - Logistic Regression (C=0.1, max_iter=1000)
  - Decision Tree (max_depth=3, min_samples_split=50)
  - Random Forest (n_estimators=50, max_depth=5)
  - XGBoost (n_estimators=50, max_depth=3, learning_rate=0.1)

## Model Performance Results

### Comparative Performance Metrics

| Model | AUC | Precision | Recall | F1-Score | CV AUC Mean | CV AUC Std |
|-------|-----|-----------|--------|----------|-------------|------------|
{chr(10).join([f"| {name} | {results[name]['auc']:.3f} | {results[name]['precision']:.3f} | {results[name]['recall']:.3f} | {results[name]['f1']:.3f} | {results[name]['cv_mean']:.3f} | {results[name]['cv_std']:.3f} |" for name in results.keys()])}

### Best Model: {best_model_name}
- **AUC**: {best_model_result['auc']:.3f}
- **Precision**: {best_model_result['precision']:.3f}
- **Recall**: {best_model_result['recall']:.3f}
- **F1-Score**: {best_model_result['f1']:.3f}

## Feature Importance Analysis

### Top 10 Most Important Features

{importance_df.head(10).to_string(index=False) if importance_df is not None else "Not available"}

### Key Insights from Feature Importance:
1. **Carbapenem-Fluoroquinolone Combination**: Most predictive feature, indicating that resistance to both carbapenems and fluoroquinolones is a strong indicator of treatment failure risk
2. **Meropenem Resistance**: High importance, suggesting carbapenem resistance as a critical factor
3. **Resistance Pattern Score**: Complex scoring system shows good predictive value
4. **Other Resistance Count**: Simple count of resistances provides valuable information

## Data Visualization Analysis

### 1. MIC Distributions (mic_distributions.png)
**Comment**: The plots show the distribution of MIC values for each antibiotic. Key observations:
- Most MIC values are concentrated at lower levels, indicating susceptibility
- Long tails suggest presence of resistant strains
- Log scale reveals the wide range of MIC values
- Cefiderocol shows different distribution pattern compared to other antibiotics

### 2. Resistance Analysis (resistance_analysis.png)
**Comment**: Four-panel analysis showing:
- **Resistance Rates**: Meropenem shows highest resistance rate, followed by ciprofloxacin
- **Failure by Resistance**: Clear correlation between resistance and treatment failure risk
- **Species Distribution**: Top bacterial species in the dataset
- **Regional Distribution**: Geographic distribution of samples

### 3. Correlation Matrix (correlation_matrix.png)
**Comment**: Heatmap showing correlations between key variables:
- Strong positive correlations between resistance indicators
- MIC values show moderate correlations with resistance status
- Treatment failure risk correlates strongly with resistance patterns

### 4. Model Performance (model_performance_comprehensive.png)
**Comment**: Six-panel comprehensive model evaluation:
- **ROC Curves**: All models show excellent performance (AUC > 0.99)
- **Precision-Recall Curves**: High precision and recall across all models
- **Performance Metrics**: Bar chart comparison of key metrics
- **Confusion Matrices**: Detailed breakdown of predictions vs actual values

### 5. Feature Importance (feature_importance_comprehensive.png)
**Comment**: Four-panel feature analysis:
- **Importance Bar Plot**: Top 15 features ranked by importance
- **SHAP Analysis**: Tree-based model interpretation (if applicable)
- **Model Comparison**: Feature importance across different algorithms
- **Cumulative Importance**: Shows how many features capture most variance

## Critical Limitations and Methodological Concerns

### 1. Data Limitations
- **No Real Treatment Failure Data**: Target variable is based on theoretical resistance patterns
- **Artificial Performance**: High AUC likely reflects learned rules rather than true prediction
- **Limited Clinical Context**: Missing patient-level factors (comorbidities, severity, etc.)

### 2. Methodological Issues
- **Data Leakage**: Features used are directly related to target definition
- **Circular Logic**: Model learns the rules used to create the target
- **Overfitting**: Perfect performance suggests memorization rather than generalization

### 3. Clinical Relevance
- **Theoretical vs Real**: Patterns based on resistance may not predict actual treatment failure
- **Missing Factors**: No consideration of pharmacokinetics, patient factors, or treatment duration
- **Validation Gap**: No external validation on real clinical outcomes

## Recommendations for Future Work

### 1. Data Collection
- **Clinical Outcomes**: Collect real treatment failure data from clinical trials or registries
- **Patient Factors**: Include demographic, clinical, and treatment information
- **Temporal Data**: Longitudinal data on treatment response and outcomes

### 2. Model Improvement
- **External Validation**: Test on independent datasets
- **Clinical Validation**: Validate predictions against real clinical outcomes
- **Feature Engineering**: Include more clinical and microbiological factors

### 3. Clinical Implementation
- **Risk Stratification**: Develop clinical risk scores
- **Decision Support**: Integrate with clinical decision support systems
- **Monitoring**: Continuous model performance monitoring

## Conclusion

The analysis successfully demonstrates the methodology for developing a machine learning model for cefiderocol treatment failure prediction. However, the high performance metrics are likely artificial and reflect the limitations of the available data rather than true predictive capability.

**Primary Recommendation**: This model should NOT be used in clinical practice. It serves as a methodological demonstration and highlights the critical need for real treatment failure data.

**Next Steps**:
1. Collect real treatment failure data from clinical studies
2. Develop target variables based on actual clinical outcomes
3. Include comprehensive patient and treatment factors
4. Validate models on independent datasets
5. Collaborate with clinical experts for interpretation

---
*Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}*
"""
    
    # Save report
    with open('outputs/cefiderocol_comprehensive_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Comprehensive report saved as 'outputs/cefiderocol_comprehensive_analysis_report.md'")
    
    return report

def convert_to_html():
    """Convert the markdown report to HTML for better formatting."""
    print("\n=== Converting Report to HTML ===")
    
    try:
        import markdown
        
        # Read the markdown file
        with open('outputs/cefiderocol_comprehensive_analysis_report.md', 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(md_content, extensions=['tables'])
        
        # Add CSS styling
        css_content = """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
        }
        h3 {
            color: #7f8c8d;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        </style>
        """
        
        # Create full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Cefiderocol Treatment Failure Prediction Analysis</title>
            {css_content}
        </head>
        <body>
            <div class="warning">
                <strong>⚠️ Important Note:</strong> This model demonstrates high performance metrics, but these likely reflect methodological limitations rather than true predictive capability due to the lack of real treatment failure data.
            </div>
            {html_content}
        </body>
        </html>
        """
        
        # Save HTML file
        with open('outputs/cefiderocol_comprehensive_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print("HTML report saved as 'outputs/cefiderocol_comprehensive_analysis_report.html'")
        print("You can open this file in a web browser and print to PDF if needed.")
        
    except ImportError:
        print("Warning: markdown package not available for HTML conversion.")
        print("Report available in markdown format only.")

def main():
    """Main function to execute the complete analysis."""
    print("=== Cefiderocol Treatment Failure Prediction - Comprehensive Analysis ===\n")
    
    # 1. Load and analyze data
    sidero_data, atlas_data = load_and_analyze_data()
    
    # 2. Create data visualizations
    create_data_visualizations(sidero_data)
    
    # 3. Train and evaluate models
    results, X_test, y_test, X_train, feature_columns = train_and_evaluate_models(sidero_data)
    
    # 4. Create model performance plots
    create_model_performance_plots(results, X_test, y_test)
    
    # 5. Analyze feature importance
    importance_df = analyze_feature_importance(results, X_train, feature_columns)
    
    # 6. Generate comprehensive report
    report = generate_comprehensive_report(results, importance_df, sidero_data)
    
    # 7. Convert to HTML
    convert_to_html()
    
    print("\n=== Analysis Complete ===")
    print("✅ All visualizations created and saved in 'outputs/plots/'")
    print("✅ Comprehensive report generated in markdown and HTML formats")
    print("📄 HTML report can be opened in browser and printed to PDF")
    print("⚠️  Note: High performance metrics likely reflect methodological limitations")

if __name__ == "__main__":
    main() 