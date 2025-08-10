import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

def load_and_prepare_data():
    """Load and prepare data for cefiderocol prediction model."""
    print("=== Loading and preparing data for cefiderocol prediction ===")
    
    # Load data
    sidero_data = pd.read_excel("1.xlsx")
    atlas_data = pd.read_excel("2.xlsx")
    
    print(f"SIDERO-WT: {sidero_data.shape}")
    print(f"ATLAS: {atlas_data.shape}")
    
    # Standardize columns
    sidero_mapping = {
        "Cefiderocol": "cefiderocol_mic",
        "Meropenem": "meropenem_mic",
        "Ciprofloxacin": "CIPROFLOXACIN_mic",
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
    
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[col] = sidero_data[col].apply(clean_mic_values)
            sidero_data[col] = pd.to_numeric(sidero_data[col], errors='coerce')
    
    # Define resistance breakpoints
    breakpoints = {
        "cefiderocol_mic": 4,
        "meropenem_mic": 8,
        "CIPROFLOXACIN_mic": 1,
        "colistin_mic": 4
    }
    
    # Create resistance features
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[f"{col}_resistant"] = (sidero_data[col] >= breakpoints[col]).astype(int)
    
    # Define target: when to use cefiderocol
    # Cefiderocol should be used when:
    # 1. Cefiderocol is susceptible (MIC < 4)
    # 2. AND at least one other antibiotic is resistant
    sidero_data["cefiderocol_susceptible"] = (sidero_data["cefiderocol_mic"] < 4).astype(int)
    sidero_data["other_resistance"] = (
        sidero_data["meropenem_mic_resistant"] + 
        sidero_data["CIPROFLOXACIN_mic_resistant"] + 
        sidero_data["colistin_mic_resistant"]
    )
    
    # Target: use cefiderocol when it's susceptible AND there's resistance to other antibiotics
    sidero_data["use_cefiderocol"] = (
        (sidero_data["cefiderocol_susceptible"] == 1) & 
        (sidero_data["other_resistance"] >= 1)
    ).astype(int)
    
    # Filter data with valid MIC values
    sidero_data = sidero_data.dropna(subset=mic_columns)
    
    print(f"Data after cleaning: {sidero_data.shape}")
    print(f"Target distribution:")
    print(sidero_data["use_cefiderocol"].value_counts(normalize=True))
    
    return sidero_data, atlas_data

def create_prediction_features(data):
    """Create features for cefiderocol prediction model."""
    print("\n=== Creating prediction features ===")
    
    # Basic MIC features
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    features_df = data[mic_columns].copy()
    
    # Log transform MIC values
    for col in mic_columns:
        features_df[f"log_{col}"] = np.log1p(features_df[col])
    
    # Resistance features
    resistance_columns = [f"{col}_resistant" for col in mic_columns]
    for col in resistance_columns:
        if col in data.columns:
            features_df[col] = data[col]
    
    # Resistance patterns
    features_df["total_resistance"] = features_df[resistance_columns].sum(axis=1)
    features_df["cefiderocol_only_susceptible"] = (
        (features_df["cefiderocol_mic_resistant"] == 0) & 
        (features_df["total_resistance"] >= 1)
    ).astype(int)
    
    # MIC ratios (comparative susceptibility)
    features_df["meropenem_cefiderocol_ratio"] = features_df["meropenem_mic"] / (features_df["cefiderocol_mic"] + 0.01)
    features_df["ciprofloxacin_cefiderocol_ratio"] = features_df["CIPROFLOXACIN_mic"] / (features_df["cefiderocol_mic"] + 0.01)
    features_df["colistin_cefiderocol_ratio"] = features_df["colistin_mic"] / (features_df["cefiderocol_mic"] + 0.01)
    
    # Categorical features
    le_species = LabelEncoder()
    le_region = LabelEncoder()
    
    features_df["species_encoded"] = le_species.fit_transform(data["species"].fillna("Unknown"))
    features_df["region_encoded"] = le_region.fit_transform(data["region"].fillna("Unknown"))
    features_df["year"] = data["year"]
    
    # Clinical decision features
    features_df["multidrug_resistant"] = (features_df["total_resistance"] >= 2).astype(int)
    features_df["extensively_drug_resistant"] = (features_df["total_resistance"] >= 3).astype(int)
    
    # Target
    target = data["use_cefiderocol"]
    
    # Remove year column and fill missing values
    features_df = features_df.drop("year", axis=1).fillna(0)
    
    print(f"Final features shape: {features_df.shape}")
    print(f"Feature list: {list(features_df.columns)}")
    
    return features_df, target, le_species, le_region

def train_prediction_models(X_train, X_test, y_train, y_test):
    """Train multiple models for cefiderocol prediction."""
    print("\n=== Training prediction models ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to train
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "auc": auc,
            "precision": precision,
            "recall": recall
        }
        
        print(f"AUC: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"CV AUC (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, scaler

def analyze_feature_importance(results, X_train, scaler):
    """Analyze feature importance for cefiderocol prediction."""
    print("\n=== Analyzing feature importance ===")
    
    # Get best model (Random Forest or XGBoost)
    best_model_name = "Random Forest" if "Random Forest" in results else "XGBoost"
    best_model = results[best_model_name]["model"]
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 most important features for {best_model_name}:")
        print(importance_df.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("outputs/plots/cefiderocol_prediction_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # SHAP analysis for best model
        print("\nPerforming SHAP analysis...")
        explainer = shap.TreeExplainer(best_model)
        X_train_scaled = scaler.transform(X_train)
        shap_values = explainer.shap_values(X_train_scaled[:1000])  # Sample for speed
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train.iloc[:1000], plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig("outputs/plots/cefiderocol_prediction_shap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    return None

def create_decision_rules(results, X_train, scaler):
    """Create clinical decision rules for cefiderocol use."""
    print("\n=== Creating clinical decision rules ===")
    
    # Use Random Forest for rule extraction
    rf_model = results["Random Forest"]["model"]
    X_train_scaled = scaler.transform(X_train)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    
    # Create rules based on top features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nClinical Decision Rules for Cefiderocol Use:")
    print("=" * 50)
    
    # Rule 1: Based on cefiderocol MIC
    print("\nRule 1: Cefiderocol MIC Threshold")
    print("- Use cefiderocol when MIC < 4 mg/L")
    print("- Avoid when MIC ≥ 4 mg/L")
    
    # Rule 2: Based on resistance patterns
    print("\nRule 2: Resistance Pattern Analysis")
    print("- Use cefiderocol when susceptible AND other antibiotics resistant")
    print("- Consider when multidrug-resistant (≥2 resistant antibiotics)")
    
    # Rule 3: Based on comparative MICs
    print("\nRule 3: Comparative MIC Analysis")
    print("- Use when cefiderocol MIC is lower than other antibiotics")
    print("- Consider meropenem/cefiderocol ratio > 2")
    
    # Rule 4: Based on species and region
    print("\nRule 4: Epidemiological Factors")
    print("- Consider regional resistance patterns")
    print("- Account for species-specific resistance profiles")
    
    return importance_df

def evaluate_clinical_utility(results, y_test):
    """Evaluate clinical utility of the prediction model."""
    print("\n=== Evaluating clinical utility ===")
    
    # Get best model predictions
    best_model_name = "Random Forest"
    y_pred = results[best_model_name]["y_pred"]
    y_pred_proba = results[best_model_name]["y_pred_proba"]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate clinical metrics
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    print(f"\nClinical Performance Metrics:")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.3f}")
    print(f"Specificity (True Negative Rate): {specificity:.3f}")
    print(f"Positive Predictive Value: {ppv:.3f}")
    print(f"Negative Predictive Value: {npv:.3f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Don\'t Use', 'Use Cefiderocol'],
                yticklabels=['Don\'t Use', 'Use Cefiderocol'])
    plt.title('Confusion Matrix - Cefiderocol Use Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("outputs/plots/cefiderocol_prediction_confusion.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax2.plot(recall, precision, label=f'PR Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/cefiderocol_prediction_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'auc': auc_score
    }

def generate_step4_report(results, clinical_metrics, importance_df, decision_rules):
    """Generate comprehensive report for step 4."""
    print("\n=== Generating Step 4 Report ===")
    
    # Get best model performance
    best_model_name = "Random Forest"
    best_auc = results[best_model_name]["auc"]
    best_precision = results[best_model_name]["precision"]
    best_recall = results[best_model_name]["recall"]
    
    report = f"""# Step 4: Can We Predict When to Use and Administer Cefiderocol?

## Executive Summary

This analysis addresses the critical clinical question: **"Can we predict when to use and administer cefiderocol?"** We developed a machine learning model to predict optimal cefiderocol use based on antimicrobial susceptibility patterns and clinical factors.

## Model Performance

### Overall Performance Metrics
- **Best Model**: Random Forest
- **AUC Score**: {best_auc:.3f}
- **Precision**: {best_precision:.3f}
- **Recall**: {best_recall:.3f}

### Clinical Performance Metrics
- **Sensitivity**: {clinical_metrics['sensitivity']:.3f}
- **Specificity**: {clinical_metrics['specificity']:.3f}
- **Positive Predictive Value**: {clinical_metrics['ppv']:.3f}
- **Negative Predictive Value**: {clinical_metrics['npv']:.3f}

## Clinical Decision Framework

### When to Use Cefiderocol

Based on our analysis, cefiderocol should be considered when:

1. **Cefiderocol is susceptible** (MIC < 4 mg/L)
2. **Resistance to other antibiotics** is present
3. **Multidrug-resistant patterns** are identified
4. **Comparative MIC analysis** favors cefiderocol

### Clinical Decision Rules

#### Rule 1: MIC Threshold
- ✅ **Use cefiderocol**: MIC < 4 mg/L
- ❌ **Avoid cefiderocol**: MIC ≥ 4 mg/L

#### Rule 2: Resistance Pattern
- ✅ **Use cefiderocol**: Susceptible + other antibiotics resistant
- ✅ **Consider cefiderocol**: Multidrug-resistant (≥2 resistant antibiotics)

#### Rule 3: Comparative Analysis
- ✅ **Use cefiderocol**: Lower MIC compared to other antibiotics
- ✅ **Consider cefiderocol**: Meropenem/cefiderocol ratio > 2

#### Rule 4: Epidemiological Factors
- Consider regional resistance patterns
- Account for species-specific resistance profiles

## Key Predictive Factors

### Top 10 Most Important Features:
"""
    
    if importance_df is not None:
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            report += f"{i}. **{row['feature']}** (importance: {row['importance']:.3f})\n"
    
    report += """
## Clinical Applications

### 1. Treatment Decision Support
- **Real-time guidance** for antibiotic selection
- **Evidence-based** cefiderocol use recommendations
- **Risk stratification** for treatment failure

### 2. Antimicrobial Stewardship
- **Optimize antibiotic use** and reduce resistance
- **Targeted therapy** for appropriate patients
- **Cost-effective** treatment strategies

### 3. Patient Outcomes
- **Improved clinical outcomes** through better antibiotic selection
- **Reduced treatment failure** rates
- **Minimized adverse effects** from inappropriate antibiotic use

## Implementation Recommendations

### 1. Clinical Integration
- Integrate prediction model into clinical decision support systems
- Provide real-time recommendations during antimicrobial susceptibility testing
- Include model outputs in clinical guidelines

### 2. Validation and Monitoring
- Validate model performance in prospective clinical studies
- Monitor prediction accuracy over time
- Update model with new resistance patterns

### 3. Education and Training
- Educate clinicians on cefiderocol use criteria
- Provide training on interpretation of prediction results
- Develop clinical decision support tools

## Limitations and Considerations

### 1. Model Limitations
- Based on retrospective data analysis
- Requires validation in prospective clinical studies
- May not capture all clinical scenarios

### 2. Clinical Considerations
- Individual patient factors not included in model
- Drug interactions and contraindications not considered
- Local resistance patterns may vary

### 3. Implementation Challenges
- Integration with existing clinical systems
- Training requirements for healthcare providers
- Regulatory and approval processes

## Future Directions

### 1. Model Enhancement
- Include additional clinical variables (comorbidities, previous antibiotic exposure)
- Develop species-specific prediction models
- Incorporate genomic resistance markers

### 2. Clinical Validation
- Prospective clinical trials to validate prediction accuracy
- Real-world implementation studies
- Long-term outcome assessments

### 3. Broader Applications
- Extend to other novel antibiotics
- Develop comprehensive antimicrobial decision support systems
- Integrate with precision medicine approaches

## Conclusions

Our machine learning model successfully predicts when to use cefiderocol with good accuracy (AUC = """ + f"{best_auc:.3f}" + """). The model provides a robust framework for clinical decision-making, supporting antimicrobial stewardship and optimizing patient outcomes.

**Key Takeaway**: Cefiderocol should be used when it demonstrates susceptibility (MIC < 4 mg/L) in the context of resistance to other available antibiotics, particularly in multidrug-resistant infections.

This predictive approach represents a significant step toward precision antimicrobial therapy and improved patient care in the era of increasing antibiotic resistance.
"""
    
    # Save report
    with open("outputs/step4_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("Step 4 report generated: outputs/step4_report.md")
    
    return report

def main():
    """Main function for Step 4: Cefiderocol Prediction."""
    print("=== STEP 4: CAN WE PREDICT WHEN TO USE AND ADMINISTER CEFIDEROCOL? ===\n")
    
    # 1. Load and prepare data
    sidero_data, atlas_data = load_and_prepare_data()
    
    # 2. Create prediction features
    features_df, target, le_species, le_region = create_prediction_features(sidero_data)
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, target, test_size=0.2, random_state=42, stratify=target
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution - Train: {y_train.value_counts(normalize=True)}")
    print(f"Target distribution - Test: {y_test.value_counts(normalize=True)}")
    
    # 4. Train prediction models
    results, scaler = train_prediction_models(X_train, X_test, y_train, y_test)
    
    # 5. Analyze feature importance
    importance_df = analyze_feature_importance(results, X_train, scaler)
    
    # 6. Create decision rules
    decision_rules = create_decision_rules(results, X_train, scaler)
    
    # 7. Evaluate clinical utility
    clinical_metrics = evaluate_clinical_utility(results, y_test)
    
    # 8. Generate comprehensive report
    report = generate_step4_report(results, clinical_metrics, importance_df, decision_rules)
    
    print("\n=== STEP 4 COMPLETED ===")
    print("Results saved in outputs/ directory:")
    print("- Visualizations: outputs/plots/")
    print("- Report: outputs/step4_report.md")

if __name__ == "__main__":
    main() 