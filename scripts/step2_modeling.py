import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration des paramètres d'affichage
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Création des dossiers pour les sorties
import os
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

def load_and_prepare_data():
    """Charge et prépare les données pour la modélisation."""
    print("=== Chargement des données ===")
    
    # Charger les données
    sidero_data = pd.read_excel("1.xlsx")
    atlas_data = pd.read_excel("2.xlsx")
    
    print(f"SIDERO-WT: {sidero_data.shape}")
    print(f"ATLAS: {atlas_data.shape}")
    
    # Standardiser les colonnes (réutiliser la logique de describe.py)
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
    
    # Nettoyer les valeurs MIC
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
    
    # Appliquer le nettoyage aux colonnes MIC
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[col] = sidero_data[col].apply(clean_mic_values)
            sidero_data[col] = pd.to_numeric(sidero_data[col], errors='coerce')
    
    # Définir la cible binaire
    sidero_data["cefiderocol_resistant"] = sidero_data["cefiderocol_mic"] >= 4
    
    # Filtrer les données avec des valeurs MIC valides
    sidero_data = sidero_data.dropna(subset=["cefiderocol_mic"])
    
    print(f"Données après nettoyage: {sidero_data.shape}")
    print(f"Distribution de la résistance: {sidero_data['cefiderocol_resistant'].value_counts(normalize=True)}")
    
    return sidero_data

def feature_selection(data):
    """Sélectionne les caractéristiques pour la prédiction."""
    print("\n=== Sélection des caractéristiques ===")
    
    # Caractéristiques numériques disponibles
    numeric_features = ["meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    
    # Caractéristiques catégorielles
    categorical_features = ["species", "region"]
    
    # Créer des caractéristiques d'ingénierie
    features_df = data[numeric_features].copy()
    
    # Encoder les variables catégorielles
    le_species = LabelEncoder()
    le_region = LabelEncoder()
    
    features_df["species_encoded"] = le_species.fit_transform(data["species"].fillna("Unknown"))
    features_df["region_encoded"] = le_region.fit_transform(data["region"].fillna("Unknown"))
    
    # Ajouter des caractéristiques dérivées
    features_df["year"] = data["year"]
    features_df["log_meropenem"] = np.log1p(features_df["meropenem_mic"].fillna(0))
    features_df["log_ciprofloxacin"] = np.log1p(features_df["CIPROFLOXACIN_mic"].fillna(0))
    features_df["log_colistin"] = np.log1p(features_df["colistin_mic"].fillna(0))
    
    # Créer des interactions
    features_df["meropenem_ciprofloxacin"] = features_df["meropenem_mic"].fillna(0) * features_df["CIPROFLOXACIN_mic"].fillna(0)
    features_df["meropenem_colistin"] = features_df["meropenem_mic"].fillna(0) * features_df["colistin_mic"].fillna(0)
    
    # Cible
    target = data["cefiderocol_resistant"]
    
    # Supprimer les lignes avec des valeurs manquantes
    features_df = features_df.fillna(0)
    
    print(f"Caractéristiques finales: {features_df.shape}")
    print(f"Liste des caractéristiques: {list(features_df.columns)}")
    
    return features_df, target, le_species, le_region

def split_data_by_year(features_df, target):
    """Divise les données par année (2014-2018 pour l'entraînement, 2019 pour le test)."""
    print("\n=== Division des données par année ===")
    
    # Diviser par année
    train_mask = features_df["year"] <= 2018
    test_mask = features_df["year"] == 2019
    
    X_train = features_df[train_mask].drop("year", axis=1)
    X_test = features_df[test_mask].drop("year", axis=1)
    y_train = target[train_mask]
    y_test = target[test_mask]
    
    print(f"Ensemble d'entraînement (2014-2018): {X_train.shape}")
    print(f"Ensemble de test (2019): {X_test.shape}")
    print(f"Distribution de la résistance - Entraînement: {y_train.value_counts(normalize=True)}")
    print(f"Distribution de la résistance - Test: {y_test.value_counts(normalize=True)}")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    """Entraîne et évalue plusieurs modèles de classification."""
    print("\n=== Entraînement des modèles ===")
    
    # Standardiser les caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèles à entraîner
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Entraînement {name} ---")
        
        # Entraîner le modèle
        if name == "XGBoost":
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train_scaled, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Métriques
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
        print(f"Précision: {precision:.4f}")
        print(f"Rappel: {recall:.4f}")
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"CV AUC (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, scaler

def evaluate_models(results, y_test):
    """Évalue et compare les modèles."""
    print("\n=== Évaluation des modèles ===")
    
    # Comparaison des métriques
    comparison_df = pd.DataFrame({
        "AUC": [results[name]["auc"] for name in results.keys()],
        "Précision": [results[name]["precision"] for name in results.keys()],
        "Rappel": [results[name]["recall"] for name in results.keys()]
    }, index=results.keys())
    
    print("Comparaison des modèles:")
    print(comparison_df)
    
    # Courbes ROC
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_pred_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC')
    plt.legend()
    plt.grid(True)
    
    # Courbes Precision-Recall
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result["y_pred_proba"])
        plt.plot(recall, precision, label=f"{name}")
    
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title('Courbes Precision-Recall')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/model_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df

def feature_importance_analysis(results, X_train, scaler):
    """Analyse l'importance des caractéristiques."""
    print("\n=== Analyse de l'importance des caractéristiques ===")
    
    feature_names = X_train.columns
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        
        if name == "Logistic Regression":
            # Coefficients de la régression logistique
            importance = np.abs(result["model"].coef_[0])
        elif name == "Random Forest":
            # Importance des caractéristiques pour Random Forest
            importance = result["model"].feature_importances_
        elif name == "XGBoost":
            # Importance des caractéristiques pour XGBoost
            importance = result["model"].feature_importances_
        
        # Trier par importance
        indices = np.argsort(importance)[::-1]
        
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [feature_names[j] for j in indices], rotation=45, ha='right')
        plt.title(f'Importance des caractéristiques - {name}')
        plt.ylabel('Importance')
        plt.tight_layout()
    
    plt.savefig("outputs/plots/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyse SHAP pour le meilleur modèle
    best_model_name = max(results.keys(), key=lambda x: results[x]["auc"])
    best_model = results[best_model_name]["model"]
    
    print(f"Analyse SHAP pour le meilleur modèle: {best_model_name}")
    
    # Créer un explainer SHAP
    if hasattr(best_model, 'feature_importances_'):
        explainer = shap.TreeExplainer(best_model)
        X_train_scaled = scaler.transform(X_train)
        shap_values = explainer.shap_values(X_train_scaled)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train_scaled, feature_names=feature_names, show=False)
        plt.title(f'Analyse SHAP - {best_model_name}')
        plt.tight_layout()
        plt.savefig("outputs/plots/shap_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(results, comparison_df):
    """Génère un rapport de l'étape 2."""
    print("\n=== Génération du rapport ===")
    
    report = f"""
# Rapport Étape 2 : Développement de Modèles

## Résumé Exécutif

Cette étape a permis de développer et d'évaluer plusieurs modèles de classification pour prédire la résistance au céfidérocol.

## Résultats des Modèles

{comparison_df.to_markdown()}

## Meilleur Modèle

Le meilleur modèle selon l'AUC est: {max(results.keys(), key=lambda x: results[x]["auc"])}

## Métriques Clés

- **AUC moyen**: {comparison_df['AUC'].mean():.4f}
- **Précision moyenne**: {comparison_df['Précision'].mean():.4f}
- **Rappel moyen**: {comparison_df['Rappel'].mean():.4f}

## Graphiques Générés

1. `model_evaluation.png` - Courbes ROC et Precision-Recall
2. `feature_importance.png` - Importance des caractéristiques par modèle
3. `shap_analysis.png` - Analyse SHAP du meilleur modèle

## Recommandations

- Le modèle {max(results.keys(), key=lambda x: results[x]["auc"])} montre les meilleures performances
- Considérer l'utilisation de techniques de rééchantillonnage pour gérer le déséquilibre de classes
- Explorer d'autres caractéristiques dérivées pour améliorer les performances
"""
    
    with open("outputs/step2_report.md", "w") as f:
        f.write(report)
    
    print("Rapport généré: outputs/step2_report.md")

def main():
    """Fonction principale pour l'étape 2."""
    print("=== ÉTAPE 2 : DÉVELOPPEMENT DE MODÈLES ===")
    
    # 1. Charger et préparer les données
    data = load_and_prepare_data()
    
    # 2. Sélection des caractéristiques
    features_df, target, le_species, le_region = feature_selection(data)
    
    # 3. Division des données par année
    X_train, X_test, y_train, y_test = split_data_by_year(features_df, target)
    
    # 4. Entraînement des modèles
    results, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # 5. Évaluation des modèles
    comparison_df = evaluate_models(results, y_test)
    
    # 6. Analyse de l'importance des caractéristiques
    feature_importance_analysis(results, X_train, scaler)
    
    # 7. Génération du rapport
    generate_report(results, comparison_df)
    
    print("\n=== ÉTAPE 2 TERMINÉE ===")

if __name__ == "__main__":
    main() 