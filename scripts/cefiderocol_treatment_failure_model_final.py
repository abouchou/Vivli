import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
os.makedirs("outputs/plots", exist_ok=True)

def load_and_prepare_data():
    """Charge et prépare les données pour la prédiction de l'échec du traitement au céfidérocol."""
    print("=== Chargement et préparation des données ===")
    
    # Charger les données
    sidero_data = pd.read_excel("1.xlsx")
    atlas_data = pd.read_excel("2.xlsx")
    
    print(f"SIDERO-WT: {sidero_data.shape}")
    print(f"ATLAS: {atlas_data.shape}")
    
    # Standardiser les colonnes SIDERO-WT
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
    
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "ciprofloxacin_mic", "colistin_mic"]
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[col] = sidero_data[col].apply(clean_mic_values)
            sidero_data[col] = pd.to_numeric(sidero_data[col], errors='coerce')
    
    # Définir les points de cassure pour la résistance
    breakpoints = {
        "cefiderocol_mic": 4,
        "meropenem_mic": 8,
        "ciprofloxacin_mic": 1,
        "colistin_mic": 4
    }
    
    # Créer les caractéristiques de résistance
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[f"{col}_resistant"] = (sidero_data[col] >= breakpoints[col]).astype(int)
    
    # Définir la cible : risque d'échec du traitement au céfidérocol
    # Approche réaliste : utiliser les patterns de résistance complexes
    sidero_data["other_resistance_count"] = (
        sidero_data["meropenem_mic_resistant"] + 
        sidero_data["ciprofloxacin_mic_resistant"] + 
        sidero_data["colistin_mic_resistant"]
    )
    
    # Créer des patterns de résistance complexes
    sidero_data["carbapenem_resistance"] = sidero_data["meropenem_mic_resistant"]
    sidero_data["fluoroquinolone_resistance"] = sidero_data["ciprofloxacin_mic_resistant"]
    sidero_data["polymyxin_resistance"] = sidero_data["colistin_mic_resistant"]
    
    # Cible : risque élevé d'échec basé sur des patterns complexes
    # Utiliser des combinaisons spécifiques plutôt que des seuils simples
    sidero_data["treatment_failure_risk"] = (
        # Pattern 1: Résistance aux carbapénèmes + au moins une autre résistance
        ((sidero_data["carbapenem_resistance"] == 1) & (sidero_data["other_resistance_count"] >= 2)) |
        # Pattern 2: Résistance à 3 antibiotiques différents
        (sidero_data["other_resistance_count"] >= 3) |
        # Pattern 3: Résistance aux carbapénèmes + fluoroquinolones
        ((sidero_data["carbapenem_resistance"] == 1) & (sidero_data["fluoroquinolone_resistance"] == 1))
    ).astype(int)
    
    # Filtrer les données avec des valeurs MIC valides
    sidero_data = sidero_data.dropna(subset=mic_columns)
    
    print(f"Données après nettoyage: {sidero_data.shape}")
    print(f"Distribution de la cible (risque d'échec du traitement):")
    print(sidero_data["treatment_failure_risk"].value_counts(normalize=True))
    
    return sidero_data, atlas_data

def prepare_features(sidero_data):
    """Prépare les caractéristiques pour le modèle de prédiction."""
    print("\n=== Préparation des caractéristiques ===")
    
    # Caractéristiques numériques (exclure la résistance au céfidérocol pour éviter la fuite)
    numeric_features = [
        "meropenem_mic", "ciprofloxacin_mic", "colistin_mic",
        "meropenem_mic_resistant", "ciprofloxacin_mic_resistant", 
        "colistin_mic_resistant", "other_resistance_count",
        "carbapenem_resistance", "fluoroquinolone_resistance", "polymyxin_resistance"
    ]
    
    # Caractéristiques catégorielles
    categorical_features = ["species", "region"]
    
    # Encoder les variables catégorielles
    le = LabelEncoder()
    for col in categorical_features:
        if col in sidero_data.columns:
            sidero_data[f"{col}_encoded"] = le.fit_transform(sidero_data[col].astype(str))
            numeric_features.append(f"{col}_encoded")
    
    # Créer des caractéristiques d'interaction plus sophistiquées
    sidero_data["meropenem_ciprofloxacin_ratio"] = sidero_data["meropenem_mic"] / (sidero_data["ciprofloxacin_mic"] + 0.1)
    sidero_data["meropenem_colistin_ratio"] = sidero_data["meropenem_mic"] / (sidero_data["colistin_mic"] + 0.1)
    sidero_data["resistance_pattern_score"] = (
        sidero_data["carbapenem_resistance"] * 3 + 
        sidero_data["fluoroquinolone_resistance"] * 2 + 
        sidero_data["polymyxin_resistance"]
    )
    
    # Caractéristiques de combinaison
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
    
    # Sélectionner les caractéristiques finales
    feature_columns = [col for col in numeric_features if col in sidero_data.columns]
    
    print(f"Caractéristiques utilisées: {feature_columns}")
    print(f"Nombre de caractéristiques: {len(feature_columns)}")
    
    return sidero_data, feature_columns

def train_models(X_train, X_test, y_train, y_test):
    """Entraîne différents modèles de prédiction."""
    print("\n=== Entraînement des modèles ===")
    
    # Standardiser les caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Définir les modèles avec des paramètres plus conservateurs
    models = {
        "Régression logistique": LogisticRegression(random_state=42, max_iter=1000, C=0.1),
        "Arbre de décision": DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=50),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5, min_samples_split=20),
        "XGBoost": xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=3, learning_rate=0.1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEntraînement de {name}...")
        
        # Entraîner le modèle
        if name == "Régression logistique":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Évaluer les performances
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train if name != "Régression logistique" else X_train_scaled, 
                                  y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'scaler': scaler if name == "Régression logistique" else None,
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
        print(f"  Précision: {precision:.3f}")
        print(f"  Rappel: {recall:.3f}")
        print(f"  F1-score: {f1:.3f}")
        print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return results

def analyze_performance(results, X_test, y_test):
    """Analyse détaillée des performances des modèles."""
    print("\n=== Analyse des performances ===")
    
    # Comparaison des modèles
    comparison_df = pd.DataFrame({
        'Modèle': list(results.keys()),
        'AUC': [results[name]['auc'] for name in results.keys()],
        'Précision': [results[name]['precision'] for name in results.keys()],
        'Rappel': [results[name]['recall'] for name in results.keys()],
        'F1-score': [results[name]['f1'] for name in results.keys()],
        'CV AUC (moyenne)': [results[name]['cv_mean'] for name in results.keys()],
        'CV AUC (écart-type)': [results[name]['cv_std'] for name in results.keys()]
    })
    
    print("\nComparaison des modèles:")
    print(comparison_df.round(3))
    
    # Identifier le meilleur modèle
    best_model_name = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Modèle']
    best_model = results[best_model_name]['model']
    
    print(f"\nMeilleur modèle: {best_model_name} (AUC: {results[best_model_name]['auc']:.3f})")
    
    # Courbes ROC
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC')
    plt.legend()
    plt.grid(True)
    
    # Courbes Precision-Recall
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['y_pred_proba'])
        plt.plot(recall, precision, label=f"{name} (F1 = {result['f1']:.3f})")
    
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title('Courbes Precision-Recall')
    plt.legend()
    plt.grid(True)
    
    # Matrices de confusion pour le meilleur modèle
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion - {best_model_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    
    # Comparaison des métriques
    plt.subplot(2, 2, 4)
    metrics = ['AUC', 'Précision', 'Rappel', 'F1-score']
    metric_keys = ['auc', 'precision', 'recall', 'f1']
    values = [results[best_model_name][key] for key in metric_keys]
    
    plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(f'Métriques de performance - {best_model_name}')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/cefiderocol_model_performance_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model_name, best_model, comparison_df

def analyze_feature_importance(results, X_train, feature_names):
    """Analyse l'importance des caractéristiques."""
    print("\n=== Analyse de l'importance des caractéristiques ===")
    
    # Trouver le meilleur modèle
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    
    # Analyser l'importance des caractéristiques selon le type de modèle
    if hasattr(best_model, 'feature_importances_'):
        # Pour Random Forest et XGBoost
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        # Pour la régression logistique
        importances = np.abs(best_model.coef_[0])
    else:
        print("Impossible d'extraire l'importance des caractéristiques pour ce modèle.")
        return None
    
    # Créer un DataFrame avec l'importance des caractéristiques
    importance_df = pd.DataFrame({
        'Caractéristique': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 des caractéristiques les plus importantes:")
    print(importance_df.head(10))
    
    # Visualisation de l'importance des caractéristiques
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Caractéristique'])
    plt.xlabel('Importance')
    plt.title(f'Importance des caractéristiques - {best_model_name}')
    plt.gca().invert_yaxis()
    
    # Analyse SHAP pour Random Forest ou XGBoost
    if hasattr(best_model, 'feature_importances_'):
        plt.subplot(1, 2, 2)
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_train[:100])  # Échantillon pour la visualisation
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Pour la classe positive
            
            shap.summary_plot(shap_values, X_train[:100], feature_names=feature_names, 
                            show=False, plot_type="bar")
            plt.title('Analyse SHAP')
        except Exception as e:
            print(f"Erreur lors de l'analyse SHAP: {e}")
            plt.text(0.5, 0.5, 'SHAP non disponible', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/cefiderocol_feature_importance_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def generate_clinical_insights(results, sidero_data, importance_df):
    """Génère des insights cliniques basés sur les résultats du modèle."""
    print("\n=== Insights cliniques ===")
    
    # Analyser la distribution des caractéristiques importantes
    if importance_df is not None:
        top_features = importance_df.head(5)['Caractéristique'].tolist()
        
        print("\nAnalyse des caractéristiques les plus importantes:")
        for feature in top_features:
            if feature in sidero_data.columns:
                print(f"\n{feature}:")
                print(f"  Moyenne: {sidero_data[feature].mean():.3f}")
                print(f"  Écart-type: {sidero_data[feature].std():.3f}")
                print(f"  Distribution par risque d'échec:")
                print(sidero_data.groupby('treatment_failure_risk')[feature].describe())
    
    # Analyser les patterns de résistance
    print("\nPatterns de résistance associés au risque d'échec du traitement:")
    resistance_features = [col for col in sidero_data.columns if 'resistant' in col and 'cefiderocol' not in col]
    
    for feature in resistance_features:
        if feature in sidero_data.columns:
            failure_rate = sidero_data.groupby(feature)['treatment_failure_risk'].mean()
            print(f"{feature}:")
            for value, rate in failure_rate.items():
                print(f"  {value}: {rate:.3f} ({rate*100:.1f}%)")
    
    return top_features if importance_df is not None else []

def generate_report(results, best_model_name, best_model, comparison_df, importance_df, clinical_insights):
    """Génère un rapport complet du modèle de prédiction."""
    print("\n=== Génération du rapport ===")
    
    report = f"""
# Rapport Final: Modèle de Prédiction du Risque d'Échec du Traitement au Céfidérocol

## Résumé Exécutif

Ce rapport présente le développement d'un modèle de prédiction pour estimer le risque d'échec d'un traitement au céfidérocol. L'objectif est de fournir un outil d'aide à la décision clinique basé sur les caractéristiques microbiologiques disponibles.

**IMPORTANT**: Ce modèle présente des limitations importantes liées au manque de données sur les échecs de traitement réels. Les performances élevées observées reflètent probablement une définition de cible trop simplifiée plutôt que la capacité réelle de prédiction.

## Méthodologie

### Définition de la Cible
Le risque d'échec du traitement au céfidérocol a été défini comme :
- Pattern 1: Résistance aux carbapénèmes + au moins une autre résistance
- Pattern 2: Résistance à 3 antibiotiques différents
- Pattern 3: Résistance aux carbapénèmes + fluoroquinolones

**Note critique**: Cette définition est basée sur des patterns de résistance théoriques et non sur des échecs de traitement documentés.

### Caractéristiques Utilisées
- Valeurs MIC du méropénème, ciprofloxacine, colistine
- Statuts de résistance aux autres antibiotiques
- Espèce bactérienne et région géographique
- Scores de résistance multiple et ratios MIC
- Patterns de résistance complexes

## Résultats des Modèles

### Comparaison des Performances

{comparison_df.to_string(index=False)}

### Meilleur Modèle: {best_model_name}

- **AUC**: {results[best_model_name]['auc']:.3f}
- **Précision**: {results[best_model_name]['precision']:.3f}
- **Rappel**: {results[best_model_name]['recall']:.3f}
- **F1-score**: {results[best_model_name]['f1']:.3f}

## Analyse des Caractéristiques

### Caractéristiques les Plus Importantes

{importance_df.head(10).to_string(index=False) if importance_df is not None else "Non disponible"}

## Insights Cliniques

### Patterns de Résistance
Les caractéristiques suivantes sont les plus prédictives du risque d'échec du traitement :

{chr(10).join([f"- {feature}" for feature in clinical_insights[:5]]) if clinical_insights else "Non disponible"}

## Limitations Critiques

### 1. Manque de Données Réelles
- Aucune donnée sur les échecs de traitement au céfidérocol n'est disponible
- La cible est basée sur des patterns de résistance théoriques
- Les performances élevées ne reflètent pas la réalité clinique

### 2. Définition de Cible Simplifiée
- La cible est définie par des règles logiques simples
- Pas de données sur les échecs de traitement documentés
- Les modèles apprennent ces règles plutôt que des patterns cliniques réels

### 3. Fuite de Données Potentielle
- Les caractéristiques utilisées sont directement liées à la définition de la cible
- Les modèles peuvent "tricher" en apprenant les règles de définition

### 4. Biais Temporel et Géographique
- Les données peuvent ne pas refléter les patterns actuels
- Biais géographiques dans la collecte des données

## Recommandations

### 1. Validation Clinique
- Le modèle doit être validé sur des données d'échecs de traitement réels
- Collaboration avec des centres cliniques pour collecter des données de suivi

### 2. Amélioration de la Définition de Cible
- Développer une définition basée sur des échecs de traitement documentés
- Inclure des facteurs cliniques (gravité de l'infection, comorbidités, etc.)

### 3. Utilisation Prudente
- Le modèle ne doit PAS être utilisé en pratique clinique actuellement
- Utiliser uniquement comme outil de recherche et de développement

### 4. Collecte de Données
- Mettre en place des systèmes de surveillance des échecs de traitement
- Collaborer avec des réseaux de surveillance microbiologique

## Conclusion

Le modèle développé présente des performances élevées (AUC: {results[best_model_name]['auc']:.3f}), mais ces performances sont probablement artificielles et reflètent les limitations de la définition de cible plutôt que la capacité réelle de prédiction.

**Recommandation principale**: Ce modèle ne doit pas être utilisé en pratique clinique. Il sert de démonstration de la méthodologie et met en évidence le besoin critique de données sur les échecs de traitement réels.

Les prochaines étapes devraient inclure :
1. Collecte de données sur les échecs de traitement au céfidérocol
2. Développement d'une définition de cible basée sur des données cliniques réelles
3. Validation du modèle sur des données indépendantes
4. Collaboration avec des experts cliniques pour l'interprétation des résultats

---
*Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*
"""
    
    # Sauvegarder le rapport
    with open('outputs/cefiderocol_treatment_failure_report_final.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Rapport sauvegardé dans 'outputs/cefiderocol_treatment_failure_report_final.md'")
    
    return report

def main():
    """Fonction principale pour exécuter l'analyse complète."""
    print("=== Développement du Modèle de Prédiction du Risque d'Échec du Traitement au Céfidérocol ===\n")
    
    # 1. Charger et préparer les données
    sidero_data, atlas_data = load_and_prepare_data()
    
    # 2. Préparer les caractéristiques
    sidero_data, feature_columns = prepare_features(sidero_data)
    
    # 3. Préparer les données pour l'entraînement
    X = sidero_data[feature_columns]
    y = sidero_data['treatment_failure_risk']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDonnées d'entraînement: {X_train.shape}")
    print(f"Données de test: {X_test.shape}")
    print(f"Distribution de la cible dans l'entraînement:")
    print(y_train.value_counts(normalize=True))
    
    # 4. Entraîner les modèles
    results = train_models(X_train, X_test, y_train, y_test)
    
    # 5. Analyser les performances
    best_model_name, best_model, comparison_df = analyze_performance(results, X_test, y_test)
    
    # 6. Analyser l'importance des caractéristiques
    importance_df = analyze_feature_importance(results, X_train, feature_columns)
    
    # 7. Générer des insights cliniques
    clinical_insights = generate_clinical_insights(results, sidero_data, importance_df)
    
    # 8. Générer le rapport final
    report = generate_report(results, best_model_name, best_model, comparison_df, importance_df, clinical_insights)
    
    print("\n=== Analyse terminée ===")
    print(f"Le meilleur modèle ({best_model_name}) a une AUC de {results[best_model_name]['auc']:.3f}")
    print("⚠️  ATTENTION: Les performances élevées reflètent probablement des limitations méthodologiques")
    print("Le rapport complet a été sauvegardé dans 'outputs/cefiderocol_treatment_failure_report_final.md'")

if __name__ == "__main__":
    main() 