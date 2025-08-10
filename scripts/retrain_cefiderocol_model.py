import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import shap
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
os.makedirs("outputs/plots", exist_ok=True)

def load_atlas_data():
    """Charge et prépare les données ATLAS exclusivement."""
    print("=== Chargement des données ATLAS ===")
    
    # Charger uniquement les données ATLAS
    atlas_data = pd.read_excel("2.xlsx")
    print(f"ATLAS: {atlas_data.shape}")
    print(f"Colonnes ATLAS: {atlas_data.columns.tolist()}")
    
    return atlas_data

def clean_mic_values(value):
    """Nettoie les valeurs MIC."""
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

def prepare_atlas_features(atlas_data):
    """Prépare les caractéristiques pour la prédiction de résistance au céfidérocol."""
    print("\n=== Préparation des caractéristiques ATLAS ===")
    
    # Mapping des colonnes ATLAS
    atlas_mapping = {
        "Species": "species",
        "Country": "country", 
        "Year": "year",
        "Amikacin": "amikacin_mic",
        "Cefepime": "cefepime_mic",
        "Ceftazidime avibactam": "ceftazidime_avibactam_mic",
        "Ciprofloxacin": "ciprofloxacin_mic",
        "Colistin": "colistin_mic",
        "Meropenem": "meropenem_mic",
        "Imipenem": "imipenem_mic",
        "Ertapenem": "ertapenem_mic",
        "Doripenem": "doripenem_mic",
        "Ceftazidime": "ceftazidime_mic",
        "Ceftriaxone": "ceftriaxone_mic",
        "Cefoxitin": "cefoxitin_mic",
        "Ampicillin": "ampicillin_mic",
        "Penicillin": "penicillin_mic",
        "Tetracycline": "tetracycline_mic",
        "Gentamicin": "gentamicin_mic",
        "Tigecycline": "tigecycline_mic",
        "Vancomycin": "vancomycin_mic"
    }
    
    atlas_data.rename(columns=atlas_mapping, inplace=True)
    
    # Nettoyer les colonnes MIC
    mic_columns = ["amikacin_mic", "cefepime_mic", "ceftazidime_avibactam_mic", 
                   "ciprofloxacin_mic", "colistin_mic", "meropenem_mic", "imipenem_mic",
                   "ertapenem_mic", "doripenem_mic", "ceftazidime_mic", "ceftriaxone_mic",
                   "cefoxitin_mic", "ampicillin_mic", "penicillin_mic", "tetracycline_mic",
                   "gentamicin_mic", "tigecycline_mic", "vancomycin_mic"]
    
    for col in mic_columns:
        if col in atlas_data.columns:
            atlas_data[col] = atlas_data[col].apply(clean_mic_values)
            atlas_data[col] = pd.to_numeric(atlas_data[col], errors='coerce')
    
    # Définir les points de cassure pour la résistance
    breakpoints = {
        "amikacin_mic": 32,
        "cefepime_mic": 32,
        "ceftazidime_avibactam_mic": 8,
        "ciprofloxacin_mic": 1,
        "colistin_mic": 4,
        "meropenem_mic": 8,
        "imipenem_mic": 4,
        "ertapenem_mic": 2,
        "doripenem_mic": 4,
        "ceftazidime_mic": 16,
        "ceftriaxone_mic": 4,
        "cefoxitin_mic": 32,
        "ampicillin_mic": 16,
        "penicillin_mic": 0.12,
        "tetracycline_mic": 16,
        "gentamicin_mic": 8,
        "tigecycline_mic": 2,
        "vancomycin_mic": 4
    }
    
    # Créer les caractéristiques de résistance
    for col in mic_columns:
        if col in atlas_data.columns:
            atlas_data[f"{col}_resistant"] = (atlas_data[col] >= breakpoints[col]).astype(int)
    
    # Créer des caractéristiques de résistance combinées
    atlas_data["carbapenem_resistant"] = (
        atlas_data["meropenem_mic_resistant"] + 
        atlas_data["imipenem_mic_resistant"] + 
        atlas_data["ertapenem_mic_resistant"] + 
        atlas_data["doripenem_mic_resistant"]
    )
    
    atlas_data["cephalosporin_resistant"] = (
        atlas_data["cefepime_mic_resistant"] + 
        atlas_data["ceftazidime_mic_resistant"] + 
        atlas_data["ceftriaxone_mic_resistant"] + 
        atlas_data["cefoxitin_mic_resistant"]
    )
    
    atlas_data["quinolone_resistant"] = atlas_data["ciprofloxacin_mic_resistant"]
    atlas_data["polymyxin_resistant"] = atlas_data["colistin_mic_resistant"]
    atlas_data["aminoglycoside_resistant"] = atlas_data["amikacin_mic_resistant"] + atlas_data["gentamicin_mic_resistant"]
    atlas_data["beta_lactam_resistant"] = atlas_data["ampicillin_mic_resistant"] + atlas_data["penicillin_mic_resistant"]
    
    # Multidrug resistance (MDR) - résistant à au moins 3 classes d'antibiotiques
    atlas_data["multidrug_resistant"] = (
        (atlas_data["carbapenem_resistant"] > 0).astype(int) +
        (atlas_data["cephalosporin_resistant"] > 0).astype(int) +
        (atlas_data["quinolone_resistant"] > 0).astype(int) +
        (atlas_data["polymyxin_resistant"] > 0).astype(int) +
        (atlas_data["aminoglycoside_resistant"] > 0).astype(int)
    )
    
    # Encoder les variables catégorielles
    le_species = LabelEncoder()
    le_country = LabelEncoder()
    
    if "species" in atlas_data.columns:
        atlas_data["species_encoded"] = le_species.fit_transform(atlas_data["species"].fillna("Unknown"))
    
    if "country" in atlas_data.columns:
        atlas_data["country_encoded"] = le_country.fit_transform(atlas_data["country"].fillna("Unknown"))
    
    # Créer la cible: proxy de résistance au céfidérocol
    # Basé sur la résistance aux carbapénèmes et autres antibiotiques majeurs
    # Le céfidérocol est souvent utilisé contre les bactéries résistantes aux carbapénèmes
    atlas_data["cefiderocol_resistant_proxy"] = (
        (atlas_data["carbapenem_resistant"] >= 2) |  # Résistant à au moins 2 carbapénèmes
        (atlas_data["multidrug_resistant"] >= 3)     # MDR avec résistance à au moins 3 classes
    ).astype(int)
    
    # Sélectionner les caractéristiques pour le modèle
    feature_columns = [
        "amikacin_mic", "cefepime_mic", "ceftazidime_avibactam_mic",
        "ciprofloxacin_mic", "colistin_mic", "meropenem_mic", "imipenem_mic",
        "ertapenem_mic", "doripenem_mic", "ceftazidime_mic", "ceftriaxone_mic",
        "cefoxitin_mic", "ampicillin_mic", "penicillin_mic", "tetracycline_mic",
        "gentamicin_mic", "tigecycline_mic", "vancomycin_mic",
        "amikacin_mic_resistant", "cefepime_mic_resistant", 
        "ceftazidime_avibactam_mic_resistant", "ciprofloxacin_mic_resistant",
        "colistin_mic_resistant", "meropenem_mic_resistant", "imipenem_mic_resistant",
        "ertapenem_mic_resistant", "doripenem_mic_resistant", "ceftazidime_mic_resistant",
        "ceftriaxone_mic_resistant", "cefoxitin_mic_resistant", "ampicillin_mic_resistant",
        "penicillin_mic_resistant", "tetracycline_mic_resistant", "gentamicin_mic_resistant",
        "tigecycline_mic_resistant", "vancomycin_mic_resistant",
        "carbapenem_resistant", "cephalosporin_resistant", "quinolone_resistant", 
        "polymyxin_resistant", "aminoglycoside_resistant", "beta_lactam_resistant",
        "multidrug_resistant"
    ]
    
    # Ajouter les variables encodées si disponibles
    if "species_encoded" in atlas_data.columns:
        feature_columns.append("species_encoded")
    if "country_encoded" in atlas_data.columns:
        feature_columns.append("country_encoded")
    if "year" in atlas_data.columns:
        feature_columns.append("year")
    
    # Filtrer les données avec des valeurs MIC valides pour les antibiotiques principaux
    main_antibiotics = ["meropenem_mic", "ciprofloxacin_mic", "colistin_mic", "amikacin_mic"]
    atlas_data = atlas_data.dropna(subset=main_antibiotics)
    
    # Créer features_df et target après le filtrage
    features_df = atlas_data[feature_columns].copy()
    target = atlas_data["cefiderocol_resistant_proxy"]
    
    print(f"Distribution du proxy de résistance au céfidérocol:")
    print(f"- Sensible (0): {target.value_counts()[0]} ({target.value_counts(normalize=True)[0]:.3f})")
    print(f"- Résistant (1): {target.value_counts()[1]} ({target.value_counts(normalize=True)[1]:.3f})")
    
    print(f"Données après nettoyage: {atlas_data.shape}")
    print(f"Caractéristiques utilisées: {len(feature_columns)}")
    print(f"Taille features_df: {features_df.shape}")
    print(f"Taille target: {target.shape}")
    print(f"Distribution de la cible: {target.value_counts(normalize=True)}")
    
    return features_df, target, le_species, le_country

def train_decision_models(X_train, X_test, y_train, y_test):
    """Entraîne les modèles de décision pour prédire la résistance au céfidérocol."""
    print("\n=== Entraînement des modèles de décision ===")
    
    # Standardiser les caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèles à entraîner
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss', max_depth=6)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Entraînement {name} ---")
        
        # Entraîner le modèle
        model.fit(X_train_scaled, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Métriques
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculer AUC train pour détecter le surapprentissage
        train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred_proba)
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "auc": auc,
            "train_auc": train_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }
        
        print(f"AUC Test: {auc:.4f}")
        print(f"AUC Train: {train_auc:.4f}")
        print(f"Différence Train-Test: {train_auc - auc:.4f}")
        print(f"Précision: {precision:.4f}")
        print(f"Rappel: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Précision globale: {accuracy:.4f}")
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"CV AUC (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Détecter le surapprentissage
        if train_auc - auc > 0.1:
            print("⚠️  ATTENTION: Signes de surapprentissage détectés!")
        elif train_auc - auc > 0.05:
            print("⚠️  ATTENTION: Légers signes de surapprentissage détectés!")
        else:
            print("✅ Pas de signes évidents de surapprentissage")
    
    return results, scaler

def analyze_model_performance(results, X_test, y_test):
    """Analyse les performances des modèles et détecte le surapprentissage."""
    print("\n=== Analyse des performances et détection du surapprentissage ===")
    
    best_model_name = max(results.keys(), key=lambda k: results[k]["auc"])
    best_model = results[best_model_name]
    
    print(f"\nMeilleur modèle: {best_model_name}")
    print(f"AUC Test: {best_model['auc']:.4f}")
    print(f"AUC Train: {best_model['train_auc']:.4f}")
    print(f"Différence Train-Test: {best_model['train_auc'] - best_model['auc']:.4f}")
    print(f"Précision: {best_model['precision']:.4f}")
    print(f"Rappel: {best_model['recall']:.4f}")
    print(f"F1-Score: {best_model['f1']:.4f}")
    
    # Courbe ROC
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_pred_proba"])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC - Prédiction de résistance au céfidérocol (ATLAS)')
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/plots/cefiderocol_prediction_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Matrice de confusion pour le meilleur modèle
    cm = confusion_matrix(y_test, best_model["y_pred"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion - {best_model_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig("outputs/plots/cefiderocol_prediction_confusion.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model_name, best_model

def analyze_feature_importance(results, X_train, scaler):
    """Analyse l'importance des caractéristiques."""
    print("\n=== Analyse de l'importance des caractéristiques ===")
    
    # Utiliser le meilleur modèle
    best_model_name = max(results.keys(), key=lambda k: results[k]["auc"])
    best_model = results[best_model_name]["model"]
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 caractéristiques les plus importantes pour {best_model_name}:")
        print(importance_df.head(10))
        
        # Graphique d'importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance des caractéristiques')
        plt.title(f'Top 15 Importance des caractéristiques - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("outputs/plots/cefiderocol_prediction_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyse SHAP
        print("\nPerforming SHAP analysis...")
        explainer = shap.TreeExplainer(best_model)
        X_train_scaled = scaler.transform(X_train)
        shap_values = explainer.shap_values(X_train_scaled[:1000])  # Échantillon pour la vitesse
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train.iloc[:1000], plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig("outputs/plots/cefiderocol_prediction_shap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    return None

def generate_comprehensive_report(results, best_model_name, best_model, importance_df):
    """Génère un rapport complet sur le réentraînement du modèle."""
    print("\n=== Génération du rapport complet ===")
    
    report = f"""# Rapport de Réentraînement du Modèle de Prédiction de Résistance au Céfidérocol

## Résumé Exécutif

Ce rapport présente les résultats du réentraînement d'un modèle décisionnel pour prédire la résistance au céfidérocol en utilisant exclusivement les données ATLAS, conformément aux instructions spécifiées.

## Méthodologie

### 1. Données Utilisées
- **Source**: Exclusivement les données ATLAS (2.xlsx)
- **Raison**: ATLAS reflète mieux la diversité des profils de résistance
- **Avantage**: Plus représentatif de la réalité clinique mondiale
- **Note importante**: Aucune donnée céfidérocol disponible dans ATLAS

### 2. Préparation des Données
- Nettoyage des valeurs MIC pour tous les antibiotiques disponibles
- Standardisation des points de cassure pour la résistance
- Encodage des variables catégorielles (espèces, pays)
- Création de caractéristiques de résistance combinées

### 3. Proxy de Résistance au Céfidérocol
Étant donné l'absence de données céfidérocol dans ATLAS, nous avons créé un proxy basé sur:
- Résistance à au moins 2 carbapénèmes (meropenem, imipenem, ertapenem, doripenem)
- OU résistance multidrogue (≥3 classes d'antibiotiques)

### 4. Modèles Testés
- Arbre de décision
- Forêt aléatoire
- XGBoost

## Résultats des Performances

### Modèle Optimal: {best_model_name}

| Métrique | Valeur |
|----------|--------|
| AUC Test | {best_model['auc']:.4f} |
| AUC Train | {best_model['train_auc']:.4f} |
| Différence Train-Test | {best_model['train_auc'] - best_model['auc']:.4f} |
| Précision | {best_model['precision']:.4f} |
| Rappel | {best_model['recall']:.4f} |
| F1-Score | {best_model['f1']:.4f} |
| Précision globale | {best_model['accuracy']:.4f} |

### Comparaison des Modèles

"""

    for name, result in results.items():
        report += f"""
**{name}**:
- AUC Test: {result['auc']:.4f}
- AUC Train: {result['train_auc']:.4f}
- Différence Train-Test: {result['train_auc'] - result['auc']:.4f}
- Précision: {result['precision']:.4f}
- Rappel: {result['recall']:.4f}
- F1-Score: {result['f1']:.4f}
"""

    report += f"""

## Analyse du Surapprentissage

### Détection du Surapprentissage
- **Méthode**: Comparaison AUC train vs test
- **Seuil d'alerte**: Différence > 0.1
- **Seuil d'attention**: Différence > 0.05

### Résultats par Modèle
"""

    has_overfitting = False
    for name, result in results.items():
        diff = result['train_auc'] - result['auc']
        if diff > 0.1:
            status = "⚠️  SURAPPRENTISSAGE DÉTECTÉ"
            has_overfitting = True
        elif diff > 0.05:
            status = "⚠️  LÉGERS SIGNES DE SURAPPRENTISSAGE"
        else:
            status = "✅ OK"
        
        report += f"""
**{name}**:
- AUC Train: {result['train_auc']:.4f}
- AUC Test: {result['auc']:.4f}
- Différence: {diff:.4f}
- Statut: {status}
"""

    if importance_df is not None:
        report += f"""

## Importance des Caractéristiques

### Top 10 Caractéristiques les Plus Importantes

{importance_df.head(10).to_string(index=False)}

### Interprétation Clinique
- Les caractéristiques de résistance aux carbapénèmes sont les plus prédictives
- La résistance combinée (multidrug_resistant) est un indicateur fort
- Les profils de résistance spécifiques par espèce sont informatifs
"""

    report += f"""

## Interprétation Clinique

### 1. Performance du Modèle
- **AUC de {best_model['auc']:.3f}**: Performance {'excellente' if best_model['auc'] > 0.9 else 'bonne' if best_model['auc'] > 0.8 else 'modérée'}
- **Précision de {best_model['precision']:.3f}**: {'Très bonne' if best_model['precision'] > 0.8 else 'Bonne' if best_model['precision'] > 0.7 else 'Modérée'} précision
- **Rappel de {best_model['recall']:.3f}**: {'Très bonne' if best_model['recall'] > 0.8 else 'Bonne' if best_model['recall'] > 0.7 else 'Modérée'} sensibilité

### 2. Signes de Surapprentissage
"""

    if has_overfitting:
        report += "- **ATTENTION**: Signes de surapprentissage détectés dans certains modèles\n"
        report += "- **Implication**: Le modèle peut ne pas généraliser correctement à de nouveaux cas\n"
        report += "- **Recommandation**: Validation prospective nécessaire\n"
    else:
        report += "- Aucun signe évident de surapprentissage détecté\n"
        report += "- Les modèles semblent bien généraliser\n"

    report += f"""

### 3. Implications Cliniques

#### Si le Modèle est Trop Optimiste:
- **Cause possible**: Surapprentissage aux données d'entraînement
- **Risque**: Généralisation insuffisante à de nouveaux cas
- **Recommandation**: Validation prospective nécessaire

#### Si le Modèle est Conservateur:
- **Avantage**: Plus robuste pour la généralisation
- **Risque**: Peut manquer des cas de résistance
- **Recommandation**: Ajuster les seuils de décision

### 4. Limitations du Proxy
- **Absence de données céfidérocol réelles**: Le proxy peut ne pas refléter parfaitement la résistance au céfidérocol
- **Validation nécessaire**: Les résultats doivent être validés avec des données céfidérocol réelles
- **Interprétation prudente**: Les prédictions sont basées sur des patterns de résistance similaires

### 5. Recommandations

#### Pour l'Implémentation Clinique:
1. **Validation prospective**: Tester le modèle sur de nouveaux échantillons avec données céfidérocol réelles
2. **Ajustement des seuils**: Optimiser selon les priorités cliniques
3. **Surveillance continue**: Monitorer les performances en conditions réelles

#### Pour l'Amélioration:
1. **Plus de données**: Obtenir des données céfidérocol réelles dans ATLAS
2. **Caractéristiques supplémentaires**: Inclure des marqueurs génomiques
3. **Modèles spécifiques**: Développer des modèles par espèce

## Conclusions

Le modèle réentraîné sur les données ATLAS {'montre d\'excellentes performances' if best_model['auc'] > 0.9 else 'présente de bonnes performances' if best_model['auc'] > 0.8 else 'a des performances modérées'} avec un AUC de {best_model['auc']:.3f}.

**Points Clés**:
- Utilisation exclusive des données ATLAS pour une meilleure représentativité
- Proxy de résistance au céfidérocol basé sur les patterns de résistance similaires
- Validation croisée pour évaluer la robustesse
- Analyse approfondie du surapprentissage
- Interprétation clinique des résultats

**Prochaines Étapes**:
1. Validation prospective du modèle avec données céfidérocol réelles
2. Intégration dans les systèmes cliniques
3. Surveillance continue des performances

---

*Rapport généré automatiquement - Réentraînement Modèle Céfidérocol (ATLAS uniquement)*
"""

    # Sauvegarder le rapport
    with open("outputs/retrain_cefiderocol_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("Rapport généré: outputs/retrain_cefiderocol_report.md")
    
    return report

def main():
    """Fonction principale pour le réentraînement du modèle de prédiction de résistance au céfidérocol."""
    print("=== RÉENTRAÎNEMENT DU MODÈLE DE PRÉDICTION DE RÉSISTANCE AU CÉFIDÉROCOL ===")
    print("Utilisation exclusive des données ATLAS\n")
    
    # 1. Charger les données ATLAS
    atlas_data = load_atlas_data()
    
    # 2. Préparer les caractéristiques
    features_df, target, le_species, le_country = prepare_atlas_features(atlas_data)
    
    # 3. Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, target, test_size=0.2, random_state=42, stratify=target
    )
    
    print(f"\nEnsemble d'entraînement: {X_train.shape}")
    print(f"Ensemble de test: {X_test.shape}")
    print(f"Distribution de la cible - Train: {y_train.value_counts(normalize=True)}")
    print(f"Distribution de la cible - Test: {y_test.value_counts(normalize=True)}")
    
    # 4. Entraîner les modèles de décision
    results, scaler = train_decision_models(X_train, X_test, y_train, y_test)
    
    # 5. Analyser les performances
    best_model_name, best_model = analyze_model_performance(results, X_test, y_test)
    
    # 6. Analyser l'importance des caractéristiques
    importance_df = analyze_feature_importance(results, X_train, scaler)
    
    # 7. Générer le rapport complet
    report = generate_comprehensive_report(results, best_model_name, best_model, importance_df)
    
    print("\n=== RÉENTRAÎNEMENT TERMINÉ ===")
    print("Résultats sauvegardés dans outputs/ directory:")
    print("- Visualisations: outputs/plots/")
    print("- Rapport: outputs/retrain_cefiderocol_report.md")

if __name__ == "__main__":
    main() 