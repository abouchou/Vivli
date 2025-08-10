import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AntibioticDecisionTree:
    """
    Système de recommandation d'antibiotiques basé sur un arbre de décision.
    Recommande les antibiotiques dans l'ordre d'efficacité avec la céfidérocol en dernier recours.
    """
    
    def __init__(self):
        self.data = None
        self.label_encoders = {}
        self.antibiotic_order = []
        self.decision_tree = None
        
    def load_atlas_data(self, file_path="2.xlsx"):
        """Charge et prépare les données ATLAS."""
        print("Chargement des données ATLAS...")
        self.data = pd.read_excel(file_path)
        print(f"Données chargées : {self.data.shape}")
        return self.data
    
    def clean_mic_values(self, value):
        """Nettoie les valeurs MIC."""
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            import re
            cleaned_value = re.sub(r'[^\d.]', '', value)
            try:
                return float(cleaned_value)
            except ValueError:
                return np.nan
        return float(value)
    
    def prepare_antibiotic_data(self):
        """Prépare les données pour l'analyse des antibiotiques."""
        print("Préparation des données d'antibiotiques...")
        
        # Colonnes d'antibiotiques importantes (sans céfidérocol)
        antibiotic_columns = [
            'Meropenem', 'Imipenem', 'Doripenem', 'Ertapenem', 'Tebipenem',
            'Ceftazidime avibactam', 'Ceftaroline avibactam', 'Ceftolozane tazobactam',
            'Meropenem vaborbactam', 'Aztreonam avibactam', 'Ceftibuten avibactam',
            'Cefepime', 'Ceftazidime', 'Ceftriaxone', 'Cefoxitin', 'Cefixime',
            'Ceftaroline', 'Ceftibuten', 'Cefpodoxime', 'Cefoperazone sulbactam',
            'Piperacillin tazobactam', 'Ampicillin sulbactam', 'Amoxycillin clavulanate',
            'Ampicillin', 'Penicillin', 'Oxacillin', 'Sulbactam',
            'Amikacin', 'Gentamicin', 'Tobramycin', 'Streptomycin',
            'Ciprofloxacin', 'Levofloxacin', 'Gatifloxacin', 'Moxifloxacin',
            'Colistin', 'Polymyxin B',
            'Tigecycline', 'Minocycline', 'Tetracycline',
            'Vancomycin', 'Teicoplanin', 'Daptomycin', 'Linezolid',
            'Clarithromycin', 'Erythromycin', 'Azithromycin', 'Clindamycin',
            'Metronidazole', 'Trimethoprim sulfa', 'Sulfamethoxazole',
            'Quinupristin dalfopristin'
        ]
        
        # Nettoyer les valeurs MIC pour chaque antibiotique
        for col in antibiotic_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(self.clean_mic_values)
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Créer des variables binaires pour la résistance (R) vs sensible (S)
        for col in antibiotic_columns:
            if col in self.data.columns:
                # Utiliser les colonnes d'interprétation (_I) si disponibles
                interpretation_col = f"{col}_I"
                if interpretation_col in self.data.columns:
                    # Vérifier que la colonne contient des chaînes de caractères
                    if self.data[interpretation_col].dtype == 'object':
                        self.data[f"{col}_resistant"] = self.data[interpretation_col].astype(str).str.contains('R', na=False)
                    else:
                        # Si ce n'est pas une colonne de chaînes, utiliser les seuils
                        self.data[f"{col}_resistant"] = self.data[col] > self.get_resistance_threshold(col)
                else:
                    # Définir des seuils de résistance basés sur la littérature
                    self.data[f"{col}_resistant"] = self.data[col] > self.get_resistance_threshold(col)
        
        return antibiotic_columns
    
    def get_resistance_threshold(self, antibiotic):
        """Définit les seuils de résistance pour chaque antibiotique (en mg/L)."""
        thresholds = {
            'Meropenem': 8, 'Imipenem': 8, 'Doripenem': 8, 'Ertapenem': 2, 'Tebipenem': 4,
            'Ceftazidime avibactam': 8, 'Ceftaroline avibactam': 4, 'Ceftolozane tazobactam': 8,
            'Meropenem vaborbactam': 8, 'Aztreonam avibactam': 8, 'Ceftibuten avibactam': 4,
            'Cefepime': 32, 'Ceftazidime': 32, 'Ceftriaxone': 64, 'Cefoxitin': 32,
            'Cefixime': 1, 'Ceftaroline': 4, 'Ceftibuten': 4, 'Cefpodoxime': 2,
            'Cefoperazone sulbactam': 64, 'Piperacillin tazobactam': 128,
            'Ampicillin sulbactam': 32, 'Amoxycillin clavulanate': 32,
            'Ampicillin': 32, 'Penicillin': 0.12, 'Oxacillin': 2,
            'Amikacin': 64, 'Gentamicin': 16, 'Tobramycin': 16, 'Streptomycin': 64,
            'Ciprofloxacin': 4, 'Levofloxacin': 8, 'Gatifloxacin': 8, 'Moxifloxacin': 4,
            'Colistin': 4, 'Polymyxin B': 4,
            'Tigecycline': 8, 'Minocycline': 16, 'Tetracycline': 16,
            'Vancomycin': 16, 'Teicoplanin': 32, 'Daptomycin': 4, 'Linezolid': 8,
            'Clarithromycin': 8, 'Erythromycin': 8, 'Azithromycin': 16, 'Clindamycin': 4,
            'Metronidazole': 16, 'Trimethoprim sulfa': 76, 'Sulfamethoxazole': 512,
            'Quinupristin dalfopristin': 4
        }
        return thresholds.get(antibiotic, 8)  # Valeur par défaut
    
    def calculate_antibiotic_scores(self, antibiotic_columns):
        """Calcule un score d'efficacité pour chaque antibiotique."""
        print("Calcul des scores d'efficacité des antibiotiques...")
        
        # Créer un DataFrame pour les scores
        scores_df = self.data[['Species', 'Country', 'Year']].copy()
        
        for col in antibiotic_columns:
            if col in self.data.columns:
                resistant_col = f"{col}_resistant"
                if resistant_col in self.data.columns:
                    # Score basé sur la proportion de souches sensibles
                    scores_df[f"{col}_score"] = ~self.data[resistant_col]
        
        return scores_df
    
    def determine_antibiotic_order(self, scores_df, antibiotic_columns):
        """Détermine l'ordre optimal des antibiotiques basé sur l'efficacité globale."""
        print("Détermination de l'ordre optimal des antibiotiques...")
        
        # Calculer l'efficacité globale de chaque antibiotique
        antibiotic_efficacy = {}
        for col in antibiotic_columns:
            score_col = f"{col}_score"
            if score_col in scores_df.columns:
                efficacy = scores_df[score_col].mean()
                antibiotic_efficacy[col] = efficacy
        
        # Trier par efficacité décroissante
        sorted_antibiotics = sorted(antibiotic_efficacy.items(), key=lambda x: x[1], reverse=True)
        
        # Ajouter la céfidérocol à la fin (dernier recours)
        self.antibiotic_order = [ab[0] for ab in sorted_antibiotics] + ['Cefiderocol']
        
        print("Ordre recommandé des antibiotiques :")
        for i, ab in enumerate(self.antibiotic_order, 1):
            if ab in antibiotic_efficacy:
                print(f"{i}. {ab} (efficacité: {antibiotic_efficacy[ab]:.3f})")
            else:
                print(f"{i}. {ab} (dernier recours)")
        
        return self.antibiotic_order
    
    def create_decision_features(self, scores_df):
        """Crée les caractéristiques pour l'arbre de décision."""
        print("Création des caractéristiques pour l'arbre de décision...")
        
        # Encoder les variables catégorielles
        le_species = LabelEncoder()
        le_country = LabelEncoder()
        
        scores_df['species_encoded'] = le_species.fit_transform(scores_df['Species'].fillna('Unknown'))
        scores_df['country_encoded'] = le_country.fit_transform(scores_df['Country'].fillna('Unknown'))
        
        self.label_encoders['species'] = le_species
        self.label_encoders['country'] = le_country
        
        # Créer des caractéristiques de résistance par classe d'antibiotiques
        beta_lactam_cols = [col for col in scores_df.columns if any(x in col for x in ['penem', 'cef', 'penicillin', 'tazobactam', 'avibactam'])]
        aminoglycoside_cols = [col for col in scores_df.columns if any(x in col for x in ['micin', 'gentamicin', 'tobramycin', 'streptomycin'])]
        quinolone_cols = [col for col in scores_df.columns if any(x in col for x in ['floxacin'])]
        other_cols = [col for col in scores_df.columns if any(x in col for x in ['colistin', 'vancomycin', 'linezolid', 'tigecycline'])]
        
        # Calculer les scores moyens par classe
        scores_df['beta_lactam_resistance'] = scores_df[beta_lactam_cols].mean(axis=1)
        scores_df['aminoglycoside_resistance'] = scores_df[aminoglycoside_cols].mean(axis=1)
        scores_df['quinolone_resistance'] = scores_df[quinolone_cols].mean(axis=1)
        scores_df['other_resistance'] = scores_df[other_cols].mean(axis=1)
        
        return scores_df
    
    def train_decision_tree(self, features_df):
        """Entraîne l'arbre de décision pour recommander le premier antibiotique."""
        print("Entraînement de l'arbre de décision...")
        
        # Sélectionner les caractéristiques
        feature_cols = ['species_encoded', 'country_encoded', 'Year',
                       'beta_lactam_resistance', 'aminoglycoside_resistance',
                       'quinolone_resistance', 'other_resistance']
        
        X = features_df[feature_cols].fillna(0)
        
        # Créer la cible : le premier antibiotique recommandé
        # Utiliser l'antibiotique le plus efficace pour chaque échantillon
        first_choice = []
        for idx, row in features_df.iterrows():
            best_antibiotic = None
            best_score = -1
            for ab in self.antibiotic_order[:-1]:  # Exclure céfidérocol
                score_col = f"{ab}_score"
                if score_col in features_df.columns:
                    if features_df.loc[idx, score_col] > best_score:
                        best_score = features_df.loc[idx, score_col]
                        best_antibiotic = ab
            first_choice.append(best_antibiotic if best_antibiotic else self.antibiotic_order[0])
        
        # Encoder la cible
        le_target = LabelEncoder()
        y = le_target.fit_transform(first_choice)
        self.label_encoders['target'] = le_target
        
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraîner l'arbre de décision
        self.decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.decision_tree.fit(X_train, y_train)
        
        # Évaluer le modèle
        y_pred = self.decision_tree.predict(X_test)
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred, target_names=le_target.classes_))
        
        return X_train, X_test, y_train, y_test
    
    def recommend_antibiotics(self, species, country, year, resistance_profile=None):
        """Recommande une séquence d'antibiotiques pour un cas donné."""
        if self.decision_tree is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations.")
        
        # Préparer les caractéristiques
        species_encoded = self.label_encoders['species'].transform([species])[0]
        country_encoded = self.label_encoders['country'].transform([country])[0]
        
        # Créer le vecteur de caractéristiques
        features = np.array([[
            species_encoded, country_encoded, year,
            resistance_profile.get('beta_lactam', 0.5) if resistance_profile else 0.5,
            resistance_profile.get('aminoglycoside', 0.5) if resistance_profile else 0.5,
            resistance_profile.get('quinolone', 0.5) if resistance_profile else 0.5,
            resistance_profile.get('other', 0.5) if resistance_profile else 0.5
        ]])
        
        # Prédire le premier antibiotique
        first_choice_idx = self.decision_tree.predict(features)[0]
        first_choice = self.label_encoders['target'].inverse_transform([first_choice_idx])[0]
        
        # Construire la séquence recommandée
        recommendations = []
        used_antibiotics = set()
        
        # Ajouter le premier choix
        recommendations.append(first_choice)
        used_antibiotics.add(first_choice)
        
        # Ajouter les alternatives dans l'ordre d'efficacité
        for ab in self.antibiotic_order:
            if ab not in used_antibiotics:
                recommendations.append(ab)
                used_antibiotics.add(ab)
        
        return recommendations
    
    def visualize_tree(self, output_path="outputs/plots/antibiotic_decision_tree.png"):
        """Visualise l'arbre de décision."""
        if self.decision_tree is None:
            print("Aucun arbre de décision à visualiser.")
            return
        
        plt.figure(figsize=(20, 12))
        plot_tree(self.decision_tree, 
                 feature_names=['species', 'country', 'year', 'beta_lactam_res', 
                              'aminoglycoside_res', 'quinolone_res', 'other_res'],
                 class_names=self.label_encoders['target'].classes_,
                 filled=True, rounded=True, fontsize=8)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Arbre de décision sauvegardé : {output_path}")
    
    def generate_report(self, output_path="outputs/antibiotic_decision_tree_report.md"):
        """Génère un rapport détaillé du système de recommandation."""
        report = """# Rapport du Système de Recommandation d'Antibiotiques

## Vue d'ensemble
Ce système utilise un arbre de décision pour recommander les antibiotiques dans l'ordre optimal d'efficacité, avec la céfidérocol comme option de dernier recours.

## Ordre Recommandé des Antibiotiques
"""
        
        for i, ab in enumerate(self.antibiotic_order, 1):
            report += f"{i}. {ab}\n"
        
        report += """

## Caractéristiques du Modèle
- **Espèce bactérienne** : Facteur principal pour déterminer la sensibilité
- **Pays/Région** : Influence sur les profils de résistance locaux
- **Année** : Évolution temporelle des résistances
- **Profil de résistance** : Résistance par classe d'antibiotiques

## Utilisation
```python
# Exemple d'utilisation
recommendations = model.recommend_antibiotics(
    species="Escherichia coli",
    country="France", 
    year=2023,
    resistance_profile={{'beta_lactam': 0.3, 'quinolone': 0.7}}
)
```

## Recommandations
1. **Première ligne** : Antibiotique le plus efficace selon le modèle
2. **Lignes suivantes** : Alternatives en cas d'échec
3. **Dernière ligne** : Céfidérocol (dernier recours)

## Notes importantes
- Le modèle est basé sur les données ATLAS
- Les recommandations doivent être validées cliniquement
- La céfidérocol est réservée aux cas de résistance multiple
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Rapport généré : {output_path}")

def main():
    """Fonction principale pour exécuter le système de recommandation."""
    print("=== Système de Recommandation d'Antibiotiques ===\n")
    
    # Créer le modèle
    model = AntibioticDecisionTree()
    
    # Charger les données
    model.load_atlas_data()
    
    # Préparer les données d'antibiotiques
    antibiotic_columns = model.prepare_antibiotic_data()
    
    # Calculer les scores
    scores_df = model.calculate_antibiotic_scores(antibiotic_columns)
    
    # Déterminer l'ordre optimal
    model.determine_antibiotic_order(scores_df, antibiotic_columns)
    
    # Créer les caractéristiques
    features_df = model.create_decision_features(scores_df)
    
    # Entraîner le modèle
    X_train, X_test, y_train, y_test = model.train_decision_tree(features_df)
    
    # Visualiser l'arbre
    model.visualize_tree()
    
    # Générer le rapport
    model.generate_report()
    
    # Exemple de recommandation
    print("\n=== Exemple de Recommandation ===")
    recommendations = model.recommend_antibiotics(
        species="Escherichia coli",
        country="France",
        year=2023
    )
    
    print("Recommandations pour E. coli en France (2023) :")
    for i, ab in enumerate(recommendations, 1):
        print(f"{i}. {ab}")
    
    print("\nSystème de recommandation terminé avec succès !")

if __name__ == "__main__":
    main()
