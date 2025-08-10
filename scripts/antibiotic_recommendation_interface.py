import pandas as pd
import numpy as np
from antibiotic_decision_tree import AntibioticDecisionTree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class AntibioticRecommendationInterface:
    """
    Interface utilisateur pour le système de recommandation d'antibiotiques.
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def initialize_model(self):
        """Initialise et entraîne le modèle."""
        print("=== Initialisation du Système de Recommandation ===\n")
        
        # Créer le modèle
        self.model = AntibioticDecisionTree()
        
        # Charger et préparer les données
        print("Chargement des données ATLAS...")
        self.model.load_atlas_data()
        
        # Préparer les données d'antibiotiques
        antibiotic_columns = self.model.prepare_antibiotic_data()
        
        # Calculer les scores
        scores_df = self.model.calculate_antibiotic_scores(antibiotic_columns)
        
        # Déterminer l'ordre optimal
        self.model.determine_antibiotic_order(scores_df, antibiotic_columns)
        
        # Créer les caractéristiques et entraîner
        features_df = self.model.create_decision_features(scores_df)
        self.model.train_decision_tree(features_df)
        
        self.is_trained = True
        print("\n✅ Modèle entraîné avec succès !")
        
    def get_available_species(self):
        """Retourne la liste des espèces disponibles."""
        if self.model is None:
            return []
        return sorted(self.model.data['Species'].unique())
    
    def get_available_countries(self):
        """Retourne la liste des pays disponibles."""
        if self.model is None:
            return []
        return sorted(self.model.data['Country'].unique())
    
    def get_recommendations(self, species, country, year, resistance_profile=None):
        """Obtient des recommandations d'antibiotiques."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations.")
        
        try:
            recommendations = self.model.recommend_antibiotics(
                species=species,
                country=country,
                year=year,
                resistance_profile=resistance_profile
            )
            return recommendations
        except Exception as e:
            print(f"Erreur lors de la génération des recommandations : {e}")
            return []
    
    def analyze_species_resistance(self, species):
        """Analyse le profil de résistance d'une espèce bactérienne."""
        if self.model is None:
            return None
        
        # Filtrer les données pour l'espèce
        species_data = self.model.data[self.model.data['Species'] == species]
        
        if len(species_data) == 0:
            print(f"Aucune donnée trouvée pour {species}")
            return None
        
        # Calculer les taux de résistance par antibiotique
        resistance_rates = {}
        for col in species_data.columns:
            if col.endswith('_resistant'):
                antibiotic = col.replace('_resistant', '')
                resistance_rate = species_data[col].mean()
                resistance_rates[antibiotic] = resistance_rate
        
        return resistance_rates
    
    def visualize_resistance_profile(self, species, output_path=None):
        """Visualise le profil de résistance d'une espèce."""
        resistance_rates = self.analyze_species_resistance(species)
        
        if resistance_rates is None:
            return
        
        # Trier par taux de résistance
        sorted_resistance = sorted(resistance_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Prendre les 20 antibiotiques avec la plus forte résistance
        top_resistant = sorted_resistance[:20]
        
        plt.figure(figsize=(12, 8))
        antibiotics, rates = zip(*top_resistant)
        
        colors = ['red' if rate > 0.5 else 'orange' if rate > 0.2 else 'green' for rate in rates]
        
        bars = plt.barh(range(len(antibiotics)), rates, color=colors)
        plt.yticks(range(len(antibiotics)), antibiotics)
        plt.xlabel('Taux de résistance')
        plt.title(f'Profil de résistance - {species}')
        plt.xlim(0, 1)
        
        # Ajouter des annotations
        for i, (antibiotic, rate) in enumerate(top_resistant):
            plt.text(rate + 0.01, i, f'{rate:.1%}', va='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé : {output_path}")
        else:
            plt.show()
        plt.close()
    
    def generate_detailed_report(self, species, country, year, output_path=None):
        """Génère un rapport détaillé pour un cas spécifique."""
        recommendations = self.get_recommendations(species, country, year)
        resistance_rates = self.analyze_species_resistance(species)
        
        report = f"""# Rapport de Recommandation d'Antibiotiques

## Informations du cas
- **Espèce bactérienne** : {species}
- **Pays/Région** : {country}
- **Année** : {year}

## Recommandations d'antibiotiques

### Ordre de traitement recommandé :
"""
        
        for i, ab in enumerate(recommendations, 1):
            if ab == 'Cefiderocol':
                report += f"{i}. **{ab}** (dernier recours)\n"
            else:
                report += f"{i}. {ab}\n"
        
        if resistance_rates:
            report += f"""

## Analyse du profil de résistance pour {species}

### Antibiotiques avec forte résistance (>50%) :
"""
            high_resistance = {k: v for k, v in resistance_rates.items() if v > 0.5}
            for ab, rate in sorted(high_resistance.items(), key=lambda x: x[1], reverse=True):
                report += f"- {ab} : {rate:.1%}\n"
            
            report += f"""

### Antibiotiques avec résistance modérée (20-50%) :
"""
            moderate_resistance = {k: v for k, v in resistance_rates.items() if 0.2 <= v <= 0.5}
            for ab, rate in sorted(moderate_resistance.items(), key=lambda x: x[1], reverse=True):
                report += f"- {ab} : {rate:.1%}\n"
        
        report += f"""

## Recommandations cliniques

1. **Première ligne** : {recommendations[0] if recommendations else 'Non disponible'}
2. **Alternative en cas d'échec** : {recommendations[1] if len(recommendations) > 1 else 'Non disponible'}
3. **Dernier recours** : Cefiderocol

## Notes importantes
- Ces recommandations sont basées sur les données ATLAS
- La validation clinique est nécessaire avant prescription
- Considérer les allergies et contre-indications du patient
- Adapter la posologie selon la fonction rénale et hépatique
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Rapport détaillé généré : {output_path}")
        else:
            print(report)
        
        return report

def interactive_menu():
    """Interface utilisateur interactive."""
    interface = AntibioticRecommendationInterface()
    
    print("=== Système de Recommandation d'Antibiotiques ===\n")
    
    while True:
        print("\n" + "="*50)
        print("MENU PRINCIPAL")
        print("="*50)
        print("1. Initialiser/Entraîner le modèle")
        print("2. Obtenir des recommandations")
        print("3. Analyser le profil de résistance d'une espèce")
        print("4. Générer un rapport détaillé")
        print("5. Visualiser l'arbre de décision")
        print("6. Quitter")
        print("="*50)
        
        choice = input("\nChoisissez une option (1-6) : ").strip()
        
        if choice == '1':
            try:
                interface.initialize_model()
            except Exception as e:
                print(f"Erreur lors de l'initialisation : {e}")
        
        elif choice == '2':
            if not interface.is_trained:
                print("❌ Le modèle doit être entraîné d'abord (option 1)")
                continue
            
            print("\n--- Recommandations d'antibiotiques ---")
            species = input("Espèce bactérienne (ex: Escherichia coli) : ").strip()
            country = input("Pays (ex: France) : ").strip()
            year = input("Année (ex: 2023) : ").strip()
            
            try:
                year = int(year)
                recommendations = interface.get_recommendations(species, country, year)
                
                if recommendations:
                    print(f"\n✅ Recommandations pour {species} en {country} ({year}) :")
                    for i, ab in enumerate(recommendations, 1):
                        if ab == 'Cefiderocol':
                            print(f"{i}. {ab} (dernier recours)")
                        else:
                            print(f"{i}. {ab}")
                else:
                    print("❌ Aucune recommandation générée")
                    
            except ValueError:
                print("❌ Année invalide")
            except Exception as e:
                print(f"❌ Erreur : {e}")
        
        elif choice == '3':
            if not interface.is_trained:
                print("❌ Le modèle doit être entraîné d'abord (option 1)")
                continue
            
            print("\n--- Analyse du profil de résistance ---")
            species = input("Espèce bactérienne : ").strip()
            
            try:
                resistance_rates = interface.analyze_species_resistance(species)
                if resistance_rates:
                    print(f"\n📊 Profil de résistance pour {species} :")
                    sorted_resistance = sorted(resistance_rates.items(), key=lambda x: x[1], reverse=True)
                    
                    print("\nAntibiotiques avec forte résistance (>50%) :")
                    for ab, rate in sorted_resistance[:10]:
                        if rate > 0.5:
                            print(f"  - {ab} : {rate:.1%}")
                    
                    print("\nAntibiotiques avec résistance modérée (20-50%) :")
                    for ab, rate in sorted_resistance[:10]:
                        if 0.2 <= rate <= 0.5:
                            print(f"  - {ab} : {rate:.1%}")
                else:
                    print("❌ Aucune donnée trouvée pour cette espèce")
                    
            except Exception as e:
                print(f"❌ Erreur : {e}")
        
        elif choice == '4':
            if not interface.is_trained:
                print("❌ Le modèle doit être entraîné d'abord (option 1)")
                continue
            
            print("\n--- Génération de rapport détaillé ---")
            species = input("Espèce bactérienne : ").strip()
            country = input("Pays : ").strip()
            year = input("Année : ").strip()
            
            try:
                year = int(year)
                output_path = f"outputs/rapport_{species.replace(' ', '_')}_{country}_{year}.md"
                interface.generate_detailed_report(species, country, year, output_path)
                
            except ValueError:
                print("❌ Année invalide")
            except Exception as e:
                print(f"❌ Erreur : {e}")
        
        elif choice == '5':
            if not interface.is_trained:
                print("❌ Le modèle doit être entraîné d'abord (option 1)")
                continue
            
            try:
                interface.model.visualize_tree()
                print("✅ Arbre de décision visualisé")
            except Exception as e:
                print(f"❌ Erreur : {e}")
        
        elif choice == '6':
            print("\n👋 Au revoir !")
            break
        
        else:
            print("❌ Option invalide. Veuillez choisir 1-6.")

if __name__ == "__main__":
    interactive_menu()
