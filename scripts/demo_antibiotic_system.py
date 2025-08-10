#!/usr/bin/env python3
"""
Script de démonstration du système de recommandation d'antibiotiques.
Teste le système avec différents cas cliniques réels.
"""

import pandas as pd
import numpy as np
from antibiotic_decision_tree import AntibioticDecisionTree
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def run_demo():
    """Exécute une démonstration complète du système."""
    print("=== DÉMONSTRATION DU SYSTÈME DE RECOMMANDATION D'ANTIBIOTIQUES ===\n")
    
    # Créer le modèle
    print("1. Initialisation du modèle...")
    model = AntibioticDecisionTree()
    
    # Charger les données
    model.load_atlas_data()
    
    # Préparer et entraîner le modèle
    antibiotic_columns = model.prepare_antibiotic_data()
    scores_df = model.calculate_antibiotic_scores(antibiotic_columns)
    model.determine_antibiotic_order(scores_df, antibiotic_columns)
    features_df = model.create_decision_features(scores_df)
    model.train_decision_tree(features_df)
    
    print("✅ Modèle entraîné avec succès !\n")
    
    # Cas de démonstration
    demo_cases = [
        {
            'name': 'Infection urinaire à E. coli',
            'species': 'Escherichia coli',
            'country': 'France',
            'year': 2023,
            'description': 'Patient de 45 ans avec infection urinaire compliquée'
        },
        {
            'name': 'Pneumonie à Pseudomonas',
            'species': 'Pseudomonas aeruginosa',
            'country': 'Germany',
            'year': 2023,
            'description': 'Patient de 67 ans avec pneumonie nosocomiale'
        },
        {
            'name': 'Bactériémie à Staphylococcus',
            'species': 'Staphylococcus aureus',
            'country': 'United States',
            'year': 2023,
            'description': 'Patient de 28 ans avec bactériémie communautaire'
        },
        {
            'name': 'Infection à Klebsiella',
            'species': 'Klebsiella pneumoniae',
            'country': 'Italy',
            'year': 2023,
            'description': 'Patient de 52 ans avec infection abdominale post-opératoire'
        }
    ]
    
    # Générer les recommandations pour chaque cas
    print("2. Génération des recommandations pour différents cas cliniques...\n")
    
    for i, case in enumerate(demo_cases, 1):
        print(f"--- CAS {i}: {case['name']} ---")
        print(f"Description: {case['description']}")
        print(f"Espèce: {case['species']}")
        print(f"Pays: {case['country']}")
        print(f"Année: {case['year']}")
        
        try:
            recommendations = model.recommend_antibiotics(
                species=case['species'],
                country=case['country'],
                year=case['year']
            )
            
            print("\nRecommandations d'antibiotiques :")
            for j, ab in enumerate(recommendations[:10], 1):  # Afficher les 10 premiers
                if ab == 'Cefiderocol':
                    print(f"{j}. {ab} (dernier recours)")
                else:
                    print(f"{j}. {ab}")
            
            if len(recommendations) > 10:
                print(f"... et {len(recommendations) - 10} autres antibiotiques")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
        print("\n" + "-"*60 + "\n")
    
    # Analyser les profils de résistance
    print("3. Analyse des profils de résistance par espèce...\n")
    
    species_to_analyze = ['Escherichia coli', 'Pseudomonas aeruginosa', 'Staphylococcus aureus', 'Klebsiella pneumoniae']
    
    for species in species_to_analyze:
        print(f"--- Profil de résistance: {species} ---")
        
        # Filtrer les données pour l'espèce
        species_data = model.data[model.data['Species'] == species]
        
        if len(species_data) > 0:
            # Calculer les taux de résistance
            resistance_rates = {}
            for col in species_data.columns:
                if col.endswith('_resistant'):
                    antibiotic = col.replace('_resistant', '')
                    resistance_rate = species_data[col].mean()
                    resistance_rates[antibiotic] = resistance_rate
            
            # Afficher les antibiotiques avec la plus forte résistance
            sorted_resistance = sorted(resistance_rates.items(), key=lambda x: x[1], reverse=True)
            
            print("Antibiotiques avec forte résistance (>50%):")
            for ab, rate in sorted_resistance[:5]:
                if rate > 0.5:
                    print(f"  - {ab}: {rate:.1%}")
            
            print("Antibiotiques avec résistance modérée (20-50%):")
            for ab, rate in sorted_resistance[:5]:
                if 0.2 <= rate <= 0.5:
                    print(f"  - {ab}: {rate:.1%}")
        else:
            print(f"Aucune donnée trouvée pour {species}")
        
        print()
    
    # Générer un rapport de synthèse
    print("4. Génération du rapport de synthèse...\n")
    
    generate_summary_report(model, demo_cases)
    
    print("✅ Démonstration terminée avec succès !")
    print("\nFichiers générés dans le dossier 'outputs/':")
    print("- antibiotic_decision_tree_report.md")
    print("- antibiotic_decision_tree.png")
    print("- demo_summary_report.md")

def generate_summary_report(model, demo_cases):
    """Génère un rapport de synthèse de la démonstration."""
    
    report = f"""# Rapport de Synthèse - Démonstration du Système de Recommandation d'Antibiotiques

## Date de génération
{datetime.now().strftime('%d/%m/%Y à %H:%M')}

## Vue d'ensemble
Ce rapport présente les résultats de la démonstration du système de recommandation d'antibiotiques basé sur les données ATLAS.

## Cas cliniques testés

"""
    
    for i, case in enumerate(demo_cases, 1):
        report += f"""### Cas {i}: {case['name']}
- **Description**: {case['description']}
- **Espèce**: {case['species']}
- **Pays**: {case['country']}
- **Année**: {case['year']}

"""
        
        try:
            recommendations = model.recommend_antibiotics(
                species=case['species'],
                country=case['country'],
                year=case['year']
            )
            
            report += "**Recommandations principales :**\n"
            for j, ab in enumerate(recommendations[:5], 1):
                if ab == 'Cefiderocol':
                    report += f"{j}. {ab} (dernier recours)\n"
                else:
                    report += f"{j}. {ab}\n"
            
            if len(recommendations) > 5:
                report += f"... et {len(recommendations) - 5} autres options\n"
            
        except Exception as e:
            report += f"Erreur: {e}\n"
        
        report += "\n"
    
    # Statistiques globales
    report += """## Statistiques globales

### Espèces bactériennes les plus fréquentes
"""
    
    species_counts = model.data['Species'].value_counts()
    for species, count in species_counts.head(10).items():
        report += f"- {species}: {count:,} isolats\n"
    
    report += f"""

### Pays avec le plus de données
"""
    
    country_counts = model.data['Country'].value_counts()
    for country, count in country_counts.head(10).items():
        report += f"- {country}: {count:,} isolats\n"
    
    report += f"""

### Distribution temporelle
"""
    
    year_counts = model.data['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        report += f"- {year}: {count:,} isolats\n"
    
    report += """

## Recommandations d'utilisation

### Pour les cliniciens
1. **Validation clinique** : Toujours valider les recommandations avec l'antibiogramme local
2. **Considérations patient** : Tenir compte des allergies, contre-indications et fonction rénale/hépatique
3. **Surveillance** : Monitorer l'efficacité et ajuster si nécessaire
4. **Dernier recours** : La céfidérocol doit être réservée aux cas de résistance multiple

### Pour les microbiologistes
1. **Interprétation des MIC** : Utiliser les seuils de résistance appropriés
2. **Mécanismes de résistance** : Considérer les mécanismes sous-jacents
3. **Évolution temporelle** : Surveiller les tendances de résistance
4. **Collaboration** : Travailler en étroite collaboration avec les cliniciens

## Limitations

1. **Données ATLAS** : Le modèle est basé sur les données disponibles dans ATLAS
2. **Généralisation** : Les recommandations peuvent varier selon le contexte local
3. **Évolution** : Les profils de résistance évoluent dans le temps
4. **Validation** : Nécessite une validation clinique avant utilisation

## Conclusion

Ce système fournit un cadre pour la recommandation d'antibiotiques basé sur les données épidémiologiques. Il doit être utilisé comme un outil d'aide à la décision, en complément de l'expertise clinique et microbiologique.

---
*Généré automatiquement par le système de recommandation d'antibiotiques*
"""
    
    # Sauvegarder le rapport
    output_path = "outputs/demo_summary_report.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Rapport de synthèse généré : {output_path}")

def create_visualization_dashboard():
    """Crée un tableau de bord de visualisation."""
    print("5. Création du tableau de bord de visualisation...\n")
    
    # Charger les données
    model = AntibioticDecisionTree()
    model.load_atlas_data()
    
    # Créer les visualisations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution des espèces
    species_counts = model.data['Species'].value_counts().head(10)
    axes[0, 0].barh(range(len(species_counts)), species_counts.values)
    axes[0, 0].set_yticks(range(len(species_counts)))
    axes[0, 0].set_yticklabels(species_counts.index, fontsize=8)
    axes[0, 0].set_xlabel('Nombre d\'isolats')
    axes[0, 0].set_title('Top 10 des espèces bactériennes')
    
    # 2. Distribution géographique
    country_counts = model.data['Country'].value_counts().head(10)
    axes[0, 1].bar(range(len(country_counts)), country_counts.values)
    axes[0, 1].set_xticks(range(len(country_counts)))
    axes[0, 1].set_xticklabels(country_counts.index, rotation=45, fontsize=8)
    axes[0, 1].set_ylabel('Nombre d\'isolats')
    axes[0, 1].set_title('Top 10 des pays')
    
    # 3. Évolution temporelle
    year_counts = model.data['Year'].value_counts().sort_index()
    axes[1, 0].plot(year_counts.index, year_counts.values, marker='o')
    axes[1, 0].set_xlabel('Année')
    axes[1, 0].set_ylabel('Nombre d\'isolats')
    axes[1, 0].set_title('Évolution temporelle des données')
    
    # 4. Résistance globale (exemple avec quelques antibiotiques)
    antibiotic_columns = ['Meropenem', 'Ciprofloxacin', 'Colistin', 'Vancomycin']
    resistance_rates = {}
    
    for ab in antibiotic_columns:
        if ab in model.data.columns:
            resistant_col = f"{ab}_resistant"
            if resistant_col in model.data.columns:
                resistance_rates[ab] = model.data[resistant_col].mean()
    
    if resistance_rates:
        axes[1, 1].bar(range(len(resistance_rates)), list(resistance_rates.values()))
        axes[1, 1].set_xticks(range(len(resistance_rates)))
        axes[1, 1].set_xticklabels(resistance_rates.keys(), rotation=45)
        axes[1, 1].set_ylabel('Taux de résistance')
        axes[1, 1].set_title('Résistance globale par antibiotique')
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('outputs/dashboard_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Tableau de bord créé : outputs/dashboard_visualization.png")

if __name__ == "__main__":
    # Créer le dossier de sortie
    os.makedirs("outputs", exist_ok=True)
    
    # Exécuter la démonstration
    run_demo()
    
    # Créer le tableau de bord
    create_visualization_dashboard()
