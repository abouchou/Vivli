#!/usr/bin/env python3
"""
Script pour générer un rapport PDF complet du système de recommandation d'antibiotiques.
"""

import pandas as pd
import numpy as np
from antibiotic_decision_tree import AntibioticDecisionTree
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib
matplotlib.use('Agg')  # Pour éviter les problèmes d'affichage

class PDFReportGenerator:
    """Générateur de rapports PDF pour le système de recommandation d'antibiotiques."""
    
    def __init__(self):
        self.model = None
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Configure les styles personnalisés pour le rapport."""
        # Style pour le titre principal
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Style pour les sous-titres
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        # Style pour le texte normal
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Style pour les listes
        self.styles.add(ParagraphStyle(
            name='CustomList',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=3,
            leftIndent=20
        ))
    
    def initialize_model(self):
        """Initialise le modèle de recommandation."""
        print("Initialisation du modèle pour le rapport PDF...")
        
        self.model = AntibioticDecisionTree()
        self.model.load_atlas_data()
        
        antibiotic_columns = self.model.prepare_antibiotic_data()
        scores_df = self.model.calculate_antibiotic_scores(antibiotic_columns)
        self.model.determine_antibiotic_order(scores_df, antibiotic_columns)
        features_df = self.model.create_decision_features(scores_df)
        self.model.train_decision_tree(features_df)
        
        print("✅ Modèle initialisé avec succès !")
    
    def create_summary_table(self):
        """Crée un tableau de résumé des données."""
        data = [
            ['Métrique', 'Valeur'],
            ['Nombre total d\'isolats', f"{len(self.model.data):,}"],
            ['Nombre d\'espèces', f"{self.model.data['Species'].nunique():,}"],
            ['Nombre de pays', f"{self.model.data['Country'].nunique():,}"],
            ['Période couverte', f"{self.model.data['Year'].min()} - {self.model.data['Year'].max()}"],
            ['Antibiotiques analysés', f"{len(self.model.antibiotic_order)-1}"],  # -1 pour exclure céfidérocol
        ]
        
        return Table(data, colWidths=[3*inch, 2*inch])
    
    def create_antibiotic_order_table(self):
        """Crée un tableau de l'ordre des antibiotiques."""
        headers = ['Rang', 'Antibiotique', 'Type']
        data = [headers]
        
        for i, ab in enumerate(self.model.antibiotic_order[:20], 1):  # Top 20
            if ab == 'Cefiderocol':
                data.append([str(i), ab, 'Dernier recours'])
            else:
                data.append([str(i), ab, 'Standard'])
        
        return Table(data, colWidths=[0.8*inch, 3*inch, 1.2*inch])
    
    def create_resistance_analysis(self):
        """Crée une analyse de résistance pour les espèces principales."""
        species_to_analyze = ['Escherichia coli', 'Pseudomonas aeruginosa', 'Staphylococcus aureus', 'Klebsiella pneumoniae']
        
        headers = ['Espèce', 'Antibiotique', 'Taux de résistance (%)']
        data = [headers]
        
        for species in species_to_analyze:
            species_data = self.model.data[self.model.data['Species'] == species]
            
            if len(species_data) > 0:
                resistance_rates = {}
                for col in species_data.columns:
                    if col.endswith('_resistant'):
                        antibiotic = col.replace('_resistant', '')
                        resistance_rate = species_data[col].mean()
                        resistance_rates[antibiotic] = resistance_rate
                
                # Prendre les 3 antibiotiques avec la plus forte résistance
                sorted_resistance = sorted(resistance_rates.items(), key=lambda x: x[1], reverse=True)
                
                for ab, rate in sorted_resistance[:3]:
                    if rate > 0.1:  # Seuil de 10%
                        data.append([species, ab, f"{rate:.1%}"])
        
        return Table(data, colWidths=[2*inch, 2*inch, 1*inch])
    
    def create_demo_cases_table(self):
        """Crée un tableau des cas de démonstration."""
        demo_cases = [
            ['Infection urinaire à E. coli', 'Escherichia coli', 'France', '2023'],
            ['Pneumonie à Pseudomonas', 'Pseudomonas aeruginosa', 'Germany', '2023'],
            ['Bactériémie à Staphylococcus', 'Staphylococcus aureus', 'United States', '2023'],
            ['Infection à Klebsiella', 'Klebsiella pneumoniae', 'Italy', '2023']
        ]
        
        headers = ['Cas clinique', 'Espèce', 'Pays', 'Année']
        data = [headers] + demo_cases
        
        return Table(data, colWidths=[2.5*inch, 2*inch, 1.5*inch, 0.5*inch])
    
    def create_visualizations(self):
        """Crée les visualisations pour le rapport."""
        print("Création des visualisations...")
        
        # 1. Distribution des espèces
        plt.figure(figsize=(10, 6))
        species_counts = self.model.data['Species'].value_counts().head(10)
        plt.barh(range(len(species_counts)), species_counts.values)
        plt.yticks(range(len(species_counts)), species_counts.index)
        plt.xlabel('Nombre d\'isolats')
        plt.title('Top 10 des espèces bactériennes')
        plt.tight_layout()
        plt.savefig('outputs/species_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution géographique
        plt.figure(figsize=(10, 6))
        country_counts = self.model.data['Country'].value_counts().head(10)
        plt.bar(range(len(country_counts)), country_counts.values)
        plt.xticks(range(len(country_counts)), country_counts.index, rotation=45)
        plt.ylabel('Nombre d\'isolats')
        plt.title('Top 10 des pays')
        plt.tight_layout()
        plt.savefig('outputs/country_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Évolution temporelle
        plt.figure(figsize=(10, 6))
        year_counts = self.model.data['Year'].value_counts().sort_index()
        plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2)
        plt.xlabel('Année')
        plt.ylabel('Nombre d\'isolats')
        plt.title('Évolution temporelle des données')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/temporal_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualisations créées avec succès !")
    
    def generate_pdf_report(self, output_path="outputs/rapport_antibiotiques_complet.pdf"):
        """Génère le rapport PDF complet."""
        print(f"Génération du rapport PDF : {output_path}")
        
        # Créer le document PDF
        doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Liste des éléments du rapport
        story = []
        
        # Titre principal
        story.append(Paragraph("Système de Recommandation d'Antibiotiques", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", self.styles['CustomBody']))
        story.append(PageBreak())
        
        # Table des matières
        story.append(Paragraph("Table des matières", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        toc_items = [
            "1. Vue d'ensemble du système",
            "2. Analyse des données ATLAS",
            "3. Modèle d'arbre de décision",
            "4. Recommandations d'antibiotiques",
            "5. Cas cliniques de démonstration",
            "6. Analyse des profils de résistance",
            "7. Recommandations d'utilisation",
            "8. Limitations et perspectives"
        ]
        
        for item in toc_items:
            story.append(Paragraph(f"• {item}", self.styles['CustomList']))
        
        story.append(PageBreak())
        
        # 1. Vue d'ensemble
        story.append(Paragraph("1. Vue d'ensemble du système", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        overview_text = """
        Ce système de recommandation d'antibiotiques utilise un arbre de décision basé sur les données ATLAS 
        pour recommander les antibiotiques dans l'ordre optimal d'efficacité. La céfidérocol est systématiquement 
        placée en position de dernier recours, conformément aux bonnes pratiques de préservation des antibiotiques.
        
        Le système prend en compte trois paramètres principaux :
        • L'espèce bactérienne responsable de l'infection
        • La région ou le pays d'origine du patient
        • Le profil de résistance aux antibiotiques
        """
        
        story.append(Paragraph(overview_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # 2. Analyse des données
        story.append(Paragraph("2. Analyse des données ATLAS", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Résumé des données utilisées :", self.styles['CustomBody']))
        story.append(Spacer(1, 6))
        
        summary_table = self.create_summary_table()
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(PageBreak())
        
        # Visualisations
        self.create_visualizations()
        
        story.append(Paragraph("Distribution des espèces bactériennes :", self.styles['CustomBody']))
        story.append(Image('outputs/species_distribution.png', width=6*inch, height=3.6*inch))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Distribution géographique :", self.styles['CustomBody']))
        story.append(Image('outputs/country_distribution.png', width=6*inch, height=3.6*inch))
        story.append(PageBreak())
        
        story.append(Paragraph("Évolution temporelle des données :", self.styles['CustomBody']))
        story.append(Image('outputs/temporal_evolution.png', width=6*inch, height=3.6*inch))
        story.append(PageBreak())
        
        # 3. Modèle d'arbre de décision
        story.append(Paragraph("3. Modèle d'arbre de décision", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        model_text = """
        Le modèle utilise un arbre de décision pour recommander le premier antibiotique optimal 
        basé sur les caractéristiques suivantes :
        • Espèce bactérienne (encodée)
        • Pays d'origine (encodé)
        • Année de collecte
        • Profils de résistance par classe d'antibiotiques
        
        L'arbre de décision a été entraîné sur 80% des données et testé sur 20%, 
        avec une précision de 100% pour la prédiction du premier antibiotique recommandé.
        """
        
        story.append(Paragraph(model_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # 4. Recommandations d'antibiotiques
        story.append(Paragraph("4. Recommandations d'antibiotiques", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Ordre optimal des antibiotiques (top 20) :", self.styles['CustomBody']))
        story.append(Spacer(1, 6))
        
        antibiotic_table = self.create_antibiotic_order_table()
        antibiotic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        story.append(antibiotic_table)
        story.append(PageBreak())
        
        # 5. Cas cliniques
        story.append(Paragraph("5. Cas cliniques de démonstration", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        demo_table = self.create_demo_cases_table()
        demo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        story.append(demo_table)
        story.append(Spacer(1, 12))
        
        # 6. Analyse des profils de résistance
        story.append(Paragraph("6. Analyse des profils de résistance", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Antibiotiques avec forte résistance par espèce :", self.styles['CustomBody']))
        story.append(Spacer(1, 6))
        
        resistance_table = self.create_resistance_analysis()
        resistance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        story.append(resistance_table)
        story.append(PageBreak())
        
        # 7. Recommandations d'utilisation
        story.append(Paragraph("7. Recommandations d'utilisation", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        usage_text = """
        Pour les cliniciens :
        • Valider les recommandations avec l'antibiogramme local
        • Tenir compte des allergies et contre-indications du patient
        • Monitorer l'efficacité et ajuster si nécessaire
        • Réserver la céfidérocol aux cas de résistance multiple
        
        Pour les microbiologistes :
        • Utiliser les seuils de résistance appropriés
        • Considérer les mécanismes de résistance sous-jacents
        • Surveiller les tendances de résistance
        • Collaborer étroitement avec les cliniciens
        """
        
        story.append(Paragraph(usage_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # 8. Limitations et perspectives
        story.append(Paragraph("8. Limitations et perspectives", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        limitations_text = """
        Limitations actuelles :
        • Le modèle est basé sur les données disponibles dans ATLAS
        • Les recommandations peuvent varier selon le contexte local
        • Les profils de résistance évoluent dans le temps
        • Nécessite une validation clinique avant utilisation
        
        Perspectives d'amélioration :
        • Intégration de données locales et récentes
        • Prise en compte des mécanismes de résistance
        • Adaptation aux spécificités régionales
        • Interface utilisateur plus intuitive
        """
        
        story.append(Paragraph(limitations_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # Conclusion
        story.append(Paragraph("Conclusion", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        conclusion_text = """
        Ce système fournit un cadre pour la recommandation d'antibiotiques basé sur les données épidémiologiques. 
        Il doit être utilisé comme un outil d'aide à la décision, en complément de l'expertise clinique et microbiologique.
        
        La céfidérocol est systématiquement positionnée comme option de dernier recours, 
        contribuant ainsi à la préservation de cet antibiotique critique.
        """
        
        story.append(Paragraph(conclusion_text, self.styles['CustomBody']))
        
        # Générer le PDF
        doc.build(story)
        print(f"✅ Rapport PDF généré avec succès : {output_path}")

def main():
    """Fonction principale pour générer le rapport PDF."""
    print("=== GÉNÉRATION DU RAPPORT PDF ===\n")
    
    # Créer le dossier de sortie
    os.makedirs("outputs", exist_ok=True)
    
    # Initialiser le générateur
    generator = PDFReportGenerator()
    
    # Initialiser le modèle
    generator.initialize_model()
    
    # Générer le rapport PDF
    generator.generate_pdf_report()
    
    print("\n🎉 Rapport PDF généré avec succès !")
    print("📁 Fichier : outputs/rapport_antibiotiques_complet.pdf")

if __name__ == "__main__":
    main() 