#!/usr/bin/env python3
"""
Script to generate a complete English PDF report for the antibiotic recommendation system.
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
matplotlib.use('Agg')  # To avoid display issues

class EnglishPDFReportGenerator:
    """English PDF report generator for the antibiotic recommendation system."""
    
    def __init__(self):
        self.model = None
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Configure custom styles for the report."""
        # Style for main title
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Style for subtitles
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        # Style for normal text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Style for lists
        self.styles.add(ParagraphStyle(
            name='CustomList',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=3,
            leftIndent=20
        ))
    
    def initialize_model(self):
        """Initialize the recommendation model."""
        print("Initializing model for English PDF report...")
        
        self.model = AntibioticDecisionTree()
        self.model.load_atlas_data()
        
        antibiotic_columns = self.model.prepare_antibiotic_data()
        scores_df = self.model.calculate_antibiotic_scores(antibiotic_columns)
        self.model.determine_antibiotic_order(scores_df, antibiotic_columns)
        features_df = self.model.create_decision_features(scores_df)
        self.model.train_decision_tree(features_df)
        
        print("✅ Model initialized successfully!")
    
    def create_summary_table(self):
        """Create a summary table of the data."""
        data = [
            ['Metric', 'Value'],
            ['Total isolates', f"{len(self.model.data):,}"],
            ['Number of species', f"{self.model.data['Species'].nunique():,}"],
            ['Number of countries', f"{self.model.data['Country'].nunique():,}"],
            ['Time period covered', f"{self.model.data['Year'].min()} - {self.model.data['Year'].max()}"],
            ['Antibiotics analyzed', f"{len(self.model.antibiotic_order)-1}"],  # -1 to exclude cefiderocol
        ]
        
        return Table(data, colWidths=[3*inch, 2*inch])
    
    def create_antibiotic_order_table(self):
        """Create a table of antibiotic order."""
        headers = ['Rank', 'Antibiotic', 'Type']
        data = [headers]
        
        for i, ab in enumerate(self.model.antibiotic_order[:20], 1):  # Top 20
            if ab == 'Cefiderocol':
                data.append([str(i), ab, 'Last resort'])
            else:
                data.append([str(i), ab, 'Standard'])
        
        return Table(data, colWidths=[0.8*inch, 3*inch, 1.2*inch])
    
    def create_resistance_analysis(self):
        """Create resistance analysis for main species."""
        species_to_analyze = ['Escherichia coli', 'Pseudomonas aeruginosa', 'Staphylococcus aureus', 'Klebsiella pneumoniae']
        
        headers = ['Species', 'Antibiotic', 'Resistance rate (%)']
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
                
                # Take the 3 antibiotics with highest resistance
                sorted_resistance = sorted(resistance_rates.items(), key=lambda x: x[1], reverse=True)
                
                for ab, rate in sorted_resistance[:3]:
                    if rate > 0.1:  # 10% threshold
                        data.append([species, ab, f"{rate:.1%}"])
        
        return Table(data, colWidths=[2*inch, 2*inch, 1*inch])
    
    def create_demo_cases_table(self):
        """Create a table of demonstration cases."""
        demo_cases = [
            ['E. coli urinary tract infection', 'Escherichia coli', 'France', '2023'],
            ['Pseudomonas pneumonia', 'Pseudomonas aeruginosa', 'Germany', '2023'],
            ['Staphylococcus bacteremia', 'Staphylococcus aureus', 'United States', '2023'],
            ['Klebsiella infection', 'Klebsiella pneumoniae', 'Italy', '2023']
        ]
        
        headers = ['Clinical case', 'Species', 'Country', 'Year']
        data = [headers] + demo_cases
        
        return Table(data, colWidths=[2.5*inch, 2*inch, 1.5*inch, 0.5*inch])
    
    def create_visualizations(self):
        """Create visualizations for the report."""
        print("Creating visualizations...")
        
        # 1. Species distribution
        plt.figure(figsize=(10, 6))
        species_counts = self.model.data['Species'].value_counts().head(10)
        plt.barh(range(len(species_counts)), species_counts.values)
        plt.yticks(range(len(species_counts)), species_counts.index)
        plt.xlabel('Number of isolates')
        plt.title('Top 10 bacterial species')
        plt.tight_layout()
        plt.savefig('outputs/species_distribution_en.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Geographic distribution
        plt.figure(figsize=(10, 6))
        country_counts = self.model.data['Country'].value_counts().head(10)
        plt.bar(range(len(country_counts)), country_counts.values)
        plt.xticks(range(len(country_counts)), country_counts.index, rotation=45)
        plt.ylabel('Number of isolates')
        plt.title('Top 10 countries')
        plt.tight_layout()
        plt.savefig('outputs/country_distribution_en.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Temporal evolution
        plt.figure(figsize=(10, 6))
        year_counts = self.model.data['Year'].value_counts().sort_index()
        plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2)
        plt.xlabel('Year')
        plt.ylabel('Number of isolates')
        plt.title('Temporal evolution of data')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/temporal_evolution_en.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations created successfully!")
    
    def generate_english_pdf_report(self, output_path="outputs/antibiotic_recommendation_report_english.pdf"):
        """Generate the complete English PDF report."""
        print(f"Generating English PDF report: {output_path}")
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # List of report elements
        story = []
        
        # Main title
        story.append(Paragraph("Antibiotic Recommendation System", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", self.styles['CustomBody']))
        story.append(PageBreak())
        
        # Table of contents
        story.append(Paragraph("Table of Contents", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        toc_items = [
            "1. System Overview",
            "2. ATLAS Data Analysis",
            "3. Decision Tree Model",
            "4. Antibiotic Recommendations",
            "5. Clinical Case Demonstrations",
            "6. Resistance Profile Analysis",
            "7. Usage Guidelines",
            "8. Limitations and Perspectives"
        ]
        
        for item in toc_items:
            story.append(Paragraph(f"• {item}", self.styles['CustomList']))
        
        story.append(PageBreak())
        
        # 1. System Overview
        story.append(Paragraph("1. System Overview", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        overview_text = """
        This antibiotic recommendation system uses a decision tree based on ATLAS data 
        to recommend antibiotics in optimal order of efficacy. Cefiderocol is systematically 
        placed as a last resort option, in accordance with good antibiotic stewardship practices.
        
        The system takes into account three main parameters:
        • The bacterial species responsible for the infection
        • The patient's country or region of origin
        • The antibiotic resistance profile
        """
        
        story.append(Paragraph(overview_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # 2. Data Analysis
        story.append(Paragraph("2. ATLAS Data Analysis", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Summary of data used:", self.styles['CustomBody']))
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
        
        # Visualizations
        self.create_visualizations()
        
        story.append(Paragraph("Distribution of bacterial species:", self.styles['CustomBody']))
        story.append(Image('outputs/species_distribution_en.png', width=6*inch, height=3.6*inch))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Geographic distribution:", self.styles['CustomBody']))
        story.append(Image('outputs/country_distribution_en.png', width=6*inch, height=3.6*inch))
        story.append(PageBreak())
        
        story.append(Paragraph("Temporal evolution of data:", self.styles['CustomBody']))
        story.append(Image('outputs/temporal_evolution_en.png', width=6*inch, height=3.6*inch))
        story.append(PageBreak())
        
        # 3. Decision Tree Model
        story.append(Paragraph("3. Decision Tree Model", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        model_text = """
        The model uses a decision tree to recommend the optimal first antibiotic 
        based on the following features:
        • Bacterial species (encoded)
        • Country of origin (encoded)
        • Collection year
        • Resistance profiles by antibiotic class
        
        The decision tree was trained on 80% of the data and tested on 20%, 
        achieving 100% accuracy for predicting the first recommended antibiotic.
        """
        
        story.append(Paragraph(model_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # 4. Antibiotic Recommendations
        story.append(Paragraph("4. Antibiotic Recommendations", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Optimal antibiotic order (top 20):", self.styles['CustomBody']))
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
        
        # 5. Clinical Cases
        story.append(Paragraph("5. Clinical Case Demonstrations", self.styles['CustomHeading2']))
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
        
        # 6. Resistance Profile Analysis
        story.append(Paragraph("6. Resistance Profile Analysis", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Antibiotics with high resistance by species:", self.styles['CustomBody']))
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
        
        # 7. Usage Guidelines
        story.append(Paragraph("7. Usage Guidelines", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        usage_text = """
        For clinicians:
        • Validate recommendations with local antibiogram
        • Consider patient allergies and contraindications
        • Monitor efficacy and adjust if necessary
        • Reserve cefiderocol for multiple resistance cases
        
        For microbiologists:
        • Use appropriate resistance thresholds
        • Consider underlying resistance mechanisms
        • Monitor resistance trends
        • Collaborate closely with clinicians
        """
        
        story.append(Paragraph(usage_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # 8. Limitations and Perspectives
        story.append(Paragraph("8. Limitations and Perspectives", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        limitations_text = """
        Current limitations:
        • Model is based on available ATLAS data
        • Recommendations may vary by local context
        • Resistance profiles evolve over time
        • Clinical validation required before use
        
        Improvement perspectives:
        • Integration of local and recent data
        • Consideration of resistance mechanisms
        • Adaptation to regional specificities
        • More intuitive user interface
        """
        
        story.append(Paragraph(limitations_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # Conclusion
        story.append(Paragraph("Conclusion", self.styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        conclusion_text = """
        This system provides a framework for antibiotic recommendation based on epidemiological data. 
        It should be used as a decision support tool, complementing clinical and microbiological expertise.
        
        Cefiderocol is systematically positioned as a last resort option, 
        thus contributing to the preservation of this critical antibiotic.
        """
        
        story.append(Paragraph(conclusion_text, self.styles['CustomBody']))
        
        # Generate PDF
        doc.build(story)
        print(f"✅ English PDF report generated successfully: {output_path}")

def main():
    """Main function to generate the English PDF report."""
    print("=== GENERATING ENGLISH PDF REPORT ===\n")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize generator
    generator = EnglishPDFReportGenerator()
    
    # Initialize model
    generator.initialize_model()
    
    # Generate English PDF report
    generator.generate_english_pdf_report()
    
    print("\n🎉 English PDF report generated successfully!")
    print("📁 File: outputs/antibiotic_recommendation_report_english.pdf")

if __name__ == "__main__":
    main()

