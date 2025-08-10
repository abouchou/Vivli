import markdown
import weasyprint
import os
from datetime import datetime

def create_html_from_markdown(markdown_file, html_file):
    """Convert Markdown file to HTML with styling."""
    
    # Read markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    # Create styled HTML
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vivli Project Report</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 40px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                font-size: 2.5em;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
                margin-top: 30px;
                font-size: 1.8em;
            }}
            h3 {{
                color: #2c3e50;
                font-size: 1.4em;
                margin-top: 25px;
            }}
            h4 {{
                color: #34495e;
                font-size: 1.2em;
                margin-top: 20px;
            }}
            p {{
                margin-bottom: 15px;
                text-align: justify;
            }}
            ul, ol {{
                margin-bottom: 15px;
                padding-left: 30px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            strong {{
                color: #2c3e50;
                font-weight: bold;
            }}
            em {{
                color: #7f8c8d;
                font-style: italic;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                color: #e74c3c;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
                font-style: italic;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 2px solid #ecf0f1;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #ecf0f1;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .highlight {{
                background-color: #fff3cd;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
                margin: 20px 0;
            }}
            .success {{
                background-color: #d4edda;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
                margin: 20px 0;
            }}
            .warning {{
                background-color: #f8d7da;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #dc3545;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Vivli Project Report</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </div>
        
        {html_content}
        
        <div class="footer">
            <p>Vivli Project - Antimicrobial Resistance Analysis</p>
            <p>This report was automatically generated from the analysis of SIDERO-WT and ATLAS datasets.</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    print(f"HTML file created: {html_file}")

def convert_html_to_pdf(html_file, pdf_file):
    """Convert HTML file to PDF using weasyprint."""
    try:
        # Read HTML file
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Convert to PDF
        weasyprint.HTML(string=html_content).write_pdf(pdf_file)
        print(f"PDF file created: {pdf_file}")
        
    except Exception as e:
        print(f"Error creating PDF: {e}")
        print("Please make sure weasyprint is installed on your system.")
        print("You can install it with: pip install weasyprint")

def main():
    """Generate PDF reports for steps 3 and 4."""
    print("=== Generating PDF Reports for Steps 3 and 4 ===\n")
    
    # Create reports directory if it doesn't exist
    os.makedirs("outputs/reports", exist_ok=True)
    
    # Step 3 Report
    print("Processing Step 3 Report...")
    step3_md = "outputs/step3_report.md"
    step3_html = "outputs/reports/step3_report.html"
    step3_pdf = "outputs/reports/step3_phenotypic_signatures.pdf"
    
    if os.path.exists(step3_md):
        create_html_from_markdown(step3_md, step3_html)
        convert_html_to_pdf(step3_html, step3_pdf)
    else:
        print(f"Warning: {step3_md} not found")
    
    # Step 4 Report
    print("\nProcessing Step 4 Report...")
    step4_md = "outputs/step4_report.md"
    step4_html = "outputs/reports/step4_report.html"
    step4_pdf = "outputs/reports/step4_cefiderocol_prediction.pdf"
    
    if os.path.exists(step4_md):
        create_html_from_markdown(step4_md, step4_html)
        convert_html_to_pdf(step4_html, step4_pdf)
    else:
        print(f"Warning: {step4_md} not found")
    
    # Combined Report
    print("\nCreating Combined Report...")
    combined_content = ""
    
    # Add Step 3 content
    if os.path.exists(step3_md):
        with open(step3_md, 'r', encoding='utf-8') as f:
            combined_content += "# Step 3: Discovery of Phenotypic Signatures\n\n"
            combined_content += f.read()
            combined_content += "\n\n---\n\n"
    
    # Add Step 4 content
    if os.path.exists(step4_md):
        with open(step4_md, 'r', encoding='utf-8') as f:
            combined_content += f.read()
    
    # Save combined markdown
    combined_md = "outputs/reports/combined_steps_3_4.md"
    with open(combined_md, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    # Convert combined report to PDF
    combined_html = "outputs/reports/combined_report.html"
    combined_pdf = "outputs/reports/combined_steps_3_4_report.pdf"
    
    create_html_from_markdown(combined_md, combined_html)
    convert_html_to_pdf(combined_html, combined_pdf)
    
    print("\n=== PDF Generation Complete ===")
    print("Generated files:")
    print(f"- Step 3: {step3_pdf}")
    print(f"- Step 4: {step4_pdf}")
    print(f"- Combined: {combined_pdf}")
    print("\nFiles are located in: outputs/reports/")

if __name__ == "__main__":
    main() 