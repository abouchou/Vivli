#!/usr/bin/env python3
"""
Script to convert vivli_complete_methodology.md to HTML with professional styling.
"""

import markdown
import os
from datetime import datetime

def convert_methodology_to_html():
    """Convert the complete methodology markdown file to HTML with professional styling."""
    
    # Read the markdown file
    with open('vivli_complete_methodology.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
    
    # Create professional HTML template
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vivli System - Complete Methodology</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #27ae60;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        h3 {{
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}
        
        h4 {{
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        ul, ol {{
            margin-bottom: 20px;
            padding-left: 30px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            color: #e74c3c;
        }}
        
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.4;
        }}
        
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        th {{
            background-color: #27ae60;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e8f5e8;
        }}
        
        .highlight {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .info-box {{
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .warning-box {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .success-box {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            border-radius: 10px;
        }}
        
        .header h1 {{
            color: white;
            border-bottom: none;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #ecf0f1;
            margin-bottom: 0;
        }}
        
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #27ae60;
        }}
        
        .toc h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .toc li {{
            margin-bottom: 5px;
        }}
        
        .toc a {{
            color: #27ae60;
            text-decoration: none;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
        }}
        
        .step-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #27ae60;
        }}
        
        .step-box h3 {{
            color: #27ae60;
            margin-top: 0;
        }}
        
        .methodology-overview {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .methodology-overview h3 {{
            color: white;
            margin-top: 0;
        }}
        
        .methodology-overview ol {{
            color: white;
        }}
        
        .methodology-overview li {{
            color: white;
        }}
        
        .performance-metrics {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .performance-metrics h3 {{
            color: white;
            margin-top: 0;
        }}
        
        .clinical-workflow {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .clinical-workflow h3 {{
            color: white;
            margin-top: 0;
        }}
        
        .clinical-workflow pre {{
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            h2 {{
                font-size: 1.5em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Vivli System</h1>
            <p>Complete Methodology</p>
            <p>From Step 1 to Final Implementation</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </div>
        
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#methodology-overview">Methodology Overview</a></li>
                <li><a href="#step-1">Step 1: Data Preparation and Exploration</a></li>
                <li><a href="#step-2">Step 2: Antibiotic Decision Tree Model Development</a></li>
                <li><a href="#step-3">Step 3: Phenotypic Signature Analysis and Clustering</a></li>
                <li><a href="#step-4">Step 4: Cefiderocol Use Prediction Model</a></li>
                <li><a href="#integration">Integration and Clinical Application</a></li>
                <li><a href="#validation">Validation and Limitations</a></li>
                <li><a href="#future-directions">Future Directions</a></li>
                <li><a href="#summary">Summary</a></li>
            </ul>
        </div>
        
        {html_content}
        
        <div class="footer">
            <p>This document provides the complete methodology for the Vivli system, covering all steps from data preparation to final clinical implementation.</p>
            <p>For technical support or questions, please refer to the source code and documentation.</p>
        </div>
    </div>
    
    <script>
        // Add smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start'
                    }});
                }}
            }});
        }});
        
        // Add syntax highlighting for code blocks
        document.querySelectorAll('pre code').forEach(block => {{
            block.style.display = 'block';
        }});
        
        // Add hover effects for step boxes
        document.querySelectorAll('.step-box').forEach(box => {{
            box.addEventListener('mouseenter', function() {{
                this.style.transform = 'translateY(-2px)';
                this.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
            }});
            
            box.addEventListener('mouseleave', function() {{
                this.style.transform = 'translateY(0)';
                this.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            }});
        }});
        
        // Add step numbering
        document.querySelectorAll('h2').forEach((h2, index) => {{
            if (h2.textContent.includes('Step')) {{
                const stepNumber = index + 1;
                h2.innerHTML = '<span style="background-color: #27ae60; color: white; padding: 5px 10px; border-radius: 50%; margin-right: 10px;">' + stepNumber + '</span>' + h2.textContent;
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    with open('vivli_complete_methodology.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("✅ Successfully converted vivli_complete_methodology.md to HTML!")
    print("📁 File created: vivli_complete_methodology.html")
    print("🌐 Open the HTML file in your browser to view the formatted document")

def main():
    """Main function to convert markdown to HTML."""
    print("=== CONVERTING COMPLETE METHODOLOGY TO HTML ===\n")
    
    # Check if the markdown file exists
    if not os.path.exists('vivli_complete_methodology.md'):
        print("❌ Error: vivli_complete_methodology.md not found!")
        return
    
    # Convert the file
    convert_methodology_to_html()
    
    print("\n🎉 Conversion completed successfully!")

if __name__ == "__main__":
    main()
