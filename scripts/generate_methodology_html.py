#!/usr/bin/env python3
"""
Simplified script to generate methodology HTML.
"""

import markdown
import os
from datetime import datetime

def generate_methodology_html():
    """Generate HTML for the complete methodology."""
    
    # Read the markdown file
    with open('vivli_complete_methodology.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
    
    # Simple HTML template
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
        
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
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
        
        {html_content}
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; color: #7f8c8d; border-top: 1px solid #ecf0f1;">
            <p>This document provides the complete methodology for the Vivli system, covering all steps from data preparation to final clinical implementation.</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write the HTML file
    with open('vivli_complete_methodology.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("✅ Successfully generated vivli_complete_methodology.html!")
    print("📁 File created: vivli_complete_methodology.html")
    print("🌐 Open the HTML file in your browser to view the formatted document")

def main():
    """Main function."""
    print("=== GENERATING METHODOLOGY HTML ===\n")
    
    if not os.path.exists('vivli_complete_methodology.md'):
        print("❌ Error: vivli_complete_methodology.md not found!")
        return
    
    generate_methodology_html()
    print("\n🎉 Generation completed successfully!")

if __name__ == "__main__":
    main()
