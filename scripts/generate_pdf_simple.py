import os
import subprocess
import shutil
from datetime import datetime

def copy_images_to_reports():
    """Copy plot images to reports directory for proper PDF generation."""
    print("Copying plot images to reports directory...")
    
    # Create plots directory in reports if it doesn't exist
    reports_plots_dir = "outputs/reports/plots"
    os.makedirs(reports_plots_dir, exist_ok=True)
    
    # Copy all plot images
    source_plots_dir = "outputs/plots"
    if os.path.exists(source_plots_dir):
        for file in os.listdir(source_plots_dir):
            if file.endswith('.png'):
                source_file = os.path.join(source_plots_dir, file)
                dest_file = os.path.join(reports_plots_dir, file)
                shutil.copy2(source_file, dest_file)
                print(f"Copied: {file}")
    else:
        print(f"Warning: {source_plots_dir} not found")

def convert_markdown_to_pdf(markdown_file, pdf_file):
    """Convert Markdown file to PDF using pandoc."""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs("outputs/reports", exist_ok=True)
        
        # Change to reports directory for proper relative paths
        original_dir = os.getcwd()
        os.chdir("outputs/reports")
        
        # Pandoc command with styling
        cmd = [
            'pandoc',
            os.path.basename(markdown_file),
            '-o', os.path.basename(pdf_file),
            '--metadata', f'title=Vivli Project Report - {datetime.now().strftime("%B %d, %Y")}',
            '--metadata', 'author=Vivli Analysis Team',
            '--toc',
            '--number-sections'
        ]
        
        # Try to run pandoc
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"PDF file created: {pdf_file}")
        else:
            print(f"Error with pandoc: {result.stderr}")
            # Fallback: try without CSS
            cmd_simple = [
                'pandoc',
                os.path.basename(markdown_file),
                '-o', os.path.basename(pdf_file),
                '--metadata', f'title=Vivli Project Report - {datetime.now().strftime("%B %d, %Y")}',
                '--toc',
                '--number-sections'
            ]
            result_simple = subprocess.run(cmd_simple, capture_output=True, text=True)
            if result_simple.returncode == 0:
                print(f"PDF file created (simple): {pdf_file}")
            else:
                print(f"Error with simple pandoc: {result_simple.stderr}")
        
        # Change back to original directory
        os.chdir(original_dir)
                
    except Exception as e:
        print(f"Error creating PDF: {e}")
        # Change back to original directory in case of error
        os.chdir(original_dir)

def main():
    """Generate PDF reports for steps 3 and 4."""
    print("=== Generating PDF Reports for Steps 3 and 4 ===\n")
    
    # Copy images to reports directory
    copy_images_to_reports()
    
    # Step 3 Report
    print("Processing Step 3 Report...")
    step3_md = "outputs/step3_report.md"
    step3_pdf = "outputs/reports/step3_phenotypic_signatures.pdf"
    
    if os.path.exists(step3_md):
        convert_markdown_to_pdf(step3_md, step3_pdf)
    else:
        print(f"Warning: {step3_md} not found")
    
    # Step 4 Report
    print("\nProcessing Step 4 Report...")
    step4_md = "outputs/step4_report.md"
    step4_pdf = "outputs/reports/step4_cefiderocol_prediction.pdf"
    
    if os.path.exists(step4_md):
        convert_markdown_to_pdf(step4_md, step4_pdf)
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
    combined_pdf = "outputs/reports/combined_steps_3_4_report.pdf"
    convert_markdown_to_pdf(combined_md, combined_pdf)
    
    print("\n=== PDF Generation Complete ===")
    print("Generated files:")
    print(f"- Step 3: {step3_pdf}")
    print(f"- Step 4: {step4_pdf}")
    print(f"- Combined: {combined_pdf}")
    print("\nFiles are located in: outputs/reports/")

if __name__ == "__main__":
    main() 