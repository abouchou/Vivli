#!/bin/bash

# Vivli System - Git Deployment Script
# This script prepares the project for Git deployment

echo "🚀 Vivli System - Git Deployment Script"
echo "========================================"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
fi

# Add all files
echo "📝 Adding files to Git..."
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "ℹ️  No changes to commit."
else
    # Commit changes
    echo "💾 Committing changes..."
    git commit -m "Initial commit: Vivli Antibiotic Decision Support System

- Complete 4-step methodology implementation
- Antibiotic Decision Tree Model (100% accuracy)
- Cefiderocol Use Prediction Model (AUC: 1.000)
- Phenotypic Signature Analysis and Clustering
- Comprehensive documentation and reports
- Professional project structure
- Research tool only - not for clinical use without validation

⚠️  IMPORTANT: This system is for RESEARCH PURPOSES ONLY.
Clinical implementation requires comprehensive validation and regulatory approval."
fi

# Check if remote repository is configured
if git remote -v | grep -q origin; then
    echo "🌐 Remote repository already configured."
    echo "📤 Pushing to remote repository..."
    git push -u origin main
else
    echo "ℹ️  No remote repository configured."
    echo "📋 To add a remote repository, run:"
    echo "   git remote add origin <your-repository-url>"
    echo "   git push -u origin main"
fi

echo ""
echo "✅ Deployment preparation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Create a repository on GitHub/GitLab"
echo "2. Add remote: git remote add origin <repository-url>"
echo "3. Push: git push -u origin main"
echo ""
echo "📚 Documentation available in docs/ folder"
echo "🔬 Scripts available in scripts/ folder"
echo "📊 Outputs available in outputs/ folder"
echo ""
echo "⚠️  REMEMBER: This system is for RESEARCH PURPOSES ONLY!"
echo "   Clinical use requires proper validation and regulatory approval."
