from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vivli-system",
    version="1.0.0",
    author="Vivli System Team",
    author_email="contact@vivli-system.com",
    description="Antibiotic Decision Support System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/vivli-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "vivli-antibiotic=scripts.antibiotic_decision_tree:main",
            "vivli-cefiderocol=scripts.step4_prediction:main",
            "vivli-report=scripts.generate_english_antibiotic_report:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.html", "*.css"],
    },
    keywords="antibiotic, decision support, machine learning, clinical, research",
    project_urls={
        "Bug Reports": "https://github.com/your-username/vivli-system/issues",
        "Source": "https://github.com/your-username/vivli-system",
        "Documentation": "https://github.com/your-username/vivli-system/docs",
    },
)
