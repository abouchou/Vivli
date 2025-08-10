# Antibiotic Recommendation System - Prediction Model Details

## Overview

The **Antibiotic Recommendation System** is the core prediction model in the Vivli system. It uses a decision tree to recommend antibiotics in optimal order of efficacy, with cefiderocol as the last resort option. This model directly addresses the clinical requirement to provide sequential treatment recommendations.

## Model Architecture

### Algorithm
- **Primary Model**: Decision Tree Classifier
- **Max Depth**: 10
- **Random State**: 42
- **Train/Test Split**: 80/20

### Model Parameters
```python
DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)
```

## Target Definition

The model predicts the **first-choice antibiotic** for a given infection, then provides a complete sequence of alternatives:

**Target**: First antibiotic recommendation (categorical)
**Output**: Complete antibiotic sequence with cefiderocol as last resort

## Feature Engineering

### 1. Primary Features (7 total)
1. **Species encoded** (LabelEncoder)
2. **Country encoded** (LabelEncoder)
3. **Year** (temporal data)
4. **Beta-lactam resistance** (mean resistance score)
5. **Aminoglycoside resistance** (mean resistance score)
6. **Quinolone resistance** (mean resistance score)
7. **Other resistance** (mean resistance score)

### 2. Antibiotic Analysis (40+ antibiotics)
The system analyzes 40+ antibiotics including:

**Carbapenems**: Meropenem, Imipenem, Doripenem, Ertapenem, Tebipenem
**Beta-lactam combinations**: Ceftazidime avibactam, Ceftaroline avibactam, Ceftolozane tazobactam
**Cephalosporins**: Cefepime, Ceftazidime, Ceftriaxone, Cefoxitin, Cefixime
**Aminoglycosides**: Amikacin, Gentamicin, Tobramycin, Streptomycin
**Fluoroquinolones**: Ciprofloxacin, Levofloxacin, Gatifloxacin, Moxifloxacin
**Others**: Colistin, Vancomycin, Linezolid, Tigecycline, etc.

## Training Process

### 1. Data Loading and Cleaning
```python
# Load ATLAS data
self.data = pd.read_excel("2.xlsx")

# Clean MIC values
def clean_mic_values(self, value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        import re
        cleaned_value = re.sub(r'[^\d.]', '', value)
        try:
            return float(cleaned_value)
        except ValueError:
            return np.nan
    return float(value)
```

### 2. Resistance Threshold Definition
```python
thresholds = {
    'Meropenem': 8, 'Imipenem': 8, 'Doripenem': 8, 'Ertapenem': 2, 'Tebipenem': 4,
    'Ceftazidime avibactam': 8, 'Ceftaroline avibactam': 4, 'Ceftolozane tazobactam': 8,
    'Meropenem vaborbactam': 8, 'Aztreonam avibactam': 8, 'Ceftibuten avibactam': 4,
    'Cefepime': 32, 'Ceftazidime': 32, 'Ceftriaxone': 64, 'Cefoxitin': 32,
    'Cefixime': 1, 'Ceftaroline': 4, 'Ceftibuten': 4, 'Cefpodoxime': 2,
    'Cefoperazone sulbactam': 64, 'Piperacillin tazobactam': 128,
    'Ampicillin sulbactam': 32, 'Amoxycillin clavulanate': 32,
    'Ampicillin': 32, 'Penicillin': 0.12, 'Oxacillin': 2,
    'Amikacin': 64, 'Gentamicin': 16, 'Tobramycin': 16, 'Streptomycin': 64,
    'Ciprofloxacin': 4, 'Levofloxacin': 8, 'Gatifloxacin': 8, 'Moxifloxacin': 4,
    'Colistin': 4, 'Polymyxin B': 4,
    'Tigecycline': 8, 'Minocycline': 16, 'Tetracycline': 16,
    'Vancomycin': 16, 'Teicoplanin': 32, 'Daptomycin': 4, 'Linezolid': 8,
    'Clarithromycin': 8, 'Erythromycin': 8, 'Azithromycin': 16, 'Clindamycin': 4,
    'Metronidazole': 16, 'Trimethoprim sulfa': 76, 'Sulfamethoxazole': 512,
    'Quinupristin dalfopristin': 4
}
```

### 3. Antibiotic Scoring
```python
def calculate_antibiotic_scores(self, antibiotic_columns):
    """Calculate efficacy scores for each antibiotic."""
    scores_df = self.data[['Species', 'Country', 'Year']].copy()
    
    for col in antibiotic_columns:
        if col in self.data.columns:
            resistant_col = f"{col}_resistant"
            if resistant_col in self.data.columns:
                # Score based on proportion of susceptible strains
                scores_df[f"{col}_score"] = ~self.data[resistant_col]
    
    return scores_df
```

### 4. Optimal Antibiotic Order Determination
```python
def determine_antibiotic_order(self, scores_df, antibiotic_columns):
    """Determine optimal antibiotic order based on global efficacy."""
    antibiotic_efficacy = {}
    for col in antibiotic_columns:
        score_col = f"{col}_score"
        if score_col in scores_df.columns:
            efficacy = scores_df[score_col].mean()
            antibiotic_efficacy[col] = efficacy
    
    # Sort by decreasing efficacy
    sorted_antibiotics = sorted(antibiotic_efficacy.items(), key=lambda x: x[1], reverse=True)
    
    # Add cefiderocol at the end (last resort)
    self.antibiotic_order = [ab[0] for ab in sorted_antibiotics] + ['Cefiderocol']
    
    return self.antibiotic_order
```

### 5. Decision Feature Creation
```python
def create_decision_features(self, scores_df):
    """Create features for decision tree."""
    # Encode categorical variables
    le_species = LabelEncoder()
    le_country = LabelEncoder()
    
    scores_df['species_encoded'] = le_species.fit_transform(scores_df['Species'].fillna('Unknown'))
    scores_df['country_encoded'] = le_country.fit_transform(scores_df['Country'].fillna('Unknown'))
    
    # Create resistance features by antibiotic class
    beta_lactam_cols = [col for col in scores_df.columns if any(x in col for x in ['penem', 'cef', 'penicillin', 'tazobactam', 'avibactam'])]
    aminoglycoside_cols = [col for col in scores_df.columns if any(x in col for x in ['micin', 'gentamicin', 'tobramycin', 'streptomycin'])]
    quinolone_cols = [col for col in scores_df.columns if any(x in col for x in ['floxacin'])]
    other_cols = [col for col in scores_df.columns if any(x in col for x in ['colistin', 'vancomycin', 'linezolid', 'tigecycline'])]
    
    # Calculate mean resistance scores by class
    scores_df['beta_lactam_resistance'] = scores_df[beta_lactam_cols].mean(axis=1)
    scores_df['aminoglycoside_resistance'] = scores_df[aminoglycoside_cols].mean(axis=1)
    scores_df['quinolone_resistance'] = scores_df[quinolone_cols].mean(axis=1)
    scores_df['other_resistance'] = scores_df[other_cols].mean(axis=1)
    
    return scores_df
```

### 6. Model Training
```python
def train_decision_tree(self, features_df):
    """Train decision tree to recommend first antibiotic."""
    # Select features
    feature_cols = ['species_encoded', 'country_encoded', 'Year',
                   'beta_lactam_resistance', 'aminoglycoside_resistance',
                   'quinolone_resistance', 'other_resistance']
    
    X = features_df[feature_cols].fillna(0)
    
    # Create target: first antibiotic recommendation
    first_choice = []
    for idx, row in features_df.iterrows():
        best_antibiotic = None
        best_score = -1
        for ab in self.antibiotic_order[:-1]:  # Exclude cefiderocol
            score_col = f"{ab}_score"
            if score_col in features_df.columns:
                if features_df.loc[idx, score_col] > best_score:
                    best_score = features_df.loc[idx, score_col]
                    best_antibiotic = ab
        first_choice.append(best_antibiotic if best_antibiotic else self.antibiotic_order[0])
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(first_choice)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train decision tree
    self.decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    self.decision_tree.fit(X_train, y_train)
    
    return X_train, X_test, y_train, y_test
```

## Performance Metrics

### Model Performance
- **Accuracy**: 100% (reported)
- **Target**: First antibiotic recommendation
- **Output**: Complete antibiotic sequence

### Antibiotic Order (Top 20)
1. Cefoperazone sulbactam
2. Gatifloxacin
3. Tetracycline
4. Metronidazole
5. Cefoxitin
6. Linezolid
7. Daptomycin
8. Ertapenem
9. Quinupristin dalfopristin
10. Teicoplanin
11. Tigecycline
12. Meropenem vaborbactam
13. Sulbactam
14. Ceftibuten
15. Vancomycin
16. Clarithromycin
17. Azithromycin
18. Ceftaroline avibactam
19. Doripenem
20. Ceftazidime avibactam
...
49. Cefiderocol (last resort)

## Clinical Application

### Recommendation Process
```python
def recommend_antibiotics(self, species, country, year, resistance_profile=None):
    """Recommend antibiotic sequence for a given case."""
    # Prepare features
    species_encoded = self.label_encoders['species'].transform([species])[0]
    country_encoded = self.label_encoders['country'].transform([country])[0]
    
    # Create feature vector
    features = np.array([[
        species_encoded, country_encoded, year,
        resistance_profile.get('beta_lactam', 0.5) if resistance_profile else 0.5,
        resistance_profile.get('aminoglycoside', 0.5) if resistance_profile else 0.5,
        resistance_profile.get('quinolone', 0.5) if resistance_profile else 0.5,
        resistance_profile.get('other', 0.5) if resistance_profile else 0.5
    ]])
    
    # Predict first antibiotic
    first_choice_idx = self.decision_tree.predict(features)[0]
    first_choice = self.label_encoders['target'].inverse_transform([first_choice_idx])[0]
    
    # Build complete sequence
    recommendations = []
    used_antibiotics = set()
    
    # Add first choice
    recommendations.append(first_choice)
    used_antibiotics.add(first_choice)
    
    # Add alternatives in efficacy order
    for ab in self.antibiotic_order:
        if ab not in used_antibiotics:
            recommendations.append(ab)
            used_antibiotics.add(ab)
    
    return recommendations
```

### Example Usage
```python
# Example recommendation
recommendations = model.recommend_antibiotics(
    species="Escherichia coli",
    country="France", 
    year=2023,
    resistance_profile={'beta_lactam': 0.3, 'quinolone': 0.7}
)

# Output: Complete antibiotic sequence with cefiderocol as last resort
```

## Clinical Decision Framework

### 1. First-Line Treatment
- Model predicts most effective antibiotic based on species, region, and resistance patterns
- Considers global efficacy data from ATLAS database

### 2. Sequential Alternatives
- Provides complete sequence of alternatives
- Each subsequent option is less effective but still viable
- Maintains therapeutic options for treatment failure

### 3. Last Resort (Cefiderocol)
- Always positioned as final option
- Preserves this critical antibiotic
- Used only when other options fail

## Data Sources

### Primary Data
- **ATLAS Database** (2.xlsx): Global antimicrobial susceptibility data
- **Coverage**: Multiple countries, species, and time periods
- **Antibiotics**: 40+ antibiotics with MIC values

### Data Quality
- MIC value cleaning and standardization
- Resistance threshold application
- Missing value handling
- Categorical encoding

## Technical Implementation

### Code Structure
```python
class AntibioticDecisionTree:
    def __init__(self)
    def load_atlas_data()
    def clean_mic_values()
    def prepare_antibiotic_data()
    def get_resistance_threshold()
    def calculate_antibiotic_scores()
    def determine_antibiotic_order()
    def create_decision_features()
    def train_decision_tree()
    def recommend_antibiotics()
    def visualize_tree()
    def generate_report()
```

### Dependencies
- **Scikit-learn**: DecisionTreeClassifier, LabelEncoder
- **Pandas/NumPy**: Data manipulation
- **Matplotlib**: Visualization
- **Seaborn**: Plotting

### Output Files
- **Reports**: `outputs/antibiotic_decision_tree_report.md`
- **Visualizations**: `outputs/plots/antibiotic_decision_tree.png`
- **PDF Reports**: `outputs/antibiotic_recommendation_report_english.pdf`

## Clinical Validation

### Strengths
1. **Evidence-based**: Uses global ATLAS data
2. **Sequential approach**: Provides complete treatment sequence
3. **Antimicrobial stewardship**: Preserves cefiderocol
4. **Regional consideration**: Accounts for geographic resistance patterns
5. **Species-specific**: Tailored to bacterial species

### Limitations
1. **Retrospective data**: Based on historical susceptibility patterns
2. **Simplified resistance**: Uses binary resistance indicators
3. **Clinical factors**: Doesn't include patient-specific factors
4. **Validation needed**: Requires clinical validation before use

## Future Enhancements

### 1. Clinical Integration
- Include patient factors (age, comorbidities, allergies)
- Add pharmacokinetic considerations
- Integrate with local resistance patterns

### 2. Model Improvements
- Include genomic resistance markers
- Add temporal resistance trends
- Develop species-specific models

### 3. Clinical Validation
- Prospective clinical studies
- Real-world implementation
- Outcome assessment

## Summary

The **Antibiotic Recommendation System** provides a comprehensive framework for sequential antibiotic treatment recommendations. It successfully addresses the clinical requirement to provide first-line treatment options while preserving critical antibiotics like cefiderocol for last resort use.

**Key Clinical Value**: 
- Evidence-based antibiotic sequencing
- Antimicrobial stewardship compliance
- Regional resistance pattern consideration
- Complete treatment pathway provision

This model serves as the foundation for clinical decision support in antibiotic therapy, providing a systematic approach to treatment selection that balances efficacy with antibiotic preservation.

