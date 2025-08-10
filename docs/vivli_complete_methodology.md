# Vivli System - Complete Methodology

## Executive Summary

The Vivli system is a comprehensive antibiotic decision support platform that combines multiple prediction models to provide evidence-based antibiotic recommendations. This methodology document outlines the complete approach from data preparation through final clinical recommendations.

## Methodology Overview

The Vivli system follows a structured 4-step methodology:

1. **Step 1**: Data Preparation and Exploration
2. **Step 2**: Antibiotic Decision Tree Model Development
3. **Step 3**: Phenotypic Signature Analysis and Clustering
4. **Step 4**: Cefiderocol Use Prediction Model

---

## Step 1: Data Preparation and Exploration

### 1.1 Data Sources

**Primary Databases:**
- **ATLAS Database** (2.xlsx): Global antimicrobial susceptibility data
- **SIDERO-WT Database** (1.xlsx): Cefiderocol-specific susceptibility data

**Data Characteristics:**
- **ATLAS**: 966,805 isolates, 134 variables, multiple countries and species
- **SIDERO-WT**: Cefiderocol-specific data with MIC values and resistance patterns

### 1.2 Data Cleaning and Preprocessing

#### MIC Value Standardization
```python
def clean_mic_values(value):
    """Standardize MIC values across databases."""
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

#### Resistance Threshold Application
```python
breakpoints = {
    "cefiderocol_mic": 4,
    "meropenem_mic": 8,
    "CIPROFLOXACIN_mic": 1,
    "colistin_mic": 4
}
```

### 1.3 Exploratory Data Analysis

#### Species Distribution Analysis
- Top bacterial species identification
- Geographic distribution patterns
- Temporal evolution analysis

#### Resistance Pattern Exploration
- Global resistance trends
- Regional variations
- Species-specific resistance profiles

---

## Step 2: Antibiotic Decision Tree Model Development

### 2.1 Model Architecture

**Algorithm**: Decision Tree Classifier
**Parameters**:
- Max Depth: 10
- Random State: 42
- Train/Test Split: 80/20

### 2.2 Feature Engineering

#### Primary Features (7 total)
1. **Species encoded** (LabelEncoder)
2. **Country encoded** (LabelEncoder)
3. **Year** (temporal data)
4. **Beta-lactam resistance** (mean resistance score)
5. **Aminoglycoside resistance** (mean resistance score)
6. **Quinolone resistance** (mean resistance score)
7. **Other resistance** (mean resistance score)

#### Antibiotic Analysis (40+ antibiotics)
**Categories analyzed**:
- Carbapenems: Meropenem, Imipenem, Doripenem, Ertapenem, Tebipenem
- Beta-lactam combinations: Ceftazidime avibactam, Ceftaroline avibactam, Ceftolozane tazobactam
- Cephalosporins: Cefepime, Ceftazidime, Ceftriaxone, Cefoxitin, Cefixime
- Aminoglycosides: Amikacin, Gentamicin, Tobramycin, Streptomycin
- Fluoroquinolones: Ciprofloxacin, Levofloxacin, Gatifloxacin, Moxifloxacin
- Others: Colistin, Vancomycin, Linezolid, Tigecycline, etc.

### 2.3 Training Process

#### Step 2.1: Data Loading and Cleaning
```python
# Load ATLAS data
self.data = pd.read_excel("2.xlsx")

# Clean MIC values for all antibiotics
for col in antibiotic_columns:
    if col in self.data.columns:
        self.data[col] = self.data[col].apply(self.clean_mic_values)
        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
```

#### Step 2.2: Resistance Threshold Definition
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

#### Step 2.3: Antibiotic Scoring
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

#### Step 2.4: Optimal Antibiotic Order Determination
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

#### Step 2.5: Decision Feature Creation
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

#### Step 2.6: Model Training
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

### 2.4 Model Performance

**Performance Metrics:**
- **Accuracy**: 100% (reported)
- **Target**: First antibiotic recommendation
- **Output**: Complete antibiotic sequence

**Antibiotic Order (Top 20):**
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

---

## Step 3: Phenotypic Signature Analysis and Clustering

### 3.1 Data Preparation for Clustering

#### Feature Selection
- MIC values for key antibiotics
- Resistance patterns
- Species and geographic information
- Temporal data

#### Dimensionality Reduction
```python
# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)
```

### 3.2 Clustering Analysis

#### Optimal Cluster Determination
```python
# Elbow method for optimal cluster number
from sklearn.cluster import KMeans

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)
```

#### Hierarchical Clustering
```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Perform hierarchical clustering
linkage_matrix = linkage(X_pca, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
```

### 3.3 Phenotypic Signature Identification

#### Cluster Characterization
- Resistance patterns by cluster
- Species distribution within clusters
- Geographic patterns
- Temporal trends

#### Signature Validation
- Cross-validation of cluster stability
- Biological interpretation of signatures
- Clinical relevance assessment

---

## Step 4: Cefiderocol Use Prediction Model

### 4.1 Model Architecture

**Best Model**: Random Forest Classifier
**Alternative Models Tested**: XGBoost, Gradient Boosting, Logistic Regression
**Cross-validation**: 5-fold StratifiedKFold
**Feature Scaling**: StandardScaler

### 4.2 Target Definition

**Use cefiderocol when:**
- Cefiderocol is susceptible (MIC < 4 mg/L)
- AND resistance to other antibiotics is present

**Target variable**: `use_cefiderocol` (binary: 0 = don't use, 1 = use)

### 4.3 Feature Engineering

#### Basic MIC Features
- `cefiderocol_mic`: Direct susceptibility measure
- `meropenem_mic`: Carbapenem susceptibility
- `CIPROFLOXACIN_mic`: Fluoroquinolone susceptibility  
- `colistin_mic`: Polymyxin susceptibility

#### Log-Transformed MIC Features
- `log_cefiderocol_mic`
- `log_meropenem_mic`
- `log_CIPROFLOXACIN_mic`
- `log_colistin_mic`

#### Resistance Binary Features
- `cefiderocol_mic_resistant`: Binary resistance indicator
- `meropenem_mic_resistant`: Carbapenem resistance
- `CIPROFLOXACIN_mic_resistant`: Fluoroquinolone resistance
- `colistin_mic_resistant`: Polymyxin resistance

#### Composite Resistance Features
- `total_resistance`: Sum of all resistance indicators
- `cefiderocol_only_susceptible`: Cefiderocol susceptible + others resistant
- `multidrug_resistant`: ≥2 resistant antibiotics
- `extensively_drug_resistant`: ≥3 resistant antibiotics

#### MIC Ratio Features (Comparative Analysis)
- `meropenem_cefiderocol_ratio`: Comparative susceptibility
- `ciprofloxacin_cefiderocol_ratio`: Comparative susceptibility
- `colistin_cefiderocol_ratio`: Comparative susceptibility

#### Categorical Features
- `species_encoded`: Encoded bacterial species
- `region_encoded`: Encoded geographic region

### 4.4 Model Training Process

#### Step 4.1: Data Preparation
```python
def load_and_prepare_data():
    """Load and prepare data for cefiderocol prediction model."""
    # Load data
    sidero_data = pd.read_excel("1.xlsx")
    atlas_data = pd.read_excel("2.xlsx")
    
    # Standardize columns
    sidero_mapping = {
        "Cefiderocol": "cefiderocol_mic",
        "Meropenem": "meropenem_mic",
        "Ciprofloxacin": "CIPROFLOXACIN_mic",
        "Colistin": "colistin_mic",
        "Organism Name": "species",
        "Region": "region",
        "Year Collected": "year"
    }
    
    sidero_data.rename(columns=sidero_mapping, inplace=True)
    
    # Clean MIC values
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[col] = sidero_data[col].apply(clean_mic_values)
            sidero_data[col] = pd.to_numeric(sidero_data[col], errors='coerce')
    
    return sidero_data, atlas_data
```

#### Step 4.2: Feature Creation
```python
def create_prediction_features(data):
    """Create features for cefiderocol prediction model."""
    # Basic MIC features
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    features_df = data[mic_columns].copy()
    
    # Log transform MIC values
    for col in mic_columns:
        features_df[f"log_{col}"] = np.log1p(features_df[col])
    
    # Resistance features
    resistance_columns = [f"{col}_resistant" for col in mic_columns]
    for col in resistance_columns:
        if col in data.columns:
            features_df[col] = data[col]
    
    # Resistance patterns
    features_df["total_resistance"] = features_df[resistance_columns].sum(axis=1)
    features_df["cefiderocol_only_susceptible"] = (
        (features_df["cefiderocol_mic_resistant"] == 0) & 
        (features_df["total_resistance"] >= 1)
    ).astype(int)
    
    # MIC ratios (comparative susceptibility)
    features_df["meropenem_cefiderocol_ratio"] = features_df["meropenem_mic"] / (features_df["cefiderocol_mic"] + 0.01)
    features_df["ciprofloxacin_cefiderocol_ratio"] = features_df["CIPROFLOXACIN_mic"] / (features_df["cefiderocol_mic"] + 0.01)
    features_df["colistin_cefiderocol_ratio"] = features_df["colistin_mic"] / (features_df["cefiderocol_mic"] + 0.01)
    
    # Categorical features
    le_species = LabelEncoder()
    le_region = LabelEncoder()
    
    features_df["species_encoded"] = le_species.fit_transform(data["species"].fillna("Unknown"))
    features_df["region_encoded"] = le_region.fit_transform(data["region"].fillna("Unknown"))
    features_df["year"] = data["year"]
    
    # Clinical decision features
    features_df["multidrug_resistant"] = (features_df["total_resistance"] >= 2).astype(int)
    features_df["extensively_drug_resistant"] = (features_df["total_resistance"] >= 3).astype(int)
    
    # Target
    target = data["use_cefiderocol"]
    
    # Remove year column and fill missing values
    features_df = features_df.drop("year", axis=1).fillna(0)
    
    return features_df, target, le_species, le_region
```

#### Step 4.3: Model Training
```python
def train_prediction_models(X_train, X_test, y_train, y_test):
    """Train multiple models for cefiderocol prediction."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to train
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "auc": auc,
            "precision": precision,
            "recall": recall
        }
    
    return results, scaler
```

### 4.5 Performance Metrics

**Overall Performance:**
- **AUC Score**: 1.000
- **Precision**: 1.000
- **Recall**: 1.000

**Clinical Performance:**
- **Sensitivity**: 1.000
- **Specificity**: 1.000
- **Positive Predictive Value**: 1.000
- **Negative Predictive Value**: 1.000

### 4.6 Feature Importance Analysis

**Top 10 Most Important Features (Random Forest):**
1. `cefiderocol_only_susceptible` (importance: 0.267)
2. `total_resistance` (importance: 0.239)
3. `CIPROFLOXACIN_mic` (importance: 0.091)
4. `colistin_mic_resistant` (importance: 0.063)
5. `log_CIPROFLOXACIN_mic` (importance: 0.062)
6. `colistin_mic` (importance: 0.059)
7. `CIPROFLOXACIN_mic_resistant` (importance: 0.046)
8. `log_colistin_mic` (importance: 0.045)
9. `ciprofloxacin_cefiderocol_ratio` (importance: 0.043)
10. `colistin_cefiderocol_ratio` (importance: 0.018)

### 4.7 Clinical Decision Rules

#### Rule 1: MIC Threshold
- **Use cefiderocol**: MIC < 4 mg/L
- **Avoid cefiderocol**: MIC ≥ 4 mg/L

#### Rule 2: Resistance Pattern Analysis
- **Use cefiderocol**: Susceptible + other antibiotics resistant
- **Consider cefiderocol**: Multidrug-resistant (≥2 resistant antibiotics)

#### Rule 3: Comparative MIC Analysis
- **Use cefiderocol**: Lower MIC compared to other antibiotics
- **Consider cefiderocol**: Meropenem/cefiderocol ratio > 2

#### Rule 4: Epidemiological Factors
- Consider regional resistance patterns
- Account for species-specific resistance profiles

---

## Integration and Clinical Application

### Clinical Decision Framework

#### 1. First-Line Treatment (Step 2)
- Model predicts most effective antibiotic based on species, region, and resistance patterns
- Considers global efficacy data from ATLAS database

#### 2. Sequential Alternatives (Step 2)
- Provides complete sequence of alternatives
- Each subsequent option is less effective but still viable
- Maintains therapeutic options for treatment failure

#### 3. Phenotypic Analysis (Step 3)
- Identifies resistance patterns and clusters
- Provides insights into resistance mechanisms
- Supports personalized treatment approaches

#### 4. Last Resort Decision (Step 4)
- Determines when to use cefiderocol
- Always positioned as final option
- Preserves this critical antibiotic

### Example Clinical Workflow

```python
# Complete clinical workflow
def clinical_workflow(species, country, year, resistance_profile=None):
    """Complete clinical workflow for antibiotic recommendation."""
    
    # Step 2: Get antibiotic sequence
    recommendations = antibiotic_model.recommend_antibiotics(
        species=species,
        country=country,
        year=year,
        resistance_profile=resistance_profile
    )
    
    # Step 3: Analyze phenotypic signatures
    cluster_analysis = phenotypic_model.analyze_signatures(species, resistance_profile)
    
    # Step 4: Determine cefiderocol use
    cefiderocol_decision = cefiderocol_model.predict_use(
        species=species,
        resistance_profile=resistance_profile
    )
    
    return {
        'antibiotic_sequence': recommendations,
        'phenotypic_signatures': cluster_analysis,
        'cefiderocol_recommendation': cefiderocol_decision
    }
```

---

## Validation and Limitations

### Model Validation

#### Strengths
1. **Evidence-based**: Uses global ATLAS data
2. **Sequential approach**: Provides complete treatment sequence
3. **Antimicrobial stewardship**: Preserves cefiderocol
4. **Regional consideration**: Accounts for geographic resistance patterns
5. **Species-specific**: Tailored to bacterial species

#### Limitations
1. **Retrospective data**: Based on historical susceptibility patterns
2. **Simplified resistance**: Uses binary resistance indicators
3. **Clinical factors**: Doesn't include patient-specific factors
4. **Validation needed**: Requires clinical validation before use

### Critical Considerations

#### Data Limitations
- No real treatment failure data available
- Targets based on theoretical resistance patterns
- High performance likely reflects simplified target definitions
- Geographic and temporal biases possible

#### Clinical Implementation
- **Do NOT use for clinical decisions** without validation
- **Validate on real treatment failure data**
- **Include clinical factors** (comorbidities, previous exposure)
- **Prospective validation** required before implementation
- **Use as research tool** only

---

## Future Directions

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

### 4. Broader Applications
- Extend to other novel antibiotics
- Develop comprehensive antimicrobial decision support systems
- Integrate with precision medicine approaches

---

## Summary

The Vivli system provides a comprehensive framework for antibiotic decision support through a structured 4-step methodology:

1. **Data Preparation**: Standardization and exploration of global susceptibility data
2. **Antibiotic Sequencing**: Evidence-based recommendation of treatment sequences
3. **Phenotypic Analysis**: Identification of resistance patterns and signatures
4. **Cefiderocol Optimization**: Targeted use of critical antibiotics

**Key Clinical Value**: 
- Evidence-based antibiotic sequencing
- Antimicrobial stewardship compliance
- Regional resistance pattern consideration
- Complete treatment pathway provision
- Phenotypic signature identification
- Optimized cefiderocol use

This methodology serves as the foundation for clinical decision support in antibiotic therapy, providing a systematic approach to treatment selection that balances efficacy with antibiotic preservation while supporting precision medicine approaches.
