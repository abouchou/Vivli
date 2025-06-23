import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# Configuration des paramètres d'affichage
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Création des dossiers pour les sorties
os.makedirs("outputs/plots", exist_ok=True)

def clean_mic_values(value):
    """Nettoie les valeurs MIC en supprimant <=, >= et en convertissant en float."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        # Supprimer <=, >=, ou autres caractères non numériques
        cleaned_value = re.sub(r'[^\d.]', '', value)
        try:
            return float(cleaned_value)
        except ValueError:
            return pd.NA
    return float(value)

def load_data(sidero_path="sidero-wt.xlsx", atlas_path="atlas.xlsx"):
    """Charge les fichiers SIDERO-WT et ATLAS."""
    try:
        sidero_data = pd.read_excel(sidero_path)
        atlas_data = pd.read_excel(atlas_path)
        print("SIDERO-WT chargé : ", sidero_data.shape)
        print("ATLAS chargé : ", atlas_data.shape)
        return sidero_data, atlas_data
    except Exception as e:
        print(f"Erreur de chargement des données : {e}")
        return None, None

def standardize_columns(sidero_data, atlas_data):
    """Standardise les noms de colonnes et traite les valeurs manquantes."""
    # Mapping pour les colonnes pertinentes
    sidero_mapping = {
        "Cefiderocol": "cefiderocol_mic",
        "Meropenem": "meropenem_mic",
        "Ciprofloxacin": "CIPROFLOXACIN_mic",
        "Colistin": "colistin_mic",
        "Organism Name": "species",
        "Region": "region",
        "Year Collected": "year"
    }
    atlas_mapping = {
        "Species": "species",
        "Country": "country",
        "Year": "year",
        "Amikacin": "amikacin_mic",
        "Cefepime": "cefepime_mic",
        "Ceftazidime avibactam": "ceftazidime_avibactam_mic",
        "Ciprofloxacin": "CIPROFLOXACIN_mic",
        "Colistin": "colistin_mic",
        "Meropenem": "meropenem_mic"
    }

    # Renommer les colonnes
    sidero_data.rename(columns=sidero_mapping, inplace=True)
    atlas_data.rename(columns=atlas_mapping, inplace=True)

    # Nettoyer les colonnes MIC
    mic_columns = ["meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic", 
                   "amikacin_mic", "cefepime_mic", "ceftazidime_avibactam_mic"]
    for col in mic_columns:
        if col in atlas_data.columns:
            atlas_data[col] = atlas_data[col].apply(clean_mic_values)
        if col in sidero_data.columns:
            sidero_data[col] = sidero_data[col].apply(clean_mic_values)

    # Convertir les colonnes MIC en float
    for col in mic_columns:
        if col in atlas_data.columns:
            atlas_data[col] = pd.to_numeric(atlas_data[col], errors='coerce')
        if col in sidero_data.columns:
            sidero_data[col] = pd.to_numeric(sidero_data[col], errors='coerce')

    # Convertir les valeurs NULL en NaN
    sidero_data.fillna(pd.NA, inplace=True)
    atlas_data.fillna(pd.NA, inplace=True)

    # Vérifier les colonnes après standardisation
    print("Colonnes SIDERO-WT : ", sidero_data.columns.tolist())
    print("Colonnes ATLAS : ", atlas_data.columns.tolist())

    return sidero_data, atlas_data

def define_resistance_target(sidero_data):
    """Définit la cible binaire pour la résistance au céfidérocol (MIC ≥ 4)."""
    sidero_data["cefiderocol_mic"] = pd.to_numeric(sidero_data["cefiderocol_mic"], errors='coerce')
    sidero_data["cefiderocol_resistant"] = sidero_data["cefiderocol_mic"] >= 4
    print("Distribution de la résistance au céfidérocol :")
    print(sidero_data["cefiderocol_resistant"].value_counts(normalize=True))
    return sidero_data

def exploratory_data_analysis(sidero_data, atlas_data):
    """Réalise l'analyse exploratoire des données."""
    # Distribution des MIC pour le céfidérocol (SIDERO-WT)
    plt.figure()
    sns.histplot(sidero_data["cefiderocol_mic"].dropna(), bins=20, kde=True)
    plt.title("Distribution des MIC pour le céfidérocol")
    plt.xlabel("MIC (µg/mL)")
    plt.ylabel("Fréquence")
    plt.savefig("outputs/plots/cefiderocol_mic_distribution.png")
    plt.close()

    # MIC du céfidérocol par année (SIDERO-WT)
    plt.figure()
    sns.boxplot(x="year", y="cefiderocol_mic", data=sidero_data)
    plt.title("MIC du céfidérocol par année")
    plt.xlabel("Année")
    plt.ylabel("MIC (µg/mL)")
    plt.savefig("outputs/plots/cefiderocol_mic_by_year.png")
    plt.close()

    # Résistance au céfidérocol par région (SIDERO-WT)
    plt.figure()
    sns.countplot(x="region", hue="cefiderocol_resistant", data=sidero_data)
    plt.title("Résistance au céfidérocol par région")
    plt.xlabel("Région")
    plt.ylabel("Nombre d'isolats")
    plt.savefig("outputs/plots/cefiderocol_resistance_by_region.png")
    plt.close()

    # Fréquence des espèces dans SIDERO-WT et ATLAS
    print("\nEspèces les plus fréquentes dans SIDERO-WT :")
    print(sidero_data["species"].value_counts().head(10))
    print("\nEspèces les plus fréquentes dans ATLAS :")
    print(atlas_data["species"].value_counts().head(10))

    # Distribution des MIC pour le méropénem (ATLAS)
    plt.figure()
    sns.histplot(atlas_data["meropenem_mic"].dropna(), bins=20, kde=True)
    plt.title("Distribution des MIC pour le méropénem (ATLAS)")
    plt.xlabel("MIC (µg/mL)")
    plt.ylabel("Fréquence")
    plt.savefig("outputs/plots/meropenem_mic_distribution_atlas.png")
    plt.close()

def main():
    """Fonction principale pour exécuter la configuration et l'EDA."""
    # Étape 1 : Chargement des données
    sidero_data, atlas_data = load_data()

    if sidero_data is None or atlas_data is None:
        return

    # Étape 2 : Standardisation des colonnes
    sidero_data, atlas_data = standardize_columns(sidero_data, atlas_data)

    # Étape 3 : Définition de la cible de résistance
    sidero_data = define_resistance_target(sidero_data)

    # Étape 4 : Analyse exploratoire
    exploratory_data_analysis(sidero_data, atlas_data)

if __name__ == "__main__":
    main()