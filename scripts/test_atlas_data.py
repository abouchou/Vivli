import pandas as pd
import numpy as np

def test_atlas_data():
    """Test simple pour examiner les données ATLAS."""
    print("=== Test des données ATLAS ===")
    
    try:
        # Charger les données ATLAS
        atlas_data = pd.read_excel("2.xlsx")
        print(f"Shape ATLAS: {atlas_data.shape}")
        print(f"Colonnes ATLAS: {atlas_data.columns.tolist()}")
        
        # Afficher les premières lignes
        print("\nPremières lignes:")
        print(atlas_data.head())
        
        # Vérifier les valeurs manquantes
        print("\nValeurs manquantes par colonne:")
        print(atlas_data.isnull().sum())
        
        # Chercher les colonnes liées au céfidérocol
        cefiderocol_cols = [col for col in atlas_data.columns if 'cefiderocol' in col.lower() or 'cefid' in col.lower()]
        print(f"\nColonnes liées au céfidérocol: {cefiderocol_cols}")
        
        # Chercher les colonnes d'antibiotiques
        antibiotic_cols = [col for col in atlas_data.columns if any(abx in col.lower() for abx in ['meropenem', 'ciprofloxacin', 'colistin', 'amikacin', 'cefepime'])]
        print(f"\nColonnes d'antibiotiques: {antibiotic_cols}")
        
        return atlas_data
        
    except Exception as e:
        print(f"Erreur: {e}")
        return None

if __name__ == "__main__":
    test_atlas_data() 