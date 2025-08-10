import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

def main():
    print("=== ÉTAPE 3: DÉCOUVERTE DES SIGNATURES PHÉNOTYPIQUES (VERSION RAPIDE) ===\n")
    
    # 1. Charger les données
    print("Chargement des données...")
    sidero_data = pd.read_excel("1.xlsx")
    
    # Standardiser les colonnes
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
    
    # Nettoyer les valeurs MIC
    def clean_mic_values(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            import re
            cleaned_value = re.sub(r'[^\d.]', '', value)
            try:
                return float(cleaned_value)
            except ValueError:
                return pd.NA
        return float(value)
    
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[col] = sidero_data[col].apply(clean_mic_values)
            sidero_data[col] = pd.to_numeric(sidero_data[col], errors='coerce')
    
    # Filtrer les données avec des valeurs MIC valides
    sidero_data = sidero_data.dropna(subset=mic_columns)
    print(f"Données après nettoyage: {sidero_data.shape}")
    
    # 2. Préparer les données pour le clustering
    print("\nPréparation des données pour le clustering...")
    X = sidero_data[mic_columns].values
    
    # Transformation log
    X_log = np.log1p(X)
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    # 3. Déterminer le nombre optimal de clusters
    print("Détermination du nombre optimal de clusters...")
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        print(f"k={k}: Silhouette={silhouette_scores[-1]:.3f}")
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nNombre optimal de clusters: {optimal_k}")
    
    # 4. Clustering K-means
    print(f"\nClustering K-means avec {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Ajouter les labels aux données
    sidero_data['cluster'] = cluster_labels
    
    # 5. Analyse PCA
    print("Analyse PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 6. Visualisations
    print("Création des visualisations...")
    
    # PCA avec clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, s=30)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.title('Clusters dans l\'espace PCA')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.savefig("outputs/plots/clusters_pca.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Analyser les signatures de résistance
    print("\nAnalyse des signatures de résistance...")
    breakpoints = {
        "cefiderocol_mic": 4,
        "meropenem_mic": 8,
        "CIPROFLOXACIN_mic": 1,
        "colistin_mic": 4
    }
    
    signatures = {}
    for i in range(optimal_k):
        cluster_data = sidero_data[sidero_data['cluster'] == i]
        
        signature = {}
        resistance_pattern = []
        
        for col in mic_columns:
            resistance_rate = (cluster_data[col] >= breakpoints[col]).mean()
            median_mic = cluster_data[col].median()
            signature[col] = {
                'resistance_rate': resistance_rate,
                'median_mic': median_mic
            }
            
            if resistance_rate > 0.5:
                resistance_pattern.append(f"{col.split('_')[0]}+")
            else:
                resistance_pattern.append(f"{col.split('_')[0]}-")
        
        signature_name = "".join(resistance_pattern)
        signatures[i] = {
            'name': signature_name,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(sidero_data) * 100,
            'details': signature
        }
        
        print(f"\nCluster {i} ({signature_name}):")
        print(f"  Taille: {len(cluster_data)} échantillons ({len(cluster_data)/len(sidero_data)*100:.1f}%)")
        for col in mic_columns:
            print(f"  {col}: {signature[col]['resistance_rate']:.1%} résistance, MIC médiane={signature[col]['median_mic']:.2f}")
    
    # 8. Heatmap des signatures
    print("\nCréation de la heatmap des signatures...")
    resistance_matrix = []
    cluster_names = []
    
    for i in range(optimal_k):
        cluster_data = sidero_data[sidero_data['cluster'] == i]
        cluster_resistance = []
        
        for col in mic_columns:
            resistance_rate = (cluster_data[col] >= breakpoints[col]).mean()
            cluster_resistance.append(resistance_rate)
        
        resistance_matrix.append(cluster_resistance)
        cluster_names.append(f"Cluster {i}\n({signatures[i]['name']})")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(resistance_matrix, 
               xticklabels=[col.split('_')[0].title() for col in mic_columns],
               yticklabels=cluster_names,
               annot=True, 
               fmt='.2f',
               cmap='RdYlBu_r',
               cbar_kws={'label': 'Taux de résistance'})
    
    plt.title('Signatures de résistance par cluster')
    plt.xlabel('Antibiotique')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig("outputs/plots/resistance_signatures_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 9. Générer le rapport
    print("\nGénération du rapport...")
    report = f"""# Étape 3: Découverte des signatures phénotypiques

## Résultats du clustering

### Nombre optimal de clusters
- **Méthode** : Analyse du score de silhouette
- **Nombre optimal** : {optimal_k} clusters
- **Score de silhouette** : {max(silhouette_scores):.3f}

### Distribution des clusters
"""
    
    for i in range(optimal_k):
        report += f"""
**Cluster {i} ({signatures[i]['name']})**
- Taille : {signatures[i]['size']} échantillons ({signatures[i]['percentage']:.1f}%)
"""
        for col in mic_columns:
            ab_name = col.split('_')[0].title()
            report += f"- {ab_name} : {signatures[i]['details'][col]['resistance_rate']:.1%} résistance, MIC médiane = {signatures[i]['details'][col]['median_mic']:.2f}\n"
    
    report += """
## Signatures phénotypiques identifiées

### Interprétation clinique
1. **Profils multirésistants** : Clusters avec résistance à plusieurs antibiotiques
2. **Profils spécifiques** : Résistance sélective à certains antibiotiques
3. **Profils sensibles** : Sensibilité à la plupart des antibiotiques testés

### Applications
- Orientation thérapeutique basée sur les signatures
- Surveillance épidémiologique des profils de résistance
- Développement de tests de diagnostic rapide

## Conclusions

L'analyse de clustering a révélé des patterns distincts dans les profils de résistance, permettant de catégoriser les isolats selon leurs signatures phénotypiques et d'identifier des groupes à risque pour la résistance aux antibiotiques.
"""
    
    with open("outputs/step3_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("=== ÉTAPE 3 TERMINÉE ===")
    print("Résultats sauvegardés :")
    print("- Visualisations : outputs/plots/")
    print("- Rapport : outputs/step3_report.md")

if __name__ == "__main__":
    main() 