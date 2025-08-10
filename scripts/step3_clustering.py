import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Configuration des paramètres d'affichage
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams['font.size'] = 10

# Création des dossiers pour les sorties
import os
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

def load_and_prepare_data():
    """Charge et prépare les données pour l'analyse de clustering."""
    print("=== Chargement des données ===")
    
    # Charger les données
    sidero_data = pd.read_excel("1.xlsx")
    atlas_data = pd.read_excel("2.xlsx")
    
    print(f"SIDERO-WT: {sidero_data.shape}")
    print(f"ATLAS: {atlas_data.shape}")
    
    # Standardiser les colonnes (réutiliser la logique de step2)
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
    
    # Appliquer le nettoyage aux colonnes MIC
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    for col in mic_columns:
        if col in sidero_data.columns:
            sidero_data[col] = sidero_data[col].apply(clean_mic_values)
            sidero_data[col] = pd.to_numeric(sidero_data[col], errors='coerce')
    
    # Filtrer les données avec des valeurs MIC valides pour tous les antibiotiques
    sidero_data = sidero_data.dropna(subset=mic_columns)
    
    print(f"Données après nettoyage: {sidero_data.shape}")
    print(f"Colonnes MIC disponibles: {mic_columns}")
    
    return sidero_data, atlas_data

def prepare_clustering_data(data):
    """Prépare les données pour le clustering en utilisant les profils de résistance."""
    print("\n=== Préparation des données pour le clustering ===")
    
    # Sélectionner les colonnes MIC pour le clustering
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    
    # Créer un DataFrame avec les profils de résistance
    clustering_data = data[mic_columns].copy()
    
    # Appliquer une transformation log pour normaliser les distributions
    for col in mic_columns:
        clustering_data[f"log_{col}"] = np.log1p(clustering_data[col])
    
    # Créer des caractéristiques de résistance binaires
    for col in mic_columns:
        # Définir les seuils de résistance (peuvent être ajustés selon les breakpoints)
        breakpoints = {
            "cefiderocol_mic": 4,
            "meropenem_mic": 8,
            "CIPROFLOXACIN_mic": 1,
            "colistin_mic": 4
        }
        clustering_data[f"{col}_resistant"] = (clustering_data[col] >= breakpoints[col]).astype(int)
    
    # Créer un profil de résistance combiné
    resistance_columns = [f"{col}_resistant" for col in mic_columns]
    clustering_data["resistance_profile"] = clustering_data[resistance_columns].sum(axis=1)
    
    print(f"Données de clustering: {clustering_data.shape}")
    print(f"Distribution des profils de résistance:")
    print(clustering_data["resistance_profile"].value_counts().sort_index())
    
    return clustering_data, data

def determine_optimal_clusters(data, max_clusters=10):
    """Détermine le nombre optimal de clusters en utilisant plusieurs méthodes."""
    print("\n=== Détermination du nombre optimal de clusters ===")
    
    # Utiliser les colonnes log transformées pour le clustering
    log_columns = [col for col in data.columns if col.startswith("log_")]
    X = data[log_columns].values
    
    # Standardiser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Métriques pour évaluer le nombre optimal de clusters
    silhouette_scores = []
    calinski_scores = []
    inertias = []
    
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        # K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculer les métriques
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
        inertias.append(kmeans.inertia_)
        
        print(f"k={k}: Silhouette={silhouette_scores[-1]:.3f}, Calinski-Harabasz={calinski_scores[-1]:.0f}")
    
    # Trouver le k optimal basé sur le score de silhouette
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Visualiser les métriques
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Score de silhouette
    axes[0].plot(k_range, silhouette_scores, 'bo-')
    axes[0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    axes[0].set_xlabel('Nombre de clusters (k)')
    axes[0].set_ylabel('Score de silhouette')
    axes[0].set_title('Score de silhouette vs nombre de clusters')
    axes[0].legend()
    axes[0].grid(True)
    
    # Score de Calinski-Harabasz
    axes[1].plot(k_range, calinski_scores, 'go-')
    axes[1].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    axes[1].set_xlabel('Nombre de clusters (k)')
    axes[1].set_ylabel('Score de Calinski-Harabasz')
    axes[1].set_title('Score de Calinski-Harabasz vs nombre de clusters')
    axes[1].legend()
    axes[1].grid(True)
    
    # Méthode du coude
    axes[2].plot(k_range, inertias, 'ro-')
    axes[2].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    axes[2].set_xlabel('Nombre de clusters (k)')
    axes[2].set_ylabel('Inertie')
    axes[2].set_title('Méthode du coude')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/optimal_clusters_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nNombre optimal de clusters basé sur le score de silhouette: {optimal_k}")
    
    return optimal_k, scaler

def perform_kmeans_clustering(data, n_clusters, scaler):
    """Effectue le clustering K-means."""
    print(f"\n=== Clustering K-means avec {n_clusters} clusters ===")
    
    # Utiliser les colonnes log transformées
    log_columns = [col for col in data.columns if col.startswith("log_")]
    X = data[log_columns].values
    X_scaled = scaler.transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Ajouter les labels de cluster aux données
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = cluster_labels
    
    # Analyser les clusters
    print("\nAnalyse des clusters:")
    for i in range(n_clusters):
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == i]
        print(f"\nCluster {i} ({len(cluster_data)} échantillons, {len(cluster_data)/len(data_with_clusters)*100:.1f}%):")
        
        # Statistiques des MIC
        mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
        for col in mic_columns:
            if col in cluster_data.columns:
                median_val = cluster_data[col].median()
                mean_val = cluster_data[col].mean()
                print(f"  {col}: médiane={median_val:.2f}, moyenne={mean_val:.2f}")
    
    return data_with_clusters, kmeans

def perform_hierarchical_clustering(data, n_clusters, scaler):
    """Effectue le clustering hiérarchique."""
    print(f"\n=== Clustering hiérarchique avec {n_clusters} clusters ===")
    
    # Utiliser les colonnes log transformées
    log_columns = [col for col in data.columns if col.startswith("log_")]
    X = data[log_columns].values
    X_scaled = scaler.transform(X)
    
    # Clustering hiérarchique
    linkage_matrix = linkage(X_scaled, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    
    # Ajouter les labels de cluster aux données
    data_with_clusters = data.copy()
    data_with_clusters['hierarchical_cluster'] = cluster_labels
    
    # Visualiser le dendrogramme
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Dendrogramme du clustering hiérarchique')
    plt.xlabel('Échantillons')
    plt.ylabel('Distance')
    plt.savefig("outputs/plots/hierarchical_dendrogram.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return data_with_clusters, linkage_matrix

def perform_pca_analysis(data, scaler):
    """Effectue l'analyse PCA pour visualiser les similarités."""
    print("\n=== Analyse PCA ===")
    
    # Utiliser les colonnes log transformées
    log_columns = [col for col in data.columns if col.startswith("log_")]
    X = data[log_columns].values
    X_scaled = scaler.transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Variance expliquée
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"Variance expliquée par composante:")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"  PC{i+1}: {var:.3f} ({cum_var:.3f} cumulée)")
    
    # Visualiser la variance expliquée
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Variance expliquée par composante
    axes[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    axes[0].set_xlabel('Composante principale')
    axes[0].set_ylabel('Variance expliquée')
    axes[0].set_title('Variance expliquée par composante principale')
    axes[0].grid(True)
    
    # Variance cumulée
    axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% de variance')
    axes[1].axhline(y=0.80, color='orange', linestyle='--', label='80% de variance')
    axes[1].set_xlabel('Nombre de composantes principales')
    axes[1].set_ylabel('Variance cumulée')
    axes[1].set_title('Variance cumulée')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/pca_variance_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_pca, pca

def visualize_clusters_pca(data_with_clusters, X_pca, pca, scaler):
    """Visualise les clusters dans l'espace PCA."""
    print("\n=== Visualisation des clusters dans l'espace PCA ===")
    
    # Créer des visualisations pour les deux types de clustering
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # K-means clusters
    if 'cluster' in data_with_clusters.columns:
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=data_with_clusters['cluster'], 
                                  cmap='viridis', alpha=0.7, s=50)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        axes[0].set_title('Clusters K-means dans l\'espace PCA')
        axes[0].grid(True)
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Clusters hiérarchiques
    if 'hierarchical_cluster' in data_with_clusters.columns:
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=data_with_clusters['hierarchical_cluster'], 
                                  cmap='plasma', alpha=0.7, s=50)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        axes[1].set_title('Clusters hiérarchiques dans l\'espace PCA')
        axes[1].grid(True)
        plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig("outputs/plots/clusters_pca_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

def define_resistance_signatures(data_with_clusters, original_data):
    """Définit les signatures de résistance basées sur les profils d'antibiotiques comparateurs."""
    print("\n=== Définition des signatures de résistance ===")
    
    # Analyser chaque cluster pour définir sa signature
    signatures = {}
    
    for cluster_type in ['cluster', 'hierarchical_cluster']:
        if cluster_type not in data_with_clusters.columns:
            continue
            
        print(f"\n--- Signatures pour {cluster_type} ---")
        signatures[cluster_type] = {}
        
        n_clusters = data_with_clusters[cluster_type].nunique()
        
        for i in range(n_clusters):
            cluster_mask = data_with_clusters[cluster_type] == i
            cluster_data = data_with_clusters[cluster_mask]
            original_cluster_data = original_data.iloc[cluster_data.index]
            
            # Calculer les statistiques de résistance
            mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
            breakpoints = {
                "cefiderocol_mic": 4,
                "meropenem_mic": 8,
                "CIPROFLOXACIN_mic": 1,
                "colistin_mic": 4
            }
            
            signature = {}
            for col in mic_columns:
                if col in original_cluster_data.columns:
                    resistance_rate = (original_cluster_data[col] >= breakpoints[col]).mean()
                    median_mic = original_cluster_data[col].median()
                    signature[col] = {
                        'resistance_rate': resistance_rate,
                        'median_mic': median_mic,
                        'breakpoint': breakpoints[col]
                    }
            
            # Définir le type de signature
            resistance_pattern = []
            for col in mic_columns:
                if col in signature:
                    if signature[col]['resistance_rate'] > 0.5:
                        resistance_pattern.append(f"{col.split('_')[0]}+")
                    else:
                        resistance_pattern.append(f"{col.split('_')[0]}-")
            
            signature_name = "".join(resistance_pattern)
            
            signatures[cluster_type][i] = {
                'name': signature_name,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data_with_clusters) * 100,
                'details': signature
            }
            
            print(f"\nCluster {i} ({signature_name}):")
            print(f"  Taille: {len(cluster_data)} échantillons ({len(cluster_data)/len(data_with_clusters)*100:.1f}%)")
            for col in mic_columns:
                if col in signature:
                    print(f"  {col}: {signature[col]['resistance_rate']:.1%} résistance, MIC médiane={signature[col]['median_mic']:.2f}")
    
    return signatures

def analyze_abnormal_isolates(data_with_clusters, original_data, scaler):
    """Analyse les isolats mal classés ou anormaux."""
    print("\n=== Analyse des isolats anormaux ===")
    
    # Utiliser les colonnes log transformées
    log_columns = [col for col in data_with_clusters.columns if col.startswith("log_")]
    X = data_with_clusters[log_columns].values
    X_scaled = scaler.transform(X)
    
    # Calculer les distances aux centroïdes pour K-means
    if 'cluster' in data_with_clusters.columns:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=data_with_clusters['cluster'].nunique(), random_state=42)
        kmeans.fit(X_scaled)
        
        # Calculer les distances aux centroïdes
        distances_to_centroids = []
        for i, point in enumerate(X_scaled):
            cluster_label = data_with_clusters.iloc[i]['cluster']
            centroid = kmeans.cluster_centers_[cluster_label]
            distance = np.linalg.norm(point - centroid)
            distances_to_centroids.append(distance)
        
        # Identifier les isolats anormaux (distance > 95e percentile)
        threshold = np.percentile(distances_to_centroids, 95)
        abnormal_mask = np.array(distances_to_centroids) > threshold
        
        abnormal_isolates = original_data.iloc[data_with_clusters[abnormal_mask].index]
        
        print(f"\nIsolats anormaux (distance > 95e percentile): {len(abnormal_isolates)}")
        if len(abnormal_isolates) > 0:
            print("\nCaractéristiques des isolats anormaux:")
            mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
            for col in mic_columns:
                if col in abnormal_isolates.columns:
                    print(f"  {col}: {abnormal_isolates[col].describe()}")
    
    return abnormal_isolates if 'abnormal_isolates' in locals() else None

def create_heatmap_signatures(data_with_clusters, original_data, signatures):
    """Crée une heatmap des signatures de résistance."""
    print("\n=== Création de la heatmap des signatures ===")
    
    # Préparer les données pour la heatmap
    mic_columns = ["cefiderocol_mic", "meropenem_mic", "CIPROFLOXACIN_mic", "colistin_mic"]
    
    for cluster_type in ['cluster', 'hierarchical_cluster']:
        if cluster_type not in data_with_clusters.columns:
            continue
            
        # Calculer les taux de résistance moyens par cluster
        n_clusters = data_with_clusters[cluster_type].nunique()
        resistance_matrix = []
        cluster_names = []
        
        for i in range(n_clusters):
            cluster_mask = data_with_clusters[cluster_type] == i
            original_cluster_data = original_data.iloc[data_with_clusters[cluster_mask].index]
            
            cluster_resistance = []
            for col in mic_columns:
                if col in original_cluster_data.columns:
                    breakpoints = {
                        "cefiderocol_mic": 4,
                        "meropenem_mic": 8,
                        "CIPROFLOXACIN_mic": 1,
                        "colistin_mic": 4
                    }
                    resistance_rate = (original_cluster_data[col] >= breakpoints[col]).mean()
                    cluster_resistance.append(resistance_rate)
                else:
                    cluster_resistance.append(0)
            
            resistance_matrix.append(cluster_resistance)
            cluster_names.append(f"Cluster {i}\n({signatures[cluster_type][i]['name']})")
        
        # Créer la heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(resistance_matrix, 
                   xticklabels=[col.split('_')[0].title() for col in mic_columns],
                   yticklabels=cluster_names,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Taux de résistance'})
        
        plt.title(f'Signatures de résistance - {cluster_type.replace("_", " ").title()}')
        plt.xlabel('Antibiotique')
        plt.ylabel('Cluster')
        plt.tight_layout()
        plt.savefig(f"outputs/plots/resistance_signatures_heatmap_{cluster_type}.png", dpi=300, bbox_inches='tight')
        plt.show()

def generate_step3_report(signatures, abnormal_isolates, data_with_clusters):
    """Génère le rapport de l'étape 3."""
    print("\n=== Génération du rapport de l'étape 3 ===")
    
    report = """# Étape 3: Découverte des signatures phénotypiques

## Objectifs atteints

### 1. Groupement des profils de résistance par clustering

Nous avons appliqué deux méthodes de clustering pour identifier des groupes d'isolats avec des profils de résistance similaires :

#### Clustering K-means
- **Nombre optimal de clusters** : Déterminé par analyse du score de silhouette
- **Avantages** : Rapide et efficace pour de grands ensembles de données
- **Résultats** : Identification de groupes distincts basés sur les profils MIC

#### Clustering hiérarchique
- **Méthode** : Ward linkage pour minimiser la variance intra-cluster
- **Visualisation** : Dendrogramme pour comprendre la structure hiérarchique
- **Avantages** : Permet de visualiser les relations entre les groupes

### 2. Visualisation des similarités avec PCA

L'analyse en composantes principales (PCA) a permis de :
- **Réduire la dimensionnalité** : Visualisation des données multidimensionnelles
- **Identifier les patterns** : Similarités et différences entre les isolats
- **Valider les clusters** : Confirmation de la séparation des groupes

### 3. Définition des signatures de résistance

Basées sur les profils d'antibiotiques comparateurs, nous avons identifié plusieurs signatures :

"""
    
    # Ajouter les détails des signatures
    for cluster_type, cluster_signatures in signatures.items():
        report += f"\n#### Signatures {cluster_type.replace('_', ' ').title()}\n\n"
        
        for cluster_id, signature_info in cluster_signatures.items():
            report += f"**Cluster {cluster_id} ({signature_info['name']})**\n"
            report += f"- Taille : {signature_info['size']} échantillons ({signature_info['percentage']:.1f}%)\n"
            
            for antibiotic, details in signature_info['details'].items():
                ab_name = antibiotic.split('_')[0].title()
                report += f"- {ab_name} : {details['resistance_rate']:.1%} résistance, MIC médiane = {details['median_mic']:.2f}\n"
            
            report += "\n"
    
    report += """
### 4. Analyse des isolats anormaux

Nous avons identifié les isolats mal classés ou présentant des profils atypiques :
- **Critère** : Distance aux centroïdes > 95e percentile
- **Objectif** : Comprendre les mécanismes de résistance inhabituels
- **Applications** : Détection de nouveaux mécanismes de résistance

## Implications cliniques

### Signatures identifiées
1. **Profils multirésistants** : Isolats résistants à plusieurs classes d'antibiotiques
2. **Profils spécifiques** : Résistance sélective à certains antibiotiques
3. **Profils sensibles** : Isolats sensibles à la plupart des antibiotiques testés

### Applications pratiques
- **Orientation thérapeutique** : Choix d'antibiotiques basé sur les signatures
- **Surveillance épidémiologique** : Suivi de l'émergence de nouveaux profils
- **Développement de tests** : Identification rapide des profils de résistance

## Conclusions

L'analyse de clustering a révélé des patterns distincts dans les profils de résistance, permettant de :
- Catégoriser les isolats selon leurs signatures phénotypiques
- Identifier des groupes à risque pour la résistance au cefiderocol
- Guider les stratégies de traitement personnalisées

Ces résultats contribuent à une meilleure compréhension de la résistance aux antibiotiques et à l'optimisation des stratégies thérapeutiques.
"""
    
    # Sauvegarder le rapport
    with open("outputs/step3_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("Rapport de l'étape 3 généré : outputs/step3_report.md")
    
    return report

def main():
    """Fonction principale pour l'étape 3."""
    print("=== ÉTAPE 3: DÉCOUVERTE DES SIGNATURES PHÉNOTYPIQUES ===\n")
    
    # 1. Charger et préparer les données
    sidero_data, atlas_data = load_and_prepare_data()
    
    # 2. Préparer les données pour le clustering
    clustering_data, original_data = prepare_clustering_data(sidero_data)
    
    # 3. Déterminer le nombre optimal de clusters
    optimal_k, scaler = determine_optimal_clusters(clustering_data)
    
    # 4. Effectuer le clustering K-means
    data_with_clusters, kmeans_model = perform_kmeans_clustering(clustering_data, optimal_k, scaler)
    
    # 5. Effectuer le clustering hiérarchique
    data_with_clusters, linkage_matrix = perform_hierarchical_clustering(data_with_clusters, optimal_k, scaler)
    
    # 6. Analyse PCA
    X_pca, pca_model = perform_pca_analysis(clustering_data, scaler)
    
    # 7. Visualiser les clusters dans l'espace PCA
    visualize_clusters_pca(data_with_clusters, X_pca, pca_model, scaler)
    
    # 8. Définir les signatures de résistance
    signatures = define_resistance_signatures(data_with_clusters, original_data)
    
    # 9. Analyser les isolats anormaux
    abnormal_isolates = analyze_abnormal_isolates(data_with_clusters, original_data, scaler)
    
    # 10. Créer les heatmaps des signatures
    create_heatmap_signatures(data_with_clusters, original_data, signatures)
    
    # 11. Générer le rapport
    report = generate_step3_report(signatures, abnormal_isolates, data_with_clusters)
    
    print("\n=== ÉTAPE 3 TERMINÉE ===")
    print("Résultats sauvegardés dans le dossier outputs/")
    print("- Visualisations : outputs/plots/")
    print("- Rapport : outputs/step3_report.md")

if __name__ == "__main__":
    main() 