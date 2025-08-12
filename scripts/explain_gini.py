import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
import pandas as pd

def explain_gini():
    """Explique l'indice de Gini avec des exemples visuels."""
    
    print("=== Understanding Gini Index in Decision Trees ===\n")
    
    # 1. Définition de l'indice de Gini
    print("1. WHAT IS GINI INDEX?")
    print("   Gini Index = 1 - Σ(p_i²) where p_i is the probability of class i")
    print("   - Measures impurity/purity of a node")
    print("   - Range: 0 (pure) to 1-1/n_classes (impure)")
    print("   - Lower Gini = Better split\n")
    
    # 2. Exemples de calcul
    print("2. GINI INDEX EXAMPLES:")
    
    # Exemple 1: Nœud pur (toutes les classes sont identiques)
    pure_node = [1, 1, 1, 1, 1]  # Tous de classe 1
    gini_pure = 1 - (5/5)**2
    print(f"   Pure node [1,1,1,1,1]: Gini = {gini_pure:.3f} (Perfect!)")
    
    # Exemple 2: Nœud impur (mélange de classes)
    impure_node = [1, 1, 1, 2, 2]  # 3 classe 1, 2 classe 2
    p1 = 3/5
    p2 = 2/5
    gini_impure = 1 - (p1**2 + p2**2)
    print(f"   Impure node [1,1,1,2,2]: Gini = {gini_impure:.3f} (Mixed)")
    
    # Exemple 3: Nœud très impur (équilibre parfait)
    very_impure = [1, 1, 2, 2, 3, 3]  # 2 de chaque classe
    p1 = p2 = p3 = 2/6
    gini_very_impure = 1 - (p1**2 + p2**2 + p3**2)
    print(f"   Very impure [1,1,2,2,3,3]: Gini = {gini_very_impure:.3f} (Worst case)\n")
    
    # 3. Visualisation de l'impureté Gini
    print("3. VISUALIZING GINI IMPURITY:")
    
    # Créer des données d'exemple
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_classes=3, 
                             n_clusters_per_class=1, n_redundant=0, random_state=42)
    
    # Créer un arbre simple
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    # Visualiser l'arbre avec les valeurs Gini
    plt.figure(figsize=(15, 10))
    plot_tree(tree, 
             feature_names=['Feature 1', 'Feature 2'],
             class_names=['Class 0', 'Class 1', 'Class 2'],
             filled=True, 
             rounded=True, 
             fontsize=10,
             proportion=True)
    
    plt.title('Decision Tree with Gini Index Values\n'
              'Each node shows: [class_counts] gini=value samples=total', 
              fontsize=14, pad=20)
    plt.savefig('outputs/plots/gini_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   Tree visualization saved as 'gini_explanation.png'")
    print("   Each node shows: [class_counts] gini=value samples=total\n")
    
    # 4. Calcul de la réduction d'impureté
    print("4. GINI IMPURITY REDUCTION:")
    print("   The tree chooses splits that maximize impurity reduction:")
    print("   Gini Reduction = Gini(parent) - Σ(weight_i × Gini(child_i))")
    print("   Higher reduction = Better split choice\n")
    
    # 5. Exemple avec notre arbre d'antibiotiques
    print("5. IN OUR ANTIBIOTIC TREE:")
    print("   - Gini measures how mixed the antibiotic recommendations are")
    print("   - Lower Gini = more consistent recommendations")
    print("   - Tree splits to minimize Gini (maximize purity)")
    print("   - Each split tries to separate different antibiotic classes\n")
    
    return tree

def create_gini_comparison():
    """Crée une comparaison visuelle de différents niveaux d'impureté Gini."""
    
    # Créer des données avec différents niveaux d'impureté
    np.random.seed(42)
    
    # Nœud pur
    pure_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Nœud légèrement impur
    slightly_impure = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    
    # Nœud très impur
    very_impure = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Calculer les indices Gini
    def calculate_gini(data):
        unique, counts = np.unique(data, return_counts=True)
        probs = counts / len(data)
        return 1 - np.sum(probs**2)
    
    gini_pure = calculate_gini(pure_data)
    gini_slight = calculate_gini(slightly_impure)
    gini_very = calculate_gini(very_impure)
    
    # Visualiser
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Nœud pur
    axes[0].bar(['Class 0', 'Class 1'], [10, 0], color=['blue', 'red'])
    axes[0].set_title(f'Pure Node\nGini = {gini_pure:.3f}')
    axes[0].set_ylabel('Count')
    
    # Nœud légèrement impur
    axes[1].bar(['Class 0', 'Class 1'], [8, 2], color=['blue', 'red'])
    axes[1].set_title(f'Slightly Impure\nGini = {gini_slight:.3f}')
    
    # Nœud très impur
    axes[2].bar(['Class 0', 'Class 1'], [5, 5], color=['blue', 'red'])
    axes[2].set_title(f'Very Impure\nGini = {gini_very:.3f}')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/gini_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   Gini comparison visualization saved as 'gini_comparison.png'")
    print(f"   Pure node Gini: {gini_pure:.3f}")
    print(f"   Slightly impure Gini: {gini_slight:.3f}")
    print(f"   Very impure Gini: {gini_very:.3f}")

if __name__ == "__main__":
    print("=== Gini Index Explanation ===\n")
    
    # Expliquer Gini
    tree = explain_gini()
    
    # Créer la comparaison visuelle
    create_gini_comparison()
    
    print("\n=== Summary ===")
    print("Gini Index tells us:")
    print("• How 'mixed' or 'pure' a node is")
    print("• Lower Gini = more consistent predictions")
    print("• Tree splits to minimize Gini")
    print("• Helps choose the best features to split on")
    print("\nCheck the generated images for visual examples!")
