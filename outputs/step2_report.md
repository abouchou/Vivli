
# Rapport Étape 2 : Développement de Modèles

## Résumé Exécutif

Cette étape a permis de développer et d'évaluer plusieurs modèles de classification pour prédire la résistance au céfidérocol.

## Résultats des Modèles

|                     |      AUC |   Précision |   Rappel |
|:--------------------|---------:|------------:|---------:|
| Logistic Regression | 0.794282 |    0        |    0     |
| Random Forest       | 0.717052 |    0.2      |    0.072 |
| XGBoost             | 0.798473 |    0.409091 |    0.036 |

## Meilleur Modèle

Le meilleur modèle selon l'AUC est: XGBoost

## Métriques Clés

- **AUC moyen**: 0.7699
- **Précision moyenne**: 0.2030
- **Rappel moyen**: 0.0360

## Graphiques Générés

1. `model_evaluation.png` - Courbes ROC et Precision-Recall
2. `feature_importance.png` - Importance des caractéristiques par modèle
3. `shap_analysis.png` - Analyse SHAP du meilleur modèle

## Recommandations

- Le modèle XGBoost montre les meilleures performances
- Considérer l'utilisation de techniques de rééchantillonnage pour gérer le déséquilibre de classes
- Explorer d'autres caractéristiques dérivées pour améliorer les performances
