# Rapport du Système de Recommandation d'Antibiotiques

## Vue d'ensemble
Ce système utilise un arbre de décision pour recommander les antibiotiques dans l'ordre optimal d'efficacité, avec la céfidérocol comme option de dernier recours.

## Ordre Recommandé des Antibiotiques
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
21. Tebipenem
22. Cefpodoxime
23. Cefixime
24. Ceftolozane tazobactam
25. Penicillin
26. Ceftibuten avibactam
27. Moxifloxacin
28. Colistin
29. Aztreonam avibactam
30. Minocycline
31. Clindamycin
32. Ampicillin sulbactam
33. Amikacin
34. Gentamicin
35. Imipenem
36. Ciprofloxacin
37. Trimethoprim sulfa
38. Meropenem
39. Oxacillin
40. Ceftriaxone
41. Ceftaroline
42. Cefepime
43. Erythromycin
44. Piperacillin tazobactam
45. Ceftazidime
46. Amoxycillin clavulanate
47. Levofloxacin
48. Ampicillin
49. Cefiderocol


## Caractéristiques du Modèle
- **Espèce bactérienne** : Facteur principal pour déterminer la sensibilité
- **Pays/Région** : Influence sur les profils de résistance locaux
- **Année** : Évolution temporelle des résistances
- **Profil de résistance** : Résistance par classe d'antibiotiques

## Utilisation
```python
# Exemple d'utilisation
recommendations = model.recommend_antibiotics(
    species="Escherichia coli",
    country="France", 
    year=2023,
    resistance_profile={{'beta_lactam': 0.3, 'quinolone': 0.7}}
)
```

## Recommandations
1. **Première ligne** : Antibiotique le plus efficace selon le modèle
2. **Lignes suivantes** : Alternatives en cas d'échec
3. **Dernière ligne** : Céfidérocol (dernier recours)

## Notes importantes
- Le modèle est basé sur les données ATLAS
- Les recommandations doivent être validées cliniquement
- La céfidérocol est réservée aux cas de résistance multiple
