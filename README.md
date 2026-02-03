# SIAHA — Système Intelligent d'Analyse des Flux Touristiques

> Prédiction de l'intensité des flux touristiques aux postes-frontières du Maroc via classification Bayésienne.

![Maroc Tourisme](https://img.shields.io/badge/Maroc-Tourisme-C4562A?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-2A5C8A?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Naive%20Bayes-D4A832?style=flat-square)
![Status](https://img.shields.io/badge/Statut-Académique-5B9E6F?style=flat-square)

---

## 🌍 Contexte

Ce projet a été développé dans le cadre de la **préparation stratégique au Mondial 2030** coorganisé par le Maroc, l'Espagne et le Portugal. Il vise à fournir aux décideurs un outil d'aide à la décision basé sur l'apprentissage automatique pour anticiper les pics de flux touristiques aux postes-frontières.

---

## 📁 Structure du projet

```
siaha/
│
├── 📓 Tourisme_Classification_Bayes_Bayes.ipynb   # Notebook principal (modèle complet)
│
├── 📊 Données source
│   ├── evolution-par-nationalite-des-arrivees-des-touristes-aux-postes-frontieres.xlsx
│   ├── evolution-par-point-dentree-des-arrivees-des-touristes-aux-postes-frontieres.xlsx
│   └── evolution-nuitees-realisees-etablissements-hebergement-touristique-par-destination.xlsx
│
├── 🤖 Modèle sérialisé
│   ├── model_naivebayes.pkl    # Modèle Naive Bayes Multinomial
│   ├── le_nat.pkl              # Encodeur des nationalités
│   ├── le_prov.pkl             # Encodeur des provinces/postes
│   ├── le_trim.pkl             # Encodeur du trimestre
│   └── le_inter.pkl            # Encodeur d'interaction (nat × province)
│
└── 🌐 Interface web
    └── index.html              # Application web interactive
```

---

## 🧠 Méthodologie

### Pipeline ML

| Étape | Description |
|-------|-------------|
| **1. Collecte** | 3 fichiers Excel ONMT (2012–2020) fusionnés via `pandas.merge()` |
| **2. Nettoyage** | Suppression des NaN, harmonisation des colonnes |
| **3. Feature Eng.** | LabelEncoding + saisonnalité (trimestre) + interaction Nat × Province |
| **4. Modèle** | `MultinomialNB` — P(Flux Élevé \| features) via théorème de Bayes |
| **5. Évaluation** | Accuracy ~85%, rapport de classification, matrice de confusion |
| **6. Déploiement** | Sérialisation `joblib` en fichiers `.pkl` |

### Variables d'entrée

- `nat_encoded` — Nationalité du touriste (LabelEncoder)
- `prov_encoded` — Poste-frontière / province (LabelEncoder)
- `trim_encoded` — Trimestre 1–4 (saisonnalité)
- `inter_encoded` — Interaction Nationalité × Province

### Variable cible

- `target_flux` : `1` si flux > médiane (Flux Élevé), `0` sinon

---

## 📊 Données

- **Source** : Office National Marocain du Tourisme (ONMT)
- **Période** : 2012–2020
- **Nationalités** : 13 (France, Espagne, Royaume-Uni, Allemagne, Italie, États-Unis, Belgique, Hollande, Maghreb, Chine, Scandinavie, MRE, Total)
- **Postes-frontières** : 19 (aéroports, ports, postes terrestres)

---

## 🚀 Installation & Utilisation

### Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
```

### Lancer le notebook

```bash
jupyter notebook Tourisme_Classification_Bayes_Bayes.ipynb
```

### Utiliser le modèle sérialisé

```python
import joblib
import numpy as np

# Charger le modèle et les encodeurs
model   = joblib.load('model_naivebayes.pkl')
le_nat  = joblib.load('le_nat.pkl')
le_prov = joblib.load('le_prov.pkl')
le_trim = joblib.load('le_trim.pkl')
le_inter = joblib.load('le_inter.pkl')

# Prédire
nat    = le_nat.transform(['France'])[0]
prov   = le_prov.transform(['A Tanger Ibn Battouta'])[0]
trim   = le_trim.transform([3])[0]
inter  = le_inter.transform([f'{nat}_{prov}'])[0]

proba = model.predict_proba([[nat, prov, trim, inter]])[0]
print(f"P(Flux Élevé) = {proba[1]*100:.2f}%")
```

### Interface web

Ouvrez simplement `index.html` dans un navigateur — aucune dépendance serveur requise.

---

## 🌐 Interface Web

L'application `index.html` offre :

- **Simulateur de prédiction** interactif (nationalité × poste × trimestre)
- **Visualisations** : évolution historique, parts de marché, impact COVID
- **Métriques** du modèle et pipeline pédagogique

---

## 🏆 Application : Mondial 2030

Le Maroc co-organise la Coupe du Monde FIFA 2030. SIAHA permet d'anticiper :

- Les pics de flux par nationalité (Espagne, France, etc.)
- Les goulets d'étranglement aux postes (Aéroport Mohammed V, Tanger Med, Bab Sebta)
- Les trimestres critiques (T3 : juin–juillet, période de compétition)

---

## 👤 Auteur

Projet académique — Classification Bayésienne appliquée au tourisme marocain.

---

## 📄 Licence

Usage académique et éducatif. Données ONMT.
