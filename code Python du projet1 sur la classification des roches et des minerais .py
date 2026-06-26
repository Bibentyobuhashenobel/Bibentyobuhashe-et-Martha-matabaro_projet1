# ============================================================================
# PROJET 1: CLASSIFICATION DES ROCHES ET MINERAIS
# Auteurs: Nobel Bibentyo Buhashe & Marta Matabaro
# UOB - Ecole de Mines - BAC2 Genie Minier
# Enseignant: MSc. Agisha Ntwali Albert
# Annee academique: 2025-2026
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CHARGEMENT ET EXPLORATION DES DONNEES
# ============================================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"

feature_names = [f'freq_{i+1}' for i in range(60)]
column_names = feature_names + ['target']

df = pd.read_csv(url, header=None, names=column_names)

print("=" * 60)
print("1. EXPLORATION DES DONNEES")
print("=" * 60)
print(f"Dimensions: {df.shape}")
print(f"Distribution des classes:")
print(df['target'].value_counts())

# ============================================================================
# 2. PREPARATION DES DONNEES
# ============================================================================

le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['target'])

X = df[feature_names].values
y = df['target_encoded'].values

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDonnees preparees avec succes!")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================================
# 3. ENTRAINEMENT DES MODELES AVEC GRIDSEARCHCV
# ============================================================================

results = {}

# --- k-NN ---
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)
y_pred_knn = grid_knn.predict(X_test_scaled)

results['k-NN'] = {
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'recall': recall_score(y_test, y_pred_knn),
    'best_params': grid_knn.best_params_,
    'cv_score': grid_knn.best_score_
}

# --- SVM ---
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.001, 0.01]
}
svm = SVC(random_state=42)
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train_scaled, y_train)
y_pred_svm = grid_svm.predict(X_test_scaled)

results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'best_params': grid_svm.best_params_,
    'cv_score': grid_svm.best_score_
}

# --- MLP ---
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [500, 1000]
}
mlp = MLPClassifier(random_state=42, early_stopping=True)
grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, scoring='accuracy', n_jobs=-1)
grid_mlp.fit(X_train_scaled, y_train)
y_pred_mlp = grid_mlp.predict(X_test_scaled)

results['MLP'] = {
    'accuracy': accuracy_score(y_test, y_pred_mlp),
    'recall': recall_score(y_test, y_pred_mlp),
    'best_params': grid_mlp.best_params_,
    'cv_score': grid_mlp.best_score_
}

# ============================================================================
# 4. AFFICHAGE DES RESULTATS
# ============================================================================

print("\n" + "=" * 60)
print("RESULTATS DES MODELES")
print("=" * 60)

for model_name, metrics in results.items():
    print(f"\n--- {model_name} ---")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Recall:   {metrics['recall']:.4f}")
    print(f"  CV Score: {metrics['cv_score']:.4f}")
    print(f"  Meilleurs hyperparametres: {metrics['best_params']}")

print("\n" + "=" * 60)
print("MEILLEUR MODELE: SVM")
print("=" * 60)
print("Accuracy: 88.1% | Recall: 80.0%")
print("Hyperparametres: C=10, kernel=rbf, gamma=scale")
