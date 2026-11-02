"""
================================================================================
PROYECTO 4: CLASIFICACIÓN PARA AUTENTICACIÓN DE BILLETES
Comparación experimental: SVM (kernel RBF) vs Random Forest
================================================================================

Dataset : Banknote Authentication (UCI ML Repository)
          data_banknote_authentication.txt (CSV sin cabecera)
Columnas: variance, skewness, curtosis, entropy, class
Clases  : 0 = billete genuino, 1 = billete falsificado

CONTEXTO / COSTOS DE ERROR (Sec. 1)
-----------------------------------
- Falso Negativo (predecir 0 cuando es 1) = aceptar una falsificación
  -> pérdida económica directa. ERROR COSTOSO.
- Falso Positivo (predecir 1 cuando es 0) = rechazar un billete genuino
  -> revisión adicional, afecta la operación pero no la pérdida directa.
=> Priorizamos RECALL de la clase 1. Usamos F2 (beta=2).

REQUISITOS
----------
    pip install numpy pandas matplotlib seaborn scikit-learn scipy

USO
---
    1) Colocar 'data_banknote_authentication.txt' en la misma carpeta
       (o ajustar DATA_PATH).
    2) python proyecto_billetes.py
    3) Resultados en ./resultados/

ESTRUCTURA DEL SCRIPT (alineada con las 7 secciones del PDF)
------------------------------------------------------------
    [S1] Formulación del problema + carga y EDA
    [S2] Selección de métricas
    [S3] Selección de algoritmos
    [S4] Preparación (pipelines)
    [S5] Diseño experimental (Nested CV)
    [S6] Análisis estadístico (Wilcoxon)
    [S7] Discusión + evaluación final en test + figuras + resultados
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, StratifiedKFold,
    GridSearchCV, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    fbeta_score, f1_score, recall_score, precision_score, roc_auc_score,
    balanced_accuracy_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score, make_scorer
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
SEED = 42
np.random.seed(SEED)

DATA_PATH  = 'data_banknote_authentication.txt'
OUTPUT_DIR = 'resultados'
FIG_DIR    = os.path.join(OUTPUT_DIR, 'figuras')
os.makedirs(FIG_DIR, exist_ok=True)

COLUMNS = ['variance', 'skewness', 'curtosis', 'entropy', 'class']


# =============================================================================
# [S1] FORMULACIÓN DEL PROBLEMA — CARGA Y EDA
# =============================================================================
print("=" * 78)
print(" PROYECTO 4 — AUTENTICACIÓN DE BILLETES ")
print("=" * 78)

# --- Pregunta de investigación (para el artículo) ------------------------
# PI: ¿Difiere el desempeño de SVM (RBF) y Random Forest en la detección
#     de billetes falsificados (clase 1), medido con F2-score, bajo un
#     diseño de validación cruzada repetida y estratificada?
# H0: no hay diferencia en la mediana del F2-score entre ambos algoritmos.
# H1: sí hay diferencia.

df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
print(f"\nDimensiones: {df.shape}")
print(f"\nDistribución de clases:\n{df['class'].value_counts().to_string()}")
print(f"\nProporción:\n{df['class'].value_counts(normalize=True).round(4).to_string()}")
print(f"\nValores faltantes totales: {df.isnull().sum().sum()}")
print(f"\nEstadísticas descriptivas:\n{df.describe().round(3).to_string()}")

# --- EDA: distribuciones por clase --------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(COLUMNS[:-1]):
    ax = axes[i // 2, i % 2]
    df.boxplot(column=col, by='class', ax=ax)
    ax.set_title(f'{col} por clase')
    ax.set_xlabel('Clase (0=Genuino, 1=Falsificado)')
plt.suptitle('EDA — Distribución de predictores por clase')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '01_eda_boxplots.png'), dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('EDA — Matriz de correlación')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '02_eda_correlacion.png'), dpi=150)
plt.close()


# =============================================================================
# SPLIT TRAIN / TEST (estratificado)
# =============================================================================
X = df.drop('class', axis=1).values
y = df['class'].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Clases train: {np.bincount(y_train)} | Clases test: {np.bincount(y_test)}")


# =============================================================================
# [S2] SELECCIÓN DE MÉTRICAS
# =============================================================================
# Métrica PRINCIPAL: F2-score de la clase 1 (beta=2 pondera recall)
# Complementarias: Recall1, Precision1, F1_1, ROC-AUC, Balanced Accuracy
# ----------------------------------------------------------------------
SCORING = {
    'f2_class1'       : make_scorer(fbeta_score, beta=2, pos_label=1),
    'f1_class1'       : make_scorer(f1_score,         pos_label=1),
    'recall_class1'   : make_scorer(recall_score,     pos_label=1),
    'precision_class1': make_scorer(precision_score,  pos_label=1),
    'roc_auc'         : 'roc_auc',
    'balanced_acc'    : 'balanced_accuracy',
    'accuracy'        : 'accuracy',
}
MAIN_METRIC = 'test_f2_class1'


# =============================================================================
# [S3] SELECCIÓN DE ALGORITMOS + [S4] PREPARACIÓN
# =============================================================================
# SVM (RBF):  necesita estandarización -> StandardScaler + SVC
# RandomForest: invariante a escala -> solo RandomForestClassifier
# ----------------------------------------------------------------------
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    SVC(kernel='rbf', probability=True, random_state=SEED))
])

rf_pipe = Pipeline([
    ('clf', RandomForestClassifier(random_state=SEED, n_jobs=-1))
])

param_grid_svm = {
    'clf__C'    : [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 0.01, 0.1, 1],
}

param_grid_rf = {
    'clf__n_estimators'    : [200, 500],
    'clf__max_depth'       : [None, 10, 20],
    'clf__min_samples_leaf': [1, 2, 4],
}


# =============================================================================
# [S5] DISEÑO EXPERIMENTAL — NESTED CROSS-VALIDATION
# =============================================================================
# - Outer: RepeatedStratifiedKFold(5 splits × 5 repeticiones) = 25 folds
# - Inner: StratifiedKFold(3) para GridSearchCV
# - Se reportan las 25 puntuaciones pareadas de la métrica principal.
# - El escalador y el GridSearch se reajustan DENTRO de cada fold externo
#   (no hay fuga de información).
# ----------------------------------------------------------------------
print("\n" + "=" * 78)
print(" [S5] NESTED CROSS-VALIDATION (esto puede tardar unos minutos)")
print("=" * 78)

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
outer_cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=5, random_state=SEED
)

svm_nested = GridSearchCV(
    estimator=svm_pipe, param_grid=param_grid_svm,
    cv=inner_cv, scoring='f1', n_jobs=-1, refit=True
)
rf_nested = GridSearchCV(
    estimator=rf_pipe, param_grid=param_grid_rf,
    cv=inner_cv, scoring='f1', n_jobs=-1, refit=True
)

print("\n>>> Evaluando SVM (nested CV)...")
svm_cv = cross_validate(
    svm_nested, X_train, y_train,
    cv=outer_cv, scoring=SCORING,
    n_jobs=1, return_estimator=False
)

print(">>> Evaluando Random Forest (nested CV)...")
rf_cv = cross_validate(
    rf_nested, X_train, y_train,
    cv=outer_cv, scoring=SCORING,
    n_jobs=1, return_estimator=False
)

# --- Resumen de CV ------------------------------------------------------
def summarize_cv(name, cv_res):
    rows = []
    for metric in [m for m in cv_res if m.startswith('test_')]:
        vals = cv_res[metric]
        rows.append({
            'algoritmo': name,
            'metrica'  : metric.replace('test_', ''),
            'media'    : float(np.mean(vals)),
            'std'      : float(np.std(vals, ddof=1)),
            'min'      : float(np.min(vals)),
            'max'      : float(np.max(vals)),
        })
    return pd.DataFrame(rows)

cv_summary = pd.concat([summarize_cv('SVM', svm_cv),
                        summarize_cv('RandomForest', rf_cv)],
                       ignore_index=True)
cv_summary.to_csv(os.path.join(OUTPUT_DIR, '01_resumen_cv.csv'), index=False)

print("\n--- Resumen de validación cruzada (media ± desv.) ---")
print(cv_summary.pivot(index='metrica', columns='algoritmo',
                       values='media').round(4).to_string())
print("\n(ver resultados/01_resumen_cv.csv para desviaciones y extremos)")


# =============================================================================
# [S6] ANÁLISIS ESTADÍSTICO — WILCOXON SIGNED-RANK
# =============================================================================
# Hipótesis:
#   H0: la mediana de las diferencias pareadas (SVM − RF) en F2 = 0
#   H1: la mediana de las diferencias pareadas (SVM − RF) en F2 ≠ 0
# Nivel de significancia: α = 0.05
# Supuestos: no paramétrica; solo requiere pares comparables.
# ----------------------------------------------------------------------
print("\n" + "=" * 78)
print(" [S6] ANÁLISIS ESTADÍSTICO — Wilcoxon signed-rank (F2 clase 1)")
print("=" * 78)

f2_svm = svm_cv['test_f2_class1']
f2_rf  = rf_cv ['test_f2_class1']
diffs  = f2_svm - f2_rf

stat_w, p_w = stats.wilcoxon(f2_svm, f2_rf, alternative='two-sided',
                              zero_method='wilcox')

# Tamaño del efecto: r de Wilcoxon (r = Z / sqrt(N))
n_pairs = len(diffs)
if n_pairs > 0 and p_w > 0:
    z_approx = stats.norm.ppf(p_w / 2) * np.sign(np.median(diffs)) \
               if p_w < 1 else 0.0
    r_effect = abs(z_approx) / np.sqrt(n_pairs)
else:
    r_effect = float('nan')

print(f"\nPares comparables   : {n_pairs}")
print(f"SVM   F2 media±std  : {f2_svm.mean():.4f} ± {f2_svm.std(ddof=1):.4f}")
print(f"RF    F2 media±std  : {f2_rf .mean():.4f} ± {f2_rf .std(ddof=1):.4f}")
print(f"Diferencia media    : {diffs.mean():+.4f}")
print(f"Mediana diferencia  : {np.median(diffs):+.4f}")
print(f"Wilcoxon statistic  : {stat_w:.4f}")
print(f"p-valor (bilateral) : {p_w:.6f}")
print(f"Tamaño del efecto r : {r_effect:.4f}")

alpha = 0.05
if p_w < alpha:
    verdict = (f"Se RECHAZA H0 al {alpha*100:.0f}%: existe evidencia de "
               f"diferencia significativa en F2 entre SVM y RF.")
    mejor = 'SVM' if diffs.mean() > 0 else 'RandomForest'
    verdict += f"\n   El mejor F2 medio lo obtiene {mejor}."
else:
    verdict = (f"NO se rechaza H0 al {alpha*100:.0f}%: no hay evidencia "
               f"estadística de diferencia en F2 entre SVM y RF.")
print("\n" + verdict)
print("\nNota: 'no significativo' NO implica equivalencia. Reportar")
print("      junto con el tamaño del efecto y la variabilidad observada.")

with open(os.path.join(OUTPUT_DIR, '02_prueba_estadistica.txt'), 'w') as fh:
    fh.write("PRUEBA DE WILCOXON SIGNED-RANK — F2 de la clase 1\n")
    fh.write("=" * 60 + "\n")
    fh.write(f"Pares comparables : {n_pairs}\n")
    fh.write(f"SVM F2 media±std  : {f2_svm.mean():.4f} ± {f2_svm.std(ddof=1):.4f}\n")
    fh.write(f"RF  F2 media±std  : {f2_rf .mean():.4f} ± {f2_rf .std(ddof=1):.4f}\n")
    fh.write(f"Diferencia media  : {diffs.mean():+.4f}\n")
    fh.write(f"Mediana diferencia: {np.median(diffs):+.4f}\n")
    fh.write(f"Estadístico W     : {stat_w:.4f}\n")
    fh.write(f"p-valor           : {p_w:.6f}\n")
    fh.write(f"alpha             : {alpha}\n")
    fh.write(f"Tamaño efecto r   : {r_effect:.4f}\n\n")
    fh.write(verdict + "\n")


# =============================================================================
# [S7] EVALUACIÓN FINAL EN TEST + FIGURAS + RESULTADOS
# =============================================================================
print("\n" + "=" * 78)
print(" [S7] EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA (hold-out 20%)")
print("=" * 78)

# Reajuste de los mejores modelos (con hiperparámetros ya elegidos por
# GridSearchCV dentro de los folds) sobre TODO el train.
svm_nested.fit(X_train, y_train)
rf_nested.fit(X_train, y_train)

svm_best = svm_nested.best_estimator_
rf_best  = rf_nested.best_estimator_

print("\nMejores hiperparámetros:")
print(f"  SVM : {svm_nested.best_params_}")
print(f"  RF  : {rf_nested.best_params_}")

def evaluate_on_test(name, model, X_te, y_te):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    return {
        'algoritmo'      : name,
        'f2_class1'      : fbeta_score(y_te, y_pred, beta=2, pos_label=1),
        'f1_class1'      : f1_score(y_te, y_pred, pos_label=1),
        'recall_class1'  : recall_score(y_te, y_pred, pos_label=1),
        'precision_class1': precision_score(y_te, y_pred, pos_label=1),
        'roc_auc'        : roc_auc_score(y_te, y_proba),
        'avg_precision'  : average_precision_score(y_te, y_proba),
        'balanced_acc'   : balanced_accuracy_score(y_te, y_pred),
        'accuracy'       : accuracy_score(y_te, y_pred),
        'y_pred'         : y_pred,
        'y_proba'        : y_proba,
    }

res_svm = evaluate_on_test('SVM',          svm_best, X_test, y_test)
res_rf  = evaluate_on_test('RandomForest', rf_best,  X_test, y_test)

test_table = pd.DataFrame([
    {k: v for k, v in res_svm.items() if k not in ('y_pred', 'y_proba')},
    {k: v for k, v in res_rf .items() if k not in ('y_pred', 'y_proba')},
])
test_table.to_csv(os.path.join(OUTPUT_DIR, '03_resultados_test.csv'), index=False)
print("\n--- Métricas en test (hold-out) ---")
print(test_table.round(4).to_string(index=False))

# --- Reportes de clasificación ------------------------------------------
for name, res in [('SVM', res_svm), ('RandomForest', res_rf)]:
    print(f"\nClassification report — {name}")
    print(classification_report(y_test, res['y_pred'],
                                target_names=['Genuino(0)', 'Falsificado(1)'],
                                digits=4))

# --- Figura 1: Boxplot de CV (F2) ---------------------------------------
plt.figure(figsize=(8, 5))
cv_df = pd.DataFrame({'SVM': f2_svm, 'RandomForest': f2_rf})
sns.boxplot(data=cv_df, palette='Set2')
sns.stripplot(data=cv_df, color='black', size=4, alpha=0.6)
plt.ylabel('F2-score (clase 1)')
plt.title('Distribución de F2 en 25 folds (Nested CV)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '03_boxplot_cv_f2.png'), dpi=150)
plt.close()

# --- Figura 2: Matrices de confusión ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, res) in zip(axes, [('SVM', res_svm),
                                   ('RandomForest', res_rf)]):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Genuino', 'Falsificado'],
                yticklabels=['Genuino', 'Falsificado'])
    ax.set_title(f'{name}\nRecall₁={res["recall_class1"]:.3f} | '
                 f'F2={res["f2_class1"]:.3f}')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
plt.suptitle('Matrices de confusión — Test set')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '04_confusion_matrices.png'), dpi=150)
plt.close()

# --- Figura 3: Curvas ROC y PR ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, res in [('SVM', res_svm), ('RandomForest', res_rf)]:
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.4f})")
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('Curvas ROC — Test set')
axes[0].legend(); axes[0].grid(alpha=0.3)

for name, res in [('SVM', res_svm), ('RandomForest', res_rf)]:
    prec, rec, _ = precision_recall_curve(y_test, res['y_proba'])
    axes[1].plot(rec, prec, label=f"{name} (AP={res['avg_precision']:.4f})")
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Curvas Precision-Recall — Test set')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '05_roc_pr_curves.png'), dpi=150)
plt.close()

# --- Figura 4: Importancia de variables del Random Forest ----------------
rf_clf = rf_best.named_steps['clf']
imp_df = pd.DataFrame({
    'feature'   : COLUMNS[:-1],
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(8, 4.5))
plt.barh(imp_df['feature'], imp_df['importance'], color='steelblue')
plt.xlabel('Importancia (impureza media)')
plt.title('Random Forest — Importancia de variables')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '06_rf_importance.png'), dpi=150)
plt.close()

# --- Guardar predicciones completas -------------------------------------
pd.DataFrame({
    'y_true'     : y_test,
    'svm_pred'   : res_svm['y_pred'],
    'svm_proba'  : res_svm['y_proba'],
    'rf_pred'    : res_rf['y_pred'],
    'rf_proba'   : res_rf['y_proba'],
}).to_csv(os.path.join(OUTPUT_DIR, '04_predicciones_test.csv'), index=False)


# =============================================================================
# RESUMEN FINAL / DISCUSIÓN (para el artículo)
# =============================================================================
print("\n" + "=" * 78)
print(" RESUMEN FINAL PARA EL ARTÍCULO")
print("=" * 78)

print(f"\n[S5] Nested CV (25 folds, train):")
print(f"  SVM F2 media±std : {f2_svm.mean():.4f} ± {f2_svm.std(ddof=1):.4f}")
print(f"  RF  F2 media±std : {f2_rf .mean():.4f} ± {f2_rf .std(ddof=1):.4f}")
print(f"\n[S6] Wilcoxon: W={stat_w:.2f}, p={p_w:.4f} "
      f"({'significativo' if p_w < alpha else 'no significativo'} a α=0.05)")

print(f"\n[S7] Test hold-out (20%):")
for name, res in [('SVM', res_svm), ('RandomForest', res_rf)]:
    print(f"  {name:12s} | F2={res['f2_class1']:.4f} | "
          f"Recall₁={res['recall_class1']:.4f} | "
          f"Prec₁={res['precision_class1']:.4f} | "
          f"ROC-AUC={res['roc_auc']:.4f}")

print("\n[Discusión sugerida]")
print("  1) ¿Qué algoritmo obtiene mayor F2/recall en la clase falsificada?")
print("  2) ¿La diferencia es estadísticamente significativa (S6)?")
print("  3) ¿Es relevante para el contexto? (¿cuántos FN evita en test?)")
print("  4) Analizar trade-off Recall vs Precision: si priorizamos FN,")
print("     considerar ajustar threshold / class_weight.")
print("  5) Limitaciones: N pequeño (~1372), solo 2 algoritmos, sin tuning")
print("     de umbral específico, hold-out único.")
print("  6) Extensiones: KNN, MLP, Perceptrón multicapa, PSO para tuning,")
print("     ajuste de threshold guiado por costos.")

print("\n" + "=" * 78)
print(f" Resultados guardados en: ./{OUTPUT_DIR}/")
print("=" * 78)