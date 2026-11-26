
# Proyecto 1 · Variante A — PCA para apoyar KNN (Digits)


Este archivo consolida: (1) la estructura de reparto en 3 parejas, (2) los fundamentos científicos y supuestos extraídos del PDF oficial (`Proyectos_aprendizaje_no_supervisado_2026S2.pdf`), (3) los estándares de código, y (4) las **precisiones críticas** que el PDF exige y que debemos cumplir al pie de la letra para no perder puntos.

> **Regla de oro:** toda decisión debe justificarse con **medias y desviaciones estándar**. El análisis es **exclusivamente descriptivo**. No se aplican pruebas estadísticas, no se reportan valores p y no se afirma significancia, equivalencia ni superioridad general.

---

## Índice

1. [Resumen del proyecto](#1-resumen-del-proyecto)
2. [Fundamentos científicos y supuestos](#2-fundamentos-científicos-y-supuestos)
3. [Estándares de código y configuración](#3-estándares-de-código-y-configuración)
4. [Estructura de trabajo: 3 parejas](#4-estructura-de-trabajo-3-parejas)
5. [Protocolo experimental detallado](#5-protocolo-experimental-detallado)
6. [Hitos de integración](#6-hitos-de-integración)
7. [Cronograma sugerido](#7-cronograma-sugerido)
8. [Checklist de entregables](#8-checklist-de-entregables)
9. [Anexo A — Precisiones críticas del PDF](#anexo-a--precisiones-críticas-del-pdf)
10. [Anexo B — Advertencia obligatoria](#anexo-b--advertencia-obligatoria)
11. [Referencias](#referencias)

---

## 1. Resumen del proyecto

| Elemento | Detalle |
|---|---|
| **Proyecto** | 1 · Clasificación de dígitos (Digits) |
| **Variante** | A · PCA antes de KNN |
| **Dataset** | `load_digits()` — 1,797 imágenes 8×8, 64 características, 10 clases (0–9) |
| **Pregunta de investigación** | ¿Una representación compacta (PCA) conserva o mejora la clasificación y reduce el costo de predicción frente a KNN con las 64 características? |
| **Base** | KNN con 64 características, k ∈ {1,3,5,7} |
| **Propuesta** | PCA (80 %, 90 %, 95 % varianza, `whiten=False`) + mismo KNN |
| **Métrica principal** | F1 macro (media ± DE en validación) |
| **Métricas secundarias** | Exactitud, dimensiones retenidas por fold, tiempo de predicción |
| **Configuraciones** | 4 base + 12 propuestas = 16 |
| **Validación** | `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` → 15 evaluaciones por configuración |
| **Semilla** | `random_state = 42` en todas las operaciones estocásticas |
| **Evaluación final** | Reentrenar con 100 % desarrollo, evaluar 20 % prueba **una sola vez** |

---

## 2. Fundamentos científicos y supuestos

### 2.1 K-Nearest Neighbors (KNN)

- **Métrica de distancia:** euclidiana (`p=2`, Minkowski). KNN depende de la geometría en ℝ^D y es sensible a la escala relativa.
- **Normalización en Digits:** como los píxeles tienen rango conocido [0, 16], se divide directamente `X / 16.0`. No se ajusta escalador (el rango es conocido y fijo).
- **Maldición de la dimensionalidad:** en 64 dimensiones, el contraste entre distancias se pierde por píxeles de fondo ruidosos. La hipótesis es que PCA reduce D y acelera la inferencia manteniendo o mejorando F1 macro.
- **Costo de inferencia:** KNN es *lazy learning*; su predicción es O(N_train · D). PCA reduce D → menos costo.

### 2.2 Análisis de Componentes Principales (PCA)

- **Transformación lineal no supervisada:** proyecta sobre ejes ortogonales (eigenvectores de la covarianza) que maximizan varianza explicada. **No usa las etiquetas.**
- **Varianza conservada:** se exploran 80 %, 90 %, 95 % acumulada.
- **`whiten=False` obligatorio:** preserva la magnitud relativa de los componentes y respeta la métrica de distancia de KNN.
- **Ajuste estricto por fold:** PCA se ajusta (`.fit()`) **solo con el train del fold**; validación y test solo se transforman (`.transform()`).

### 2.3 Protocolo de evaluación y prevención de *data leakage*

- **5 folds × 3 repeticiones = 15 evaluaciones** por configuración.
- **Estratificación:** preserva la proporción de los 10 dígitos en cada fold.
- **Sin fuga:** el escalador (si hubiera), PCA y KNN se ajustan dentro de cada fold. Nunca se ajusta nada sobre validación ni test.
- **Alcance descriptivo:** medias y DE. Sin pruebas t, sin p-valores.

---

## 3. Estándares de código y configuración

### 3.1 Reproducibilidad total

- `random_state = 42` en `train_test_split`, `RepeatedStratifiedKFold` y cualquier operación estocástica.
- **Guardar índices y semillas explícitamente** en archivo (`indices.json` o `indices.npz`) con:
  - Índices del split 80/20.
  - Índices de los 15 folds (5 × 3).
  - Semillas usadas en cada repetición.
- **No consultar prueba para tomar decisiones** (ni para elegir configuraciones ni para ajustar el pipeline).

### 3.2 Registro de versiones y parámetros no explorados

En la primera celda del notebook, registrar:

```python
import sklearn, numpy, pandas, matplotlib
print("scikit-learn:", sklearn.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)
```

Además, documentar **parámetros no explorados** (ej. `weights='distance'`, otras métricas de distancia, `algorithm='kd_tree'`, etc.) y las **reglas de desempate** aplicadas.

### 3.3 Modo oscuro en visualizaciones (opcional pero acordado por el equipo)

```python
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams["axes.facecolor"] = "#1e1e1e"
plt.rcParams["figure.facecolor"] = "#121212"
plt.rcParams["grid.color"] = "#333333"
```

> ⚠️ **Nota:** el PDF no exige estilo oscuro. Si se usa, verificar legibilidad en proyector. La claridad tiene prioridad sobre la estética.

### 3.4 Encapsulamiento mediante Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    ("pca", PCA(whiten=False)),
    ("knn", KNeighborsClassifier(metric="minkowski", p=2,
                                  weights="uniform", algorithm="brute")),
])
```

- El `Pipeline` garantiza que PCA se ajuste solo en el train de cada fold.
- **`algorithm='brute'` obligatorio** para comparaciones de tiempo justas y uniformes.

### 3.5 Medición rigurosa de tiempos

Protocolo (ver Anexo A, punto 2):

```python
import time
import numpy as np

def medir_tiempo_prediccion(modelo, X, n_mediciones=5):
    # 1 calentamiento (descartado)
    _ = modelo.predict(X)
    # 5 mediciones
    tiempos = []
    for _ in range(n_mediciones):
        t0 = time.perf_counter()
        _ = modelo.predict(X)
        t1 = time.perf_counter()
        tiempos.append((t1 - t0) * 1000)  # ms
    # mediana como tiempo del fold
    return float(np.median(tiempos))
```

- 1 calentamiento (descartado).
- 5 mediciones por fold.
- **Mediana** de las 5 = tiempo del fold.
- Resumir los 15 tiempos con media ± DE.
- Las 5 repeticiones de cronometraje **no** son repeticiones de validación.

---

## 4. Estructura de trabajo: 3 parejas

```
                    PROYECTO 1A: PCA + KNN EN DIGITS (6 personas)
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    ▼                                  ▼                                  ▼
PAREJA 1 (2 pers.)                 PAREJA 2 (2 pers.)                 PAREJA 3 (2 pers.)
Eje: Datos, CV y Modelo Base       Eje: Pipeline PCA + KNN            Eje: Evaluación Test, Errores y Tiempos
- Preprocesamiento (X/16)          - Grid 12 variantes PCA+KNN        - Reentrenamiento en 80%
- Split 80/20 & 5x3 CV (seed 42)   - Regla de selección (Δ ≤ 0.005)   - Evaluación única en 20% Test
- Modelo Base KNN (k=1,3,5,7)      - Scree plot (Dark Mode)           - Matrices de confusión y tiempos
```

> **Principio rector:** no dividir por tipo de entregable (unos programan, otros hacen diapositivas). Cada pareja **programa, analiza y expone**. El notebook final es único y fluido.

### 4.1 Pareja 1 — Datos, protocolo CV y modelo base

**Código**
- Cargar `load_digits()` y aplicar `X / 16.0`.
- Split estratificado: 80 % desarrollo (1,437 muestras) y 20 % prueba (360 muestras) con `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
- **Guardar índices** del split en archivo (`indices.json`).
- Construir `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`.
- Evaluar **Modelo Base** (KNN con 64 características) para k ∈ {1,3,5,7}, distancia euclidiana, voto uniforme, `algorithm='brute'`.
- Calcular media ± DE de F1 macro y exactitud en train y validación sobre las 15 iteraciones.
- **Calcular diferencias pareadas** (propuesta − base) fold a fold (ver Anexo A, punto 3).
- Registrar **versiones** de librerías.

**Presentación**
- Introducción y planteamiento del problema (alta dimensionalidad en 8×8).
- Motivación de la normalización `X/16`.
- Diagrama del flujo de validación cruzada 5×3 y prevención de *data leakage*.
- Resultados del Modelo Base (KNN sin PCA).

### 4.2 Pareja 2 — Pipeline PCA + KNN y regla de selección

**Código**
- Crear `Pipeline([('pca', PCA(whiten=False)), ('knn', KNeighborsClassifier(...algorithm='brute'))])`.
- Explorar las **12 propuestas**: V ∈ {0.80, 0.90, 0.95} × k ∈ {1,3,5,7}.
- Registrar por cada fold el **número real de componentes retenidos** (`pca.n_components_`).
- Programar la **regla de selección**:
  1. Mayor F1 macro medio en validación.
  2. Si dentro de la familia Δ ≤ 0.005 → menor dimensión (`D_PCA`).
  3. Luego menor DE.
  4. Finalmente menor k.
  - **Nota:** para la familia **base**, todas las configuraciones tienen D=64, así que el desempate "menor dimensión" no aplica; el orden efectivo es F1 → DE → k.
- **Calcular diferencias pareadas** vs base sobre los mismos folds.

**Presentación**
- Fundamento teórico de PCA (compresión de varianza, reducción de ruido).
- Scree plot (varianza explicada acumulada) en Dark Mode: cuántos componentes para 80 %, 90 %, 95 %.
- Tabla comparativa de las 12 configuraciones con media ± DE y diferencia vs base.

### 4.3 Pareja 3 — Evaluación en test, matrices de confusión y tiempos

**Código**
- Reentrenar la mejor Base y la mejor Propuesta con el **100 % de desarrollo** (1,437 muestras).
- Evaluar ambos sobre el **20 % de prueba** (360 muestras) **una sola vez**.
- Generar **matrices de confusión** en test (`sns.heatmap(cmap='magma')`) para identificar confusiones (ej. 3 vs. 8).
- Medir tiempos de predicción (ver §3.5) en test: 64 características vs. espacio reducido.
- **No agregar DE** a la evaluación única de test.

**Presentación**
- Resultados finales en prueba reservada (sin DE).
- Análisis de matrices de confusión.
- Comparación de tiempos de predicción (ms) bajo condiciones idénticas.
- Conclusiones sobre el compromiso compresión ↔ velocidad ↔ F1 macro.

---

## 5. Protocolo experimental detallado

### 5.1 Partición inicial

1. Separar **una sola vez** 80 % desarrollo / 20 % prueba (estratificado, `random_state=42`).
2. **Guardar índices** del split y de los folds.
3. **No consultar prueba** para ninguna decisión.

### 5.2 Validación cruzada

1. `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)` sobre el 80 %.
2. En cada fold: ajustar PCA **solo con train**, transformar train y validación, ajustar KNN en el espacio reducido.
3. Registrar F1 macro y exactitud en train y validación, dimensiones y tiempos.

### 5.3 Resumen de resultados

Para R = 15 resultados:

- Media = Σm / R
- DE muestral = √[Σ(m − media)² / (R − 1)]
- Reportar «media ± DE» e indicar siempre qué se promedia.
- **No mezclar** escenarios ni métricas en una sola media.

### 5.4 Diferencias pareadas frente a la base

Para cada configuración propuesta, sobre los **mismos folds**:

- F1 macro: `propuesta − base`
- Exactitud: `propuesta − base`
- Reportar media ± DE de esas diferencias.

### 5.5 Selección y evaluación final

1. Elegir por separado la base y la propuesta de mayor F1 macro medio en validación.
2. Aplicar desempate: Δ ≤ 0.005 → menor dimensión → menor DE → menor k.
3. Reentrenar ambas con **todo desarrollo**.
4. Evaluar en **prueba una sola vez**.
5. Reportar métricas de prueba, matrices de confusión, dimensiones y tiempos. **Sin DE** en test.

---

## 6. Hitos de integración

Para evitar el "efecto Frankenstein":

1. **Hito 1 — Semillas y convenciones (Sesión 1):**
   - Fijar `random_state=42`, estructura de carpetas del repo, diccionario global de métricas e **índices guardados**.
2. **Hito 2 — Notebook único (Sesión 2):**
   - La Pareja 3 integra los módulos de Pareja 1 y 2 en un solo Jupyter Notebook reproducible de principio a fin.
3. **Hito 3 — Revisión cruzada de diapositivas (Sesión 3):**
   - Pareja 1 revisa las de Pareja 2; Pareja 2 revisa las de Pareja 3; Pareja 3 revisa las de Pareja 1.
   - Verificar: estética coherente, advertencia obligatoria incluida, sin p-valores, evaluación única en test.
   - Designar un "Editor de Estilo" por pareja.
4. **Hito 4 — Ensayos cruzados (Sesión 4):**
   - Simulación de defensa oral con preguntas asignadas **al azar**: cualquier integrante debe poder responder sobre cualquier sección.

---

## 7. Cronograma sugerido

| Día | Actividad |
|---|---|
| 1–2 | Hito 1. Pareja 1: carga/CV/KNN base. Pareja 2: pipeline PCA. Pareja 3: scripts de matrices y tiempos. |
| 3–4 | Ejecución de las 15 evaluaciones por configuración (4 base + 12 propuestas). Selección de mejores modelos. |
| 5 | Reentrenamiento en 80 % desarrollo y evaluación única en 20 % prueba. Tablas de medias ± DE. |
| 6–7 | Armado de diapositivas con estructura de congreso, revisión cruzada y ensayo oral. |

---

## 8. Checklist de entregables

### 8.1 Código reproducible

- [ ] Notebook o scripts documentados que reproduzcan todo el proceso.
- [ ] Instrucciones de ejecución, dependencias y **versiones**.
- [ ] Semillas e **índices guardados** (split + folds).
- [ ] Tablas completas de todas las configuraciones (media ± DE).
- [ ] **Diferencias pareadas** (propuesta − base) con media ± DE.
- [ ] Visualizaciones solicitadas (scree plot, matrices de confusión).
- [ ] Resultados por fold integrados.

### 8.2 Presentación con estructura de congreso

Secciones en este orden:

1. Título y autoría.
2. Introducción y planteamiento del problema.
3. Objetivos y fundamento (incluir un ejemplo sencillo).
4. Datos y preparación.
5. Metodología (base, propuesta, diagrama, papel de PCA).
6. Diseño experimental (split, CV, semillas, configuraciones, métricas, prevención de fuga).
7. Resultados y selección (tablas, gráficas, evaluación final, matrices).
8. Discusión y limitaciones (incluir caveat de media de validación optimista y **advertencia obligatoria**).
9. Conclusiones y trabajo futuro.
10. Referencias.

### 8.3 Tabla mínima obligatoria por configuración

| Configuración | F1 macro train (media ± DE) | F1 macro val (media ± DE) | Métricas adicionales | Dimensión / tiempo | Diferencia vs base |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

---

## Anexo A — Precisiones críticas del PDF

Estas son las **precisiones que el PDF exige explícitamente** y que la propuesta inicial de la IA no cubría del todo. Deben cumplirse al pie de la letra.

### A.1 Advertencia obligatoria en la presentación

El PDF exige incluir la **advertencia textual** (ver Anexo B). Debe aparecer en la presentación, idealmente en Discusión o al pie de la diapositiva de Resultados.

### A.2 Metodología de tiempos completa

- 1 ejecución de calentamiento (descartada).
- 5 mediciones por fold.
- **Mediana** de las 5 = tiempo del fold.
- Resumir los 15 tiempos (media ± DE).
- Aclarar que las 5 repeticiones de cronometraje **no** son repeticiones de validación.

### A.3 Diferencias pareadas frente a la base

- Calcular **fold a fold** sobre los mismos folds y semilla.
- F1 / exactitud: `propuesta − base`.
- Reportar media ± DE de las diferencias.
- Incluir esta columna en la tabla mínima.

### A.4 Selección de la familia base

- Para la base, todas las configuraciones tienen D=64; el desempate "menor dimensión" **no aplica**.
- Orden efectivo para base: mayor F1 → si Δ ≤ 0.005 → menor DE → menor k.

### A.5 Guardar índices y semillas

- Guardar índices del split 80/20 y de los 15 folds en `indices.json` o `indices.npz`.
- Documentar semillas usadas en cada repetición.

### A.6 Caveat de media de validación optimista

La media de validación de la configuración seleccionada **se usó para elegirla** y puede ser optimista. La prueba reservada aporta una comprobación final. Incluir en la discusión.

### A.7 Registrar versiones y parámetros no explorados

- Versiones de scikit-learn, numpy, pandas, matplotlib.
- Parámetros no explorados (ej. `weights='distance'`, `algorithm='kd_tree'`, otras métricas).
- Reglas de desempate aplicadas.

### A.8 Estilo visual opcional

El PDF **no exige** dark mode. Si se usa, verificar legibilidad en proyector. No sacrificar claridad por estética.

### A.9 Verificar tamaños de muestra

- `n_test = ceil(1797 × 0.2) = 360`
- `n_train = 1,797 − 360 = 1,437`
- Verificar en el código con `print(X_train.shape, X_test.shape)` y reportar conteos por clase.

### A.10 Estructura en parejas = sugerencia organizativa

El PDF no prescribe cómo dividir el trabajo. La propuesta de 3 ejes es razonable, pero la evaluación es del **proyecto integrado**. Los Hitos 2 y 4 son clave para evitar el "Frankenstein".

---

## Anexo B — Advertencia obligatoria

Incluir **textualmente** en la presentación:

> «Este análisis es meramente descriptivo y no cuenta con apoyo inferencial. Las medias y desviaciones estándar resumen las evaluaciones realizadas; no demuestran diferencias estadísticamente significativas, equivalencia ni superioridad general».

---

## Referencias

- Documentación de scikit-learn: `load_digits()`, `PCA`, `KNeighborsClassifier`, `RepeatedStratifiedKFold`, `Pipeline`.
- Guía oficial del curso: `Proyectos_aprendizaje_no_supervisado_2026S2.pdf`.
- Material de apoyo: `slides.pdf`, `Machine.Learning.with.PyTorch.and.Scikit-Learn.pdf`, `Evaluating Learning Algorithms`.
