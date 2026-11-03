# Autenticación de billetes: regresión logística frente a Random Forest

Proyecto de aprendizaje supervisado en el que se comparan dos algoritmos para distinguir billetes genuinos de falsificados con el conjunto Banknote Authentication de la UCI (1,372 observaciones y 4 características numéricas obtenidas con wavelets). La clase positiva es `1` (billete falso).

Pregunta de investigación: ¿cuál de los dos modelos, la regresión logística (modelo lineal base) o el Random Forest (no lineal), identifica mejor los billetes falsos con un protocolo estratificado, reproducible y sin fuga de información?

## Diseño del experimento

- Métrica principal: recall de la clase falsa, porque aceptar una falsificación es el error más costoso. Complementarias: precisión, F1, exactitud, especificidad y AUC, además de los conteos TN, FP, FN y TP.
- Partición 80/20 estratificada. El conjunto de prueba se usa una sola vez al final.
- Validación cruzada estratificada de 5 pliegues sobre el entrenamiento, con los mismos pliegues para los dos modelos. El Random Forest ajusta sus hiperparámetros con una validación interna de 3 pliegues dentro de cada pliegue externo.
- La regresión logística lleva un `StandardScaler` dentro del `Pipeline`, así que el escalado se ajusta solo con los datos de entrenamiento de cada pliegue.
- Prueba estadística: t pareada sobre el recall por pliegue, con la corrección de Nadeau y Bengio como prueba principal (la t pareada convencional se muestra solo como referencia), con α = 0.05.
- Los 24 registros duplicados se conservan y su efecto se mide aparte.
- Semilla: `random_state=42` en la partición, los pliegues y los modelos.

## Estructura de carpetas

```
data/data_banknote_authentication.txt   dataset original
notebooks/main.ipynb                    código completo
report/plots/                           figuras 01 a 13
results/random_forest/                  resultados del Random Forest
results/comparacion/                    resultados de la comparación
requirements.txt                        versiones de las librerías
```

## Contenido del notebook

1. Datos y regresión logística: análisis exploratorio, preparación sin fuga de información, validación cruzada y modelo final.
2. Random Forest: validación cruzada anidada, importancia de variables y reglas de decisión de un árbol pequeño.
3. Comparación: matrices de confusión, curvas ROC, prueba de hipótesis, billetes mal clasificados, efecto de los duplicados y conclusiones.

## Cómo ejecutarlo

1. Instalar las librerías: `pip install -r requirements.txt` (Python 3.13).
2. Abrir `notebooks/main.ipynb` desde la carpeta `notebooks/` y ejecutar todas las celdas en orden. Las rutas son relativas a esa carpeta.
3. Tarda entre 2 y 3 minutos con 2 núcleos; casi todo el tiempo es la búsqueda de hiperparámetros de la sección 2.

Las versiones usadas en la ejecución quedan guardadas en `results/comparacion/08_versiones_software.txt`. Con esas versiones, una nueva ejecución da las mismas tablas.

## Archivos de resultados

`results/random_forest/`

| Archivo | Contenido |
|---|---|
| `01_rf_resumen_cv.csv` | media y desviación de cada métrica |
| `02_rf_mediciones_por_pliegue.csv` | métricas e hiperparámetros por pliegue |
| `03_rf_predicciones_fuera_de_muestra.csv` | predicción y probabilidad de cada billete de entrenamiento cuando estuvo en validación |
| `04_rf_importancia_variables.csv` | importancia de las variables (modelo final y por pliegue) |
| `05_rf_hiperparametros_finales.json` | hiperparámetros del modelo final |
| `06_modelo_rf_final.joblib` | modelo final ajustado con todo `X_train` |
| `07_reglas_arbol_decision.txt` | reglas del árbol de profundidad 3 (solo para interpretar) |

`results/comparacion/`

| Archivo | Contenido |
|---|---|
| `01_mediciones_por_pliegue.csv` | métricas por pliegue de los dos modelos, con especificidad, AUC y TN/FP/FN/TP |
| `02_predicciones_fuera_de_muestra.csv` | predicción y probabilidad fuera de muestra de los dos modelos |
| `03_metricas_cv_y_prueba.csv` | métricas agrupadas en validación cruzada y en prueba |
| `04_prueba_estadistica.txt` | hipótesis, supuestos, estadísticos, p-valores e intervalos |
| `05_predicciones_prueba.csv` | predicciones de los dos modelos en `X_test` |
| `06_billetes_mal_clasificados.csv` | cada billete mal clasificado, con tipo de error y características |
| `07_sensibilidad_duplicados.csv` | métricas con y sin las filas que tienen copia exacta en el entrenamiento |
| `08_versiones_software.txt` | versiones de las librerías y semilla |
| `09_resumen_por_pliegue.csv` | media ± desviación estándar por pliegue de cada métrica, para los dos modelos |

Las figuras `01` a `04` son del análisis exploratorio, `05` a `07` de la regresión logística, `08` a `10` del Random Forest, `11` son las matrices de confusión, `12` las curvas ROC y `13` el recall por pliegue.

## Resultados principales

Validación cruzada, media ± desviación estándar de los 5 pliegues:

| Métrica | Regresión logística | Random Forest |
|---|---|---|
| Recall (falsos) | 0.9959 ± 0.0056 | 0.9959 ± 0.0056 |
| Precisión | 0.9703 ± 0.0153 | 0.9801 ± 0.0209 |
| F1 | 0.9829 ± 0.0084 | 0.9879 ± 0.0131 |
| Especificidad | 0.9754 ± 0.0130 | 0.9835 ± 0.0175 |
| AUC | 0.9997 ± 0.0002 | 0.9998 ± 0.0002 |

- La prueba sobre el recall (t pareada con corrección de Nadeau y Bengio) da t(4) = 0.0000 y p = 1.0000, así que no se rechaza H0. Esto no demuestra que los modelos sean equivalentes, solo que estos datos no muestran una diferencia.
- En el conjunto de prueba ningún modelo acepta una falsificación (FN = 0 de 122).
- Las diferencias en falsos positivos no son concluyentes: la desventaja aparente de la regresión logística se concentra en filas con un registro idéntico en el entrenamiento, y sin ellas la diferencia desaparece o se invierte.

## Referencia del dataset

V. Lohweg, "Banknote Authentication", UCI Machine Learning Repository, 2012, doi: 10.24432/C55P57.
