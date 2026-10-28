# Entender el problema

> Más info del dataset: https://archive.ics.uci.edu/dataset/267/banknote+authentication 

### **1. Comprensión del Problema y del Dataset**

* **Dataset:** *Banknote Authentication* (1,372 observaciones, 4 características continuas y variable objetivo binaria).
* **Características:**
  1. **Varianza** de la imagen transformada en wavelets.
  2. **Asimetría** (*skewness*).
  3. **Curtosis** (*kurtosis*).
  4. **Entropía** de la imagen.
* **Variable Objetivo:** $y \in \{0, 1\}$ (donde $0$ suele representar billete genuino y $1$ billete falsificado).
* **Análisis Inicial Requerido:**
  * Comprobar la distribución de clases (el dataset tiene aproximadamente un 55% genuinos y 45% falsificados, por lo que está razonablemente balanceado).
  * Evaluar estadísticas descriptivas (medias, desviaciones estándar, histogramas de distribución por característica).

---

### **2. Selección y Justificación de Métricas (Análisis del Negocio)**

El documento enfatiza que **los errores tienen costos y consecuencias asimétricas** en la autenticación:

1. **Falso Positivo (Aceptar un billete falso como genuino):** Genera una **pérdida económica directa** para la entidad financiera o negocio.
2. **Falso Negativo (Rechazar un billete genuino como falso):** Causa molestias al cliente, revisiones manuales adicionales y afecta la eficiencia operativa.

#### **Propuesta de Métricas:**
* **Métrica Principal:** **Recall / Sensibilidad** sobre la clase de billetes falsos (o **F1-Score / Matriz de Confusión**), justificando el impacto económico de minimizar los falsos positivos/negativos.
* **Métricas Complementarias:** 
  * **Especificidad** (para medir la tasa de billetes genuinos bien identificados).
  * **Curva ROC y AUC:** Imprescindible en este proyecto para analizar cómo varía el desempeño al ajustar el umbral de decisión ($\tau$), permitiendo equilibrar la sensibilidad frente a las falsas alarmas según la política de riesgo del banco.

---

### **3. Selección de dos Algoritmos Comparativos**

Debes seleccionar **dos clasificadores de familias distintas** para realizar un contraste metodológico justo:

* **Opción Sugerida A:** **Regresión Logística / SVM Lineal** (Modelos discriminativos lineales basados en márgenes/probabilidades).
* **Opción Sugerida B:** **Random Forest / Árbol de Decisión** (Modelos no lineales basados en particiones del espacio de características).
* **En el marco teórico:** Explica los fundamentos matemáticos de ambos clasificadores, sus ecuaciones de decisión y por qué son adecuados para evaluar este espacio continuo de 4 dimensiones.

---

### **4. Preparación de los Datos y Control de Fuga de Información (*Data Leakage*)**

* **Transformación Numérica:**
  * Para **Regresión Logística** o **SVM**, es **obligatorio estandarizar** las 4 características continuas (`StandardScaler`) debido a que la escala de la varianza o curtosis difiere significativamente.
  * Para **Árboles/Random Forest**, el escalado es opcional.
* **Regla estricta contra la fuga de datos (*Data Leakage*):**
  * El ajuste del escalador (`scaler.fit()`) debe realizarse **única y exclusivamente con el conjunto de entrenamiento de cada pliegue**, y luego aplicarse (`transform()`) al conjunto de prueba.

---

### **5. Diseño Experimental**

Para garantizar una evaluación rigurosa e imparcial:

1. **Esquema de Validación:** Aplica **Validación Cruzada Estratificada de 5 Pliegues (Stratified 5-Fold Cross-Validation)** o **Validación Cruzada 5x2**. Esto asegura que la proporción de billetes falsos/genuinos se mantenga constante en cada subconjunto.
2. **Ajuste de Hiperparámetros (si aplica):** Si ajustas hiperparámetros (ej. profundidad en el árbol o parámetro $C$ en SVM), debes usar una **validación interna (anidada)** sobre el conjunto de entrenamiento de cada pliegue, **sin tocar el conjunto de prueba**.
3. **Reproducibilidad:** Fija las semillas aleatorias (`random_state=42`) en todas las particiones y modelos.

---

### **6. Análisis Estadístico de los Resultados**

No basta con decir cuál obtuvo mayor porcentaje; debes demostrar si la diferencia es **estadísticamente significativa**:

1. **Planteamiento de Hipótesis:**
   * **Hipótesis Nula ($H_0$):** No existe diferencia significativa en el desempeño (métricas) entre el Algoritmo A y el Algoritmo B ($E_A = E_B$).
   * **Hipótesis Alternativa ($H_1$):** Existe una diferencia estadísticamente significativa entre ambos algoritmos ($E_A \neq E_B$).
2. **Prueba Estadística:** Aplica una prueba de hipótesis emparejada adecuada para validación cruzada (por ejemplo, **Paired Student's t-test** sobre los resultados de los pliegues, o **5x2cv paired t-test**) con un nivel de significancia $\alpha = 0.05$.
3. **Interpretación:** Reporta el valor del estadístico $t$ y el $p$-valor. Discute si la diferencia encontrada tiene significancia práctica en la operación de autenticación.

---

### **7. Estructura de Entregables Requeridos**

El proyecto exige cuatro componentes de entrega:

1. **Artículo en formato Springer LNCS (PDF y editable Word/LaTeX):**
   * **Título & Abstract:** Resumen ejecutivo de problema, métodos, resultados principales y conclusión.
   * **Introducción:** Contexto de la autenticación de billetes y pregunta de investigación.
   * **Marco Teórico:** Fundamento de los dos algoritmos elegidos, métricas y validación cruzada.
   * **Metodología & Preparación:** Descripción del dataset, transformaciones y control de fuga de datos.
   * **Experimentación:** Configuración de los pliegues, semillas y procedimiento experimental.
   * **Resultados & Análisis Estadístico:** Tablas comparativas, matrices de confusión, curvas ROC y resultado de la prueba de hipótesis ($p$-valor).
   * **Discusión y Conclusiones:** Análisis de los tipos de error (FP vs. FN) y respuesta clara a la pregunta de investigación.
   * **Referencias:** Citación del dataset y literatura académica utilizada.
2. **Presentación en Diapositivas (PDF y editable):** Apoyo visual con gráficas y tablas legibles que defiendan las decisiones metodológicas.
3. **Código Reproducible:** Script de Python o Jupyter Notebook organizado con comentarios, libre de fuga de datos y ejecutable de principio a fin.
4. **Archivo de Resultados Experimentales:** Archivo estructurado (ej. `.csv`) con las mediciones por pliegue obtenidas en la experimentación.

-----

>  _

# Dividir el proyecto
>  _

La clave está en **no dividir por tipo de entregables** (donde uno solo programa, otro solo hace diapositivas y otro solo escribe). En su lugar, el proyecto se estructura por **ejes metodológicos y experimentales en parejas**, combinando sesiones colectivas de integración.

De esta manera, todos programan, todos analizan resultados y todos participan en la redacción y presentación del trabajo.

---

### 1. Esquema de Organización: Trabajar en 3 Parejas Específicas

Se dividen las tareas del proyecto en 3 ejes técnicos. Cada pareja es responsable del código y de la sección correspondiente del artículo LNCS para su eje:

```
                      PROYECTO COMPLETO (6 Integrantes)
                                     │
    ┌────────────────────────────────┼────────────────────────────────┐
    ▼                                ▼                                ▼
PAREJA A (2 pers.)               PAREJA B (2 pers.)               PAREJA C (2 pers.)
Eje: Datos y Modelo 1           Eje: Modelo 2                   Eje: Evaluación y Estadística
- EDA y Preprocesamiento        - Modelo Complejo/No Lineal     - Métricas y Curvas ROC
- Modelo Lineal/Base            - Ajuste de Hiperparámetros     - Prueba t de Student y Errores
- Redacción: Intro y Datos      - Redacción: Marco Modelo 2     - Redacción: Resultados y Discusión
```

---

### 2. Matriz de Responsabilidades Cruzadas

#### **Pareja A: Eje de Datos y Modelo Base (ej. Regresión Logística / SVM Lineal)**
* **Código:**
  * Análisis exploratorio de datos (EDA): histogramas de las 4 características de los billetes, matriz de correlación.
  * Creación de la tubería (*pipeline*) de preparación: separación de pliegues (5-Fold Stratified CV) y escalado (`StandardScaler`) asegurando no tener fuga de información (*data leakage*).
  * Entrenamiento y evaluación del Modelo 1 (Regresión Logística / SVM Lineal).
* **Redacción del Artículo LNCS:**
  * Sección **Introducción** (planteamiento del problema y contexto de la autenticación de billetes).
  * Sección **Metodología: Descripción de los datos y preprocesamiento**.
  * Sección **Marco Teórico: Modelo 1**.

#### **Pareja B: Eje de Modelo Complejo (ej. Random Forest / Árbol de Decisión)**
* **Código:**
  * Implementación y entrenamiento del Modelo 2 (Random Forest / Árbol de Decisión) bajo las mismas particiones de validación cruzada.
  * Búsqueda y ajuste de hiperparámetros (profundidad, número de estimadores) mediante validación interna sobre los datos de entrenamiento.
  * Extracción de importancia de variables o reglas de decisión del árbol.
* **Redacción del Artículo LNCS:**
  * Sección **Marco Teórico: Modelo 2**.
  * Sección **Metodología: Configuración experimental e hiperparámetros**.

#### **Pareja C: Eje de Métricas, Pruebas Estadísticas y Análisis de Errores**
* **Código:**
  * Script para generación de curvas ROC, cálculo de AUC y matrices de confusión comparativas.
  * Implementación de la prueba de hipótesis estadística emparejada (ej. *Paired Student's t-test* o *5x2cv*) para calcular el \\(p\\)-valor de diferencia entre ambos modelos.
  * Identificación de casos misclasificados (falsos positivos vs. falsos negativos en billetes).
* **Redacción del Artículo LNCS:**
  * Sección **Marco Teórico: Métricas y Validación Cruzada**.
  * Sección **Resultados y Análisis Estadístico** (tablas comparativas, reporte del \\(p\\)-valor).
  * Sección **Discusión de Errores y Conclusiones**.

---

### 3. Dinámica para Evitar el "Efecto Frankenstein"

Para garantizar coherencia narrativa, fluidez en el código y que todos dominen el tema completo, se ejecutan **4 Hitos de Integración Colectiva**:

1. **Hito 1: Pregunta de Investigación y Protocolo (Reunión inicial de 1 hr):**
   * Los 6 definen juntos la pregunta de investigación, la regla de sembrado aleatorio (`random_state=42`) y la métrica principal basada en el costo del error (ej. minimizar falsos positivos/negativos).
2. **Hito 2: Integración del Código en un Único Notebook / Script:**
   * La Pareja C junta los bloques de código creados por A y B en una sola tubería continua y ejecutable sin errores.
3. **Hito 3: Edición Cruzada del Artículo (Peer Review Interno):**
   * **Revisión en bucle:** La Pareja A lee y edita la sección escrita por B; B revisa la de C; y C revisa la de A.
   * Se asigna un **Editor de Estilo Rotativo** (un integrante de cada pareja) para uniformar el tono del texto, revisar las citas bibliográficas y formatear las tablas bajo la plantilla Springer LNCS.
4. **Hito 4: Ensayos Colectivos de la Presentación:**
   * Las diapositivas se reparten por temas, no por entregables.
   * Se realizan ensayos de preguntas y respuestas cruzadas: si al equipo le preguntan sobre el código del Modelo 2 o la prueba \\(t\\), cualquier integrante de la Pareja A o C debe estar en capacidad de responder.


---
