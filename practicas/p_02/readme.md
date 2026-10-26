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

--

