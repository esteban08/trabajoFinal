# 🧠 Predictor de Riesgo en Salud Mental (Metodología CRISP-ML)

Este proyecto implementa un sistema inteligente para clasificar y predecir el nivel de riesgo de salud mental (Bajo, Medio, Alto) en la población española utilizando datos demográficos y hábitos cotidianos (sueño, estrés, apoyo social, etc.), siguiendo rigurosamente el marco metodológico de **CRISP-ML(Q)**.

---

## 📁 Estructura del Proyecto

```
mental_health_risk_predictor/
├── 01_ETL.ipynb                  # 🧹 ETL: Carga, imputación y validación de calidad.
├── 02_EDA.ipynb                  # 📊 EDA: Análisis exploratorio estadístico y visualizaciones.
├── 03_Modeling.ipynb             # 🤖 Modelado: Regresión Logística, Random Forest y Gradient Boosting.
├── app.py                        # 🎈 Aplicación interactiva de Streamlit (Frontend/Backend).
├── index.html                    # 🖥️ Landing Page de presentación del proyecto.
├── style.css                     # 🎨 Diseño y estilos modernos para la landing page.
├── requirements.txt              # 📦 Dependencias necesarias del proyecto.
├── mental_health_spain_final.csv # 📊 Dataset de salud mental en España.
├── cleaned_data.csv              # 🧹 Dataset preprocesado tras el paso de ETL.
├── modelo_salud_mental.pkl       # 💾 Pipeline completo del modelo predictivo guardado.
├── features.pkl                  # 💾 Lista de variables predictoras originales.
└── images/                       # 📸 Gráficas y plots generados en el EDA y modelado.
```

---

## ⚙️ Ciclo de Vida CRISP-ML(Q) Implementado

1. **Comprensión del Negocio (Business Understanding):** Definición de la métrica objetivo (`RISK_LEVEL`) y factores mitigantes o agravantes de riesgo.
2. **Comprensión de los Datos (Data Understanding):** Recopilación del dataset con variables como edad, género, comunidad autónoma, horas de sueño, estrés percibido, etc.
3. **Preparación de los Datos (Data Preparation - ETL):** Tratamiento preventivo de nulos, estructuración y tipado adecuado.
4. **Modelado (Modeling):** Modelos entrenados utilizando pipelines de scikit-learn con validación cruzada y optimización de hiperparámetros (Grid Search).
5. **Evaluación (Evaluation):** Comprobación de métricas de precisión, exhaustividad (recall) y f1-score en test estratificado.
6. **Despliegue y Monitoreo (Deployment & Quality):** Publicación del modelo interactivo en una interfaz amigable (Streamlit) y Landing Page explicativa.

---

## 🚀 Cómo Ejecutar el Proyecto

### 1. Clonar/Ubicar el directorio e instalar dependencias
Asegúrate de tener Python 3.10 o superior instalado. Ejecuta desde la consola en la carpeta del proyecto:
```bash
pip install -r requirements.txt
```

### 2. Levantar la aplicación interactiva Streamlit
Ejecuta el siguiente comando para abrir el predictor interactivo en tu navegador:
```bash
streamlit run app.py
```
Por defecto, se abrirá en `http://localhost:8501`.

### 3. Abrir la Landing Page
Haz doble clic sobre el archivo `index.html` para cargarlo en cualquier navegador web o arrástralo a tu ventana para ver el diseño premium responsive que describe el marco de calidad CRISP-ML(Q).
