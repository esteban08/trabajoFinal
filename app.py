import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Configuración de página
st.set_page_config(
    page_title="Predictor de Riesgo en Salud Mental - España",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados para diseño premium
st.markdown("""
    <style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stApp {
        background-color: #0F172A;
    }
    h1, h2, h3 {
        color: #38BDF8 !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .prediction-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-top: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .recommendation-list {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    </style>
""", unsafe_allowed_html=True)

st.title("🧠 Predictor de Riesgo en Salud Mental (CRISP-ML(Q))")
st.write("Esta herramienta interactiva predice el nivel de riesgo en salud mental en la población española utilizando modelos de Machine Learning avanzados.")

# Rutas de artefactos
MODEL_PATH = "modelo_salud_mental.pkl"
FEATURES_PATH = "features.pkl"

@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    return model, features

model, features = cargar_modelo()

if model is None:
    st.error("⚠️ El modelo de Machine Learning no ha sido entrenado o no se encuentra en el directorio actual. Por favor, ejecuta primero los notebooks de modelado.")
else:
    # Crear dos columnas principales
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("📋 Datos del Usuario")
        st.write("Por favor, ingresa los datos demográficos y clínicos para evaluar el riesgo:")
        
        # Edad
        edad = st.slider("Edad", min_value=18, max_value=85, value=35, step=1)
        
        # Género
        genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
        
        # Comunidad Autónoma
        ccaa_list = [
            'Andalucía', 'Cataluña', 'Madrid', 'Comunidad Valenciana', 'Galicia', 
            'País Vasco', 'Castilla y León', 'Canarias', 'Murcia', 'Aragón', 
            'Baleares', 'Extremadura', 'Asturias', 'Navarra', 'Cantabria', 'La Rioja'
        ]
        ccaa = st.selectbox("Comunidad Autónoma", ccaa_list)
        
        # Situación Laboral
        situacion_lab = st.selectbox("Situación Laboral", ["Empleado", "Desempleado", "Estudiante", "Jubilado"])
        
        # Historial Familiar
        historial_fam = st.radio("Historial Familiar de Salud Mental", ["Sí", "No"], horizontal=True)
        
        # Estrés Diario
        estres = st.slider("Nivel de Estrés Diario (1 al 10)", min_value=1, max_value=10, value=5)
        
        # Horas de Sueño
        sueno = st.slider("Horas de Sueño por Noche", min_value=3.0, max_value=10.0, value=7.0, step=0.1)
        
        # Actividad Física
        act_fisica = st.selectbox("Nivel de Actividad Física", ["Baja", "Media", "Alta"])
        
        # Apoyo Social
        apoyo = st.slider("Nivel de Apoyo Social percibido (1 al 10)", min_value=1, max_value=10, value=7)
        
        # Consumo de Redes
        redes = st.slider("Horas diarias de Redes Sociales", min_value=0.5, max_value=12.0, value=3.0, step=0.5)

    with col2:
        st.subheader("📊 Diagnóstico y Predicción")
        
        # Crear DataFrame de entrada con los nombres de columnas exactos
        input_data = pd.DataFrame([{
            'Edad': edad,
            'Genero': genero,
            'Comunidad_Autonoma': ccaa,
            'Situacion_Laboral': situacion_lab,
            'Historial_Familiar': historial_fam,
            'Estres_Diario': estres,
            'Horas_Sueno': sueno,
            'Actividad_Fisica': act_fisica,
            'Apoyo_Social': apoyo,
            'Consumo_Redes': redes
        }])
        
        # Realizar predicción
        pred = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        # Mapeo de colores y texto para el resultado
        color_map = {
            'Bajo': '#10B981',    # Emerald
            'Medio': '#F59E0B',   # Amber
            'Alto': '#EF4444'     # Red
        }
        
        selected_color = color_map.get(pred, '#38BDF8')
        
        # Card de resultado
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Nivel de Riesgo Predicho</h3>
                <div class="metric-value" style="color: {selected_color};">{pred.upper()}</div>
                <p>El modelo clasifica este perfil con un nivel de riesgo de salud mental <strong>{pred.lower()}</strong>.</p>
            </div>
        """, unsafe_allowed_html=True)
        
        # Gráfica de Probabilidades
        st.write("#### 📈 Probabilidades por Categoría")
        prob_df = pd.DataFrame({
            'Categoría': classes,
            'Probabilidad (%)': [p * 100 for p in probs]
        }).sort_values(by='Probabilidad (%)', ascending=False)
        
        st.bar_chart(data=prob_df, x='Categoría', y='Probabilidad (%)', color='#38BDF8')
        
        # Recomendaciones dinámicas
        st.write("#### 🛡️ Recomendaciones personalizadas")
        if pred == 'Alto':
            st.markdown(f"""
                <div class="recommendation-list">
                    <strong>Recomendación Crítica:</strong> Te sugerimos fuertemente contactar con un profesional de la salud mental o médico de cabecera. En España puedes llamar gratis al <strong>024</strong> (Línea de atención a la conducta suicida) o contactar con servicios de salud de tu Comunidad Autónoma ({ccaa}).
                </div>
                <div class="recommendation-list">
                    <strong>Higiene del Sueño:</strong> Intentar aumentar las horas de sueño (actualmente en {sueno} horas). Menos de 6 horas se asocia fuertemente a mayor estrés.
                </div>
                <div class="recommendation-list">
                    <strong>Reducción Digital:</strong> Reducir el consumo diario de redes ({redes} horas). Trata de limitar su uso a menos de 2 horas.
                </div>
            """, unsafe_allowed_html=True)
        elif pred == 'Medio':
            st.markdown(f"""
                <div class="recommendation-list">
                    <strong>Autocuidado:</strong> Incorpora pausas activas durante el día y técnicas de relajación (mindfulness, respiración diafragmática).
                </div>
                <div class="recommendation-list">
                    <strong>Apoyo Social:</strong> Apóyate en amigos o familiares (tu nivel de apoyo percibido es {apoyo}/10). Compartir tus preocupaciones reduce la carga emocional.
                </div>
                <div class="recommendation-list">
                    <strong>Actividad Física:</strong> Tu nivel actual es {act_fisica.lower()}. Intentar subir a un nivel de actividad moderada (30 min diarios de caminata).
                </div>
            """, unsafe_allowed_html=True)
        else:
            st.markdown(f"""
                <div class="recommendation-list" style="border-left-color: #10B981;">
                    <strong>Excelente:</strong> Tu nivel de riesgo estimado es bajo. Sigue manteniendo buenos hábitos de sueño, actividad física y socialización.
                </div>
                <div class="recommendation-list" style="border-left-color: #10B981;">
                    <strong>Prevención:</strong> Recuerda realizar chequeos periódicos de tu bienestar emocional y mantener un balance saludable entre vida laboral y digital.
                </div>
            """, unsafe_allowed_html=True)
            
# Información metodológica al pie
st.markdown("---")
st.caption("Desarrollado bajo la metodología de ciclo de vida CRISP-ML(Q) para proyectos de Inteligencia Artificial y Machine Learning. Todos los datos ingresados se procesan localmente y no son almacenados en ningún servidor externo.")
