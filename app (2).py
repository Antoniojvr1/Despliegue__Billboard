import streamlit as st
import pandas as pd
import pickle

# Configuración de la página con un tema moderno y oscuro
st.set_page_config(
    page_title="Billboard Top 1 Predictor",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilo CSS personalizado para un diseño creativo y futurista
st.markdown("""
<style>
    /* Fondo general de la aplicación */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    /* Personalización de botones */
    .stButton>button {
        background: linear-gradient(135deg, #1f4068, #162447);
        color: #ffffff;
        border-radius: 12px;
        border: 2px solid #e43f5a;
        padding: 0.6rem 2rem;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease-in-out;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #e43f5a, #1f4068);
        transform: translateY(-2px);
        box-shadow: 0px 4px 15px rgba(228, 63, 90, 0.4);
        color: #ffffff;
    }
    /* Título con gradiente de color */
    .title-text {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff4b4b, #e43f5a, #1f4068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    /* Subtítulo */
    .subtitle-text {
        text-align: center;
        font-style: italic;
        color: #00fff0;
        font-size: 1.2rem;
        margin-top: 5px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Cabecera de la aplicación
st.markdown("<h1 class='title-text'>🎵 Billboard Top 1 Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Desarrollado con Inteligencia Artificial por <b>Tu Nombre</b> ✨</p>", unsafe_allow_html=True)

st.markdown("🌟 **¿Tienes una canción que suena a éxito masivo?** Ingresa sus características para evaluar su probabilidad de reinar en las listas más competitivas del planeta.")
st.write("--- ")

# Cargar el modelo de forma segura utilizando la memoria caché
@st.cache_resource
def cargar_modelo():
    with open("modelo_billboard.pkl", "rb") as f:
        datos = pickle.load(f)
    return datos["modelo"], datos["columnas"]

try:
    modelo, columnas_entrenamiento = cargar_modelo()
    st.info("🤖 Inteligencia Artificial lista para predecir.")
except Exception as e:
    st.error(f"Error al cargar los recursos de IA. Verifica si 'modelo_billboard.pkl' está en tu carpeta de trabajo. Detalles: {e}")
    st.stop()

# Formulario con columnas y diseño optimizado
with st.form("formulario_cancion"):
    col1, col2 = st.columns(2)
    
    with col1:
        artist_type = st.selectbox("👥 Tipo de Intérprete", ["female", "male", "mixed"])
        mood = st.selectbox("🎭 Atmósfera Emocional (Mood)", ["Energizing", "Brooding", "Excited", "Yearning", "Defiant", "Cool", "Easygoing", "Empowering", "Fiery", "Gritty", "Lively", "Melancholy", "Other", "Peaceful", "Romantic", "Rowdy", "Sensual"])
        tempo = st.selectbox("⚡ Ritmo y Tempo", ["Medium Tempo", "Slow Tempo", "Fast Tempo"])
        
    with col2:
        genre = st.selectbox("🎼 Género Musical", ["Pop", "Traditional", "Urban", "Alternative & Punk", "Electronica", "Rock", "Other", "Soundtrack", "Jazz"])
        durationSeg = st.number_input("⏱️ Duración de la pista (segundos)", min_value=10, max_value=3600, value=190)
        
    st.write("")
    boton_predecir = st.form_submit_button("🚀 PROBAR ÉXITO")

if boton_predecir:
    # Preparar el dataframe para que coincida con el pipeline del modelo original
    datos_entrada = pd.DataFrame([{
        "artist_type": artist_type,
        "mood": mood,
        "tempo": tempo,
        "genre": genre,
        "durationSeg": durationSeg
    }])
    
    # Aplicar One-Hot Encoding idéntico al entrenamiento
    datos_codificados = pd.get_dummies(datos_entrada)
    datos_finales = datos_codificados.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # Ejecutar la inferencia
    resultado = modelo.predict(datos_finales)[0]
    
    st.write("--- ")
    st.subheader("📊 Análisis del Algoritmo:")
    
    if resultado == 1:
        st.balloons()
        st.success("🎉 **¡SÍ! El modelo predice que esta canción tiene todo el ADN necesario para llegar al puesto #1 en Billboard.**")
    else:
        st.warning("😔 **El modelo sugiere que esta canción no alcanzará el puesto #1.** ¡Sigue experimentando con los parámetros! ")
