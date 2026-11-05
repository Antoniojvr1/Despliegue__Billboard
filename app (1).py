import streamlit as st
import pandas as pd
import pickle

# Configuración de la página de Streamlit
st.set_page_config(page_title="Billboard Top 1 Predictor", layout="centered")

st.title("🎵 ¿Llegará tu canción al Top 1 de Billboard?")
st.write("Ingresa los detalles de la canción para estimar si tiene posibilidades de alcanzar la cima utilizando nuestro modelo de Inteligencia Artificial.")

# Cargar el modelo entrenado y sus columnas correspondientes
@st.cache_resource
def cargar_modelo():
    with open("modelo_billboard.pkl", "rb") as f:
        datos = pickle.load(f)
    return datos["modelo"], datos["columnas"]

try:
    modelo, columnas_entrenamiento = cargar_modelo()
    st.success("¡Modelo cargado correctamente!")
except Exception as e:
    st.error(f"No se pudo cargar el archivo 'modelo_billboard.pkl'. Asegúrate de subirlo a tu repositorio de GitHub junto a este script. Error: {e}")
    st.stop()

# Formulario de entrada de datos
with st.form("formulario_cancion"):
    artist_type = st.selectbox("Tipo de Artista", ["female", "male", "mixed"])
    mood = st.selectbox("Estado de ánimo (Mood)", ["Energizing", "Brooding", "Excited", "Yearning", "Defiant", "Cool", "Easygoing", "Empowering", "Fiery", "Gritty", "Lively", "Melancholy", "Other", "Peaceful", "Romantic", "Rowdy", "Sensual"])
    tempo = st.selectbox("Tempo", ["Medium Tempo", "Slow Tempo", "Fast Tempo"])
    genre = st.selectbox("Género musical", ["Pop", "Traditional", "Urban", "Alternative & Punk", "Electronica", "Rock", "Other", "Soundtrack", "Jazz"])
    durationSeg = st.number_input("Duración de la canción (en segundos)", min_value=10, max_value=3600, value=190)
    
    boton_predecir = st.form_submit_type = st.form_submit_button("Predecir Éxito")

if boton_predecir:
    # Estructuramos la entrada del usuario en un DataFrame
    datos_entrada = pd.DataFrame([{
        "artist_type": artist_type,
        "mood": mood,
        "tempo": tempo,
        "genre": genre,
        "durationSeg": durationSeg
    }])
    
    # Aplicamos codificación One-Hot para simular lo que se hizo en el entrenamiento
    datos_codificados = pd.get_dummies(datos_entrada)
    
    # Aseguramos que tenga exactamente las mismas columnas que el modelo entrenado
    datos_finales = datos_codificados.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # Hacer la predicción
    resultado = modelo.predict(datos_finales)[0]
    
    # Mostrar el resultado final
    st.subheader("Resultado de la Predicción:")
    if resultado == 1:
        st.balloons()
        st.success("🎉 ¡Sí, el modelo estima que esta canción PUEDE llegar al puesto 1 de Billboard!")
    else:
        st.warning("😔 El modelo estima que esta canción NO tiene la tendencia para alcanzar el puesto 1.")
