import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import time
import os
from datetime import datetime
import pandas as pd
import numpy as np

from preprocesing import AudioPreprocessor
from speakerFeature import SpeakerFeatureExtractor
from wordFeature import WordFeatureExtractor
from wordClassifierSVM import WordClassifierSVM
# from speakerClassifierSVM import SpeakerClassifierSVM
from speakerClassifierRF import SpeakerClassifierRF
def create_feature_dataset(audio_files, preprocessor, speaker_extractor, word_extractor, wordModel, speakerModel):
    # Preprocesar audio
    audio = preprocessor.process_audio(audio_files)
    
    # Extraer características
    speaker_features = speaker_extractor.process_audio(audio)
    word_features = word_extractor.process_audio(audio)
    speaker_df = pd.DataFrame([speaker_features])
    word_df = pd.DataFrame([word_features])
    
    # Clasificar
    speakerLabel = speakerModel.predict(speaker_df)
    wordLabel = wordModel.predict(word_df)
    
    # Limpiar las etiquetas
    speakerLabel = speakerLabel[0] if isinstance(speakerLabel, (list, np.ndarray)) else speakerLabel
    wordLabel = wordLabel[0] if isinstance(wordLabel, (list, np.ndarray)) else wordLabel
    
    # Asegurarse de que son strings limpios sin corchetes ni comillas
    speakerLabel = str(speakerLabel).strip('[]\'\"')
    wordLabel = str(wordLabel).strip('[]\'\"')
    
    return wordLabel, speakerLabel, audio

def clean_state():
    """Limpia el estado de la aplicación y elimina archivos temporales."""
    st.session_state.recording = False
    st.session_state.audio_file = None
    # Eliminar archivos previos en el directorio de grabaciones
    if os.path.exists("recordings"):
        for file in os.listdir("recordings"):
            file_path = os.path.join("recordings", file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def create_recorder_app():
    st.title("Grabadora y Analizador de Audio")
    
    # Crear directorio para guardar audios si no existe
    if not os.path.exists("recordings"):
        os.makedirs("recordings")
    
    # Variables de configuración
    DURATION = 1  # duración en segundos
    SAMPLE_RATE = 44100
    
    preprocessor = AudioPreprocessor()
    speaker_extractor = SpeakerFeatureExtractor()
    word_extractor = WordFeatureExtractor()
    wordModel = SpeakerClassifierRF.load_model('wordClassifierXG.joblib')
    speakerModel = SpeakerClassifierRF.load_model('speakerClassifierXG.joblib')
    
    # Inicializar estado
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        # Botón de grabación
        if st.button("🎙️ Grabar Audio"):
            clean_state()  # Limpiar estado y variables
            st.session_state.recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/audio_{timestamp}.wav"
            
            # Mensaje de cuenta regresiva
            with st.spinner("Preparando grabación..."):
                for i in range(3, 0, -1):
                    st.write(f"Comenzando en {i}...")
                    time.sleep(1)
            
            # Grabar audio
            st.write("🔴 Grabando...")
            audio_data = sd.rec(int(DURATION * SAMPLE_RATE),
                                samplerate=SAMPLE_RATE,
                                channels=1,
                                dtype='int16')
            sd.wait()
            # Guardar audio
            wav.write(filename, SAMPLE_RATE, audio_data)
            
            wordLabel, speakerLabel, audio = create_feature_dataset(filename,
                                                                    preprocessor,
                                                                    speaker_extractor,
                                                                    word_extractor,
                                                                    wordModel,
                                                                    speakerModel)
                        
            st.session_state.audio_file = filename
            st.session_state.recording = False
    
    # Mostrar audio grabado y resultados
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        st.write("---")
        st.write("📼 Audio Grabado:")
        st.audio(st.session_state.audio_file)
        
        # st.write("---")
        # st.write("📼 Audio Modificado:")
        # st.audio(audio, sample_rate=SAMPLE_RATE)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Palabra Detectada")
            st.info(wordLabel)

        with col2:
            st.write("### Orador Detectado")
            st.warning(speakerLabel)

if __name__ == "__main__":
    create_recorder_app()