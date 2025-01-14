import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import time
import os
from datetime import datetime

from preprocesing import AudioPreprocessor 
from speakerFeatureModel import SpeakerFeatureExtractor
from wordFeatureModel import WordFeatureExtractor

from wordClassifierSVM import WordClassifierSVM
from speakerClassifierSVM import SpeakerClassifierSVM

def create_feature_dataset(audio_files, preprocessor, speaker_extractor, word_extractor, wordModel, speakerModel):

    # Preprocesar audio
    audio = preprocessor.process_audio(audio_files)
    
    # Extraer características
    speaker_features = speaker_extractor.process_audio(audio)
    word_features = word_extractor.process_audio(audio)
    
    # Clasificar
    speakerLabel = speakerModel.predict(speaker_features)
    wordLabel = wordModel.predict(word_features)
    
    return wordLabel, speakerLabel, audio

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
    wordModel = WordClassifierSVM.load_model('wordClassifier.joblib')
    speakerModel = SpeakerClassifierSVM.load_model('speakerClassifier.joblib')
    
    # Estado para controlar la grabación
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    
    col1, col2 = st.columns([1,1])
    
    with col2:
        # Botón de grabación
        if not st.session_state.recording:
            if st.button("🎙️ Grabar Audio"):
                st.session_state.audio_file = None
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
                
                print(wordLabel)
                print(speakerLabel)
                
                st.session_state.audio_file = filename
                st.session_state.recording = False
    
    # Mostrar audio grabado y resultados
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        st.write("---")
        st.write("📼 Audio Grabado:")
        st.audio(st.session_state.audio_file)
        
        # Aquí simularemos los resultados de los dos modelos
        # Reemplaza esto con tus modelos reales
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Resultado Modelo 1")
            st.info("Aquí puedes mostrar los resultados del primer modelo")
            # Ejemplo: st.write(resultado_modelo_1)
            
        with col2:
            st.write("### Resultado Modelo 2")
            st.warning("Aquí puedes mostrar los resultados del segundo modelo")
            # Ejemplo: st.write(resultado_modelo_2)

if __name__ == "__main__":
    create_recorder_app()