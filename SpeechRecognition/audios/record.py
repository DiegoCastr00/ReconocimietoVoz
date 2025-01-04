#pip install sounddevice scipy
import os
import time
import sounddevice as sd
from scipy.io.wavfile import write

DURATION = 1  
SAMPLE_RATE = 44100  
NUM_RECORDINGS = 100

OUTPUT_FOLDER = "cero"  # Carpeta de salida
FILE_NAME = f"Valeria_{OUTPUT_FOLDER}"  # Nombre del archivo

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for i in range(3):
    print(f"Comenzando en {3 - i}...")
    time.sleep(1)
    
print("Comenzando grabaciones...")
    
# Grabar audios
for i in range(1, NUM_RECORDINGS + 1):
    print(f"Grabando audio {i}/{NUM_RECORDINGS}...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait() 

    filename = os.path.join(OUTPUT_FOLDER, f"{FILE_NAME}_{i:03d}.wav")
    write(filename, SAMPLE_RATE, audio_data)
    print(f"Audio {i} guardado como '{filename}'.")
    
    
    # time.sleep(1) #Quitalo si quereis mas velocidad 

print("Grabaciones completadas.")