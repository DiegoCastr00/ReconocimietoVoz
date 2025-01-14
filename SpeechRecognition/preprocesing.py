import librosa
import numpy as np
from scipy import signal
import noisereduce as nr

class AudioPreprocessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path):
        """Carga un archivo de audio."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def normalize_volume(self, audio, target_dBFS=-20):
        """Normaliza el volumen del audio a un nivel objetivo en dBFS."""
        rms = librosa.feature.rms(y=audio)[0]
        mean_rms = np.mean(rms)
        current_dBFS = 20 * np.log10(mean_rms) if mean_rms > 0 else -np.inf
        adjustment = target_dBFS - current_dBFS
        normalized_audio = audio * (10 ** (adjustment / 20))
        # Prevenir clipping
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        return normalized_audio
    
    def remove_silence(self, audio, top_db=30):
        """Elimina silencios al inicio y final del audio."""
        return librosa.effects.trim(audio, top_db=top_db)[0]
    
    def apply_bandpass_filter(self, audio, lowcut=80, highcut=8000):
        """Aplica un filtro paso banda para reducir ruido."""
        nyquist = self.sample_rate // 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)
    
    def reduce_noise(self, audio, noise_audio=None):
        """Reduce el ruido de un audio utilizando un perfil de ruido."""
        if noise_audio is None:
            noise_audio = audio[:int(len(audio) * 0.1)]
        
        reduced_audio = nr.reduce_noise(y=audio, y_noise=noise_audio, sr=self.sample_rate)
        return reduced_audio
    
    def clip_prevention(self, audio, threshold=0.95):
        """Previene el clipping manteniendo la señal dentro de los límites."""
        max_val = np.max(np.abs(audio))
        if max_val > threshold:
            audio = audio * (threshold / max_val)
        return audio
    
    def process_audio(self, file_path):
        """Aplica toda la cadena de preprocesamiento a un archivo de audio."""
        # Cargar audio
        audio = self.load_audio(file_path)
        
        # Aplicar preprocesamiento
        audio = self.remove_silence(audio)
        audio = self.normalize_volume(audio)
        audio = self.apply_bandpass_filter(audio)
        audio = self.reduce_noise(audio)
        audio = self.clip_prevention(audio)
            
        return audio