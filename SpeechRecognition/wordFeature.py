import librosa
import numpy as np
import scipy
from scipy.stats import skew, kurtosis

class WordFeatureExtractor:
    def __init__(self, sample_rate=44100, n_mfcc=13, n_mels=128, frame_length=0.025, 
                 frame_step=0.01, nfilt=26, window='hamming'):
        """
        Inicializa el extractor de características.
        
        Args:
            sample_rate (int): Frecuencia de muestreo
            n_mfcc (int): Número de coeficientes MFCC
            n_mels (int): Número de bandas mel
            frame_length (float): Longitud de la ventana en segundos
            frame_step (float): Paso entre ventanas en segundos
            nfilt (int): Número de filtros mel
            window (str): Tipo de ventana ('hamming', 'hanning', etc.)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.frame_length = int(frame_length * sample_rate)
        self.frame_step = int(frame_step * sample_rate)
        self.nfilt = nfilt
        self.window = window
        
    def extract_features(self, audio):
        """
        Extrae todas las características del audio.
        """
        features = {}
        
        # 1. Características temporales
        features.update(self._extract_temporal_features(audio))
        
        # 2. Características espectrales
        features.update(self._extract_spectral_features(audio))
        
        # 3. Características cepstrales
        features.update(self._extract_cepstral_features(audio))
        
        # 4. Características de energía
        features.update(self._extract_energy_features(audio))
        
        # 5. Características rítmicas
        features.update(self._extract_rhythm_features(audio))
        
        return features
    
    def _extract_temporal_features(self, audio):
        """Extrae características del dominio temporal."""
        features = {}
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, 
                                               frame_length=self.frame_length, 
                                               hop_length=self.frame_step)[0]
        features.update({
            'zcr_mean': np.mean(zcr), # Media
            'zcr_std': np.std(zcr), # Desviación estándar
            'zcr_skew': skew(zcr), #Asimetría (skew): Hacia dónde se inclina la distribución
            'zcr_kurtosis': kurtosis(zcr), # Qué tan puntiaguda es la distribución
            'zcr_median': np.median(zcr)  # Mediana
        })
        
        # Amplitud envelope: La forma general de la onda, útil para detectar patrones de intensidad
        envelope = np.abs(scipy.signal.hilbert(audio))
        features.update({
            'envelope_mean': np.mean(envelope),
            'envelope_std': np.std(envelope),
            'envelope_skew': skew(envelope),
            'envelope_kurtosis': kurtosis(envelope)
        })
        
        return features
    
    def _extract_spectral_features(self, audio):
        """Extrae características del dominio espectral."""
        features = {}
        
        # Centroide espectral: El "centro de masa" del espectro. Indica si dominan frecuencias altas o bajas.
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, 
                                                             sr=self.sample_rate,
                                                             n_fft=self.frame_length,
                                                             hop_length=self.frame_step,
                                                             window=self.window)[0]
        features.update({
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_centroid_skew': skew(spectral_centroids)
        })
        
        # Rolloff espectral: Frecuencia debajo de la cual está el 85% de la energía.
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, 
                                                          sr=self.sample_rate,
                                                          n_fft=self.frame_length,
                                                          hop_length=self.frame_step,
                                                          window=self.window)[0]
        features.update({
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff)
        })
        
        # Ancho de banda espectral: Dispersión de las frecuencias alrededor del centroide.
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, 
                                                              sr=self.sample_rate,
                                                              n_fft=self.frame_length,
                                                              hop_length=self.frame_step,
                                                              window=self.window)[0]
        features.update({
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth)
        })
        
        # Contraste espectral: Diferencia entre picos y valles en el espectro.
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, 
                                                            sr=self.sample_rate,
                                                            n_fft=self.frame_length,
                                                            hop_length=self.frame_step)
        features.update({
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_contrast_std': np.std(spectral_contrast)
        })
        
        # Flatness espectral: Qué tan similar es el espectro a ruido blanco vs. tono puro.
        spectral_flatness = librosa.feature.spectral_flatness(y=audio,
                                                            n_fft=self.frame_length,
                                                            hop_length=self.frame_step)[0]
        features.update({
            'spectral_flatness_mean': np.mean(spectral_flatness),
            'spectral_flatness_std': np.std(spectral_flatness)
        })
        
        return features
    
    def _extract_cepstral_features(self, audio):
        """Extrae características cepstrales."""
        features = {}
        
        # MFCC y sus derivadas
        mfccs = librosa.feature.mfcc(y=audio, 
                               sr=self.sample_rate, 
                               n_mfcc=self.n_mfcc,
                               n_fft=self.frame_length,
                               hop_length=self.frame_step,
                               window=self.window,
                               n_mels=self.nfilt) 
    
        # Delta y Delta-Delta: Cambios de primer y segundo orden de los MFCC.
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Estadísticas para cada coeficiente MFCC
        for i in range(self.n_mfcc):
            features.update({
                f'mfcc_{i}_mean': np.mean(mfccs[i]),
                f'mfcc_{i}_std': np.std(mfccs[i]),
                f'mfcc_{i}_skew': skew(mfccs[i]),
                f'mfcc_{i}_delta_mean': np.mean(mfccs_delta[i]),
                f'mfcc_{i}_delta_std': np.std(mfccs_delta[i]),
                f'mfcc_{i}_delta2_mean': np.mean(mfccs_delta2[i]),
                f'mfcc_{i}_delta2_std': np.std(mfccs_delta2[i])
            })
        
        return features
    
    def _extract_energy_features(self, audio):
        """Extrae características relacionadas con la energía."""
        features = {}
        
        # RMS Energy (Root Mean Square): Magnitud promedio de la señal.
        rms = librosa.feature.rms(y=audio,
                                frame_length=self.frame_length,
                                hop_length=self.frame_step)[0]
        features.update({
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'rms_skew': skew(rms),
            'rms_kurtosis': kurtosis(rms)
        })
        
        # Energía por bandas de frecuencia (Mel Spectrogram): 
        mel_spec = librosa.feature.melspectrogram(y=audio, 
                                                sr=self.sample_rate,
                                                n_fft=self.frame_length,
                                                hop_length=self.frame_step,
                                                n_mels=self.n_mels,
                                                window=self.window)
        
        # Dividir en tercios el espectrograma mel para energía por bandas
        band_size = self.n_mels // 3
        for i in range(3):
            band = mel_spec[i*band_size:(i+1)*band_size]
            features.update({
                f'band_{i}_energy_mean': np.mean(np.sum(band, axis=0)),
                f'band_{i}_energy_std': np.std(np.sum(band, axis=0))
            })
        
        return features
    
    def _extract_rhythm_features(self, audio):
        """Extrae características rítmicas."""
        features = {}
        
        # Onset strength (ataque): Dónde comienzan los eventos sonoros.  Detecta inicios de sonidos.
        onset_env = librosa.onset.onset_strength(y=audio, 
                                               sr=self.sample_rate,
                                               hop_length=self.frame_step)
        features.update({
            'onset_strength_mean': np.mean(onset_env),
            'onset_strength_std': np.std(onset_env)
        })
        
        # Tempo: Velocidad de la habla en BPM.
        tempo, _ = librosa.beat.beat_track(y=audio,  
                                        sr=self.sample_rate,
                                        hop_length=self.frame_step)
        features['tempo'] = np.asarray(tempo).item()
        
        return features
    
    def process_audio(self, audio):
        """
        Procesa un audio y extrae todas sus características.
        Returns:
            dict: Diccionario con todas las características
        """
        if len(audio) > self.sample_rate:
            audio = audio[:self.sample_rate]
        elif len(audio) < self.sample_rate:
            audio = np.pad(audio, (0, self.sample_rate - len(audio)))
        
        return self.extract_features(audio)