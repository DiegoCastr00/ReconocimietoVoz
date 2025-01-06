import librosa
import numpy as np
from scipy.stats import skew, kurtosis

class SpeakerFeatureExtractor:
    def __init__(self, sample_rate=44100, n_mfcc=40, n_mels=128, frame_length=0.025,
                 frame_step=0.01, nfilt=40, window='hamming'):
        """
        Inicializa el extractor de características para identificación de locutor.
        
        Args:
            sample_rate (int): Frecuencia de muestreo
            n_mfcc (int): Número de coeficientes MFCC (mayor que para palabras)
            n_mels (int): Número de bandas mel
            frame_length (float): Longitud de la ventana en segundos
            frame_step (float): Paso entre ventanas en segundos
            nfilt (int): Número de filtros mel
            window (str): Tipo de ventana
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
        Extrae todas las características relevantes para identificación de locutor.
        """
        features = {}
        
        # 1. Características fundamentales de la voz
        features.update(self._extract_fundamental_features(audio))
        
        # 2. Características prosódicas
        features.update(self._extract_prosodic_features(audio))
        
        # 3. Características espectrales y cepstrales
        features.update(self._extract_spectral_features(audio))
        
        # 4. Características de calidad de voz
        features.update(self._extract_voice_quality_features(audio))
        
        return features
    
    def _extract_fundamental_features(self, audio):
        """Extrae características fundamentales de la voz."""
        features = {}
        
        # Pitch (F0) usando PYIN (más preciso que YIN)
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                   fmin=librosa.note_to_hz('C2'),
                                                   fmax=librosa.note_to_hz('C7'),
                                                   sr=self.sample_rate)
        
        # Estadísticas de F0 (solo para frames con voz)
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            features.update({
                'f0_mean': np.mean(f0_voiced),
                'f0_std': np.std(f0_voiced),
                'f0_min': np.min(f0_voiced),
                'f0_max': np.max(f0_voiced),
                'f0_range': np.ptp(f0_voiced),
                'f0_skew': skew(f0_voiced),
                'f0_kurtosis': kurtosis(f0_voiced)
            })
        
        # Porcentaje de frames con voz
        features['voiced_fraction'] = np.mean(voiced_flag)
        
        return features
    
    def _extract_prosodic_features(self, audio):
        """Extrae características prosódicas."""
        features = {}
        
        # Energía y sus variaciones
        rms = librosa.feature.rms(y=audio,
                                frame_length=self.frame_length,
                                hop_length=self.frame_step)[0]
        
        # Estadísticas de energía
        features.update({
            'energy_mean': np.mean(rms),
            'energy_std': np.std(rms),
            'energy_range': np.ptp(rms),
            'energy_skew': skew(rms),
            'energy_kurtosis': kurtosis(rms)
        })
        
        # Tasa de cruces por cero (relacionada con la frecuencia fundamental)
        zcr = librosa.feature.zero_crossing_rate(audio, 
                                               frame_length=self.frame_length,
                                               hop_length=self.frame_step)[0]
        features.update({
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'zcr_skew': skew(zcr)
        })
        
        return features
    
    def _extract_spectral_features(self, audio):
        """Extrae características espectrales y cepstrales."""
        features = {}
        
        # MFCC con más coeficientes para capturar características del tracto vocal
        mfccs = librosa.feature.mfcc(y=audio, 
                                   sr=self.sample_rate,
                                   n_mfcc=self.n_mfcc,
                                   n_fft=self.frame_length,
                                   hop_length=self.frame_step,
                                   window=self.window,
                                   n_mels=self.nfilt)
        
        # Delta y Delta-Delta
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Estadísticas para cada coeficiente MFCC y sus derivadas
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
        
        # Formantes (resonancias del tracto vocal)
        spec = np.abs(librosa.stft(audio, n_fft=self.frame_length, 
                                 hop_length=self.frame_step, 
                                 window=self.window))
        
        # Características espectrales adicionales
        spectral_centroid = librosa.feature.spectral_centroid(S=spec, 
                                                            sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spec, 
                                                              sr=self.sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(S=spec, 
                                                            sr=self.sample_rate)
        
        features.update({
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_contrast_std': np.std(spectral_contrast)
        })
        
        return features
    
    def _extract_voice_quality_features(self, audio):
        """Extrae características de calidad de voz."""
        features = {}
        
        # Jitter (variación de la frecuencia fundamental)
        f0, voiced_flag, _ = librosa.pyin(audio, 
                                        fmin=librosa.note_to_hz('C2'),
                                        fmax=librosa.note_to_hz('C7'),
                                        sr=self.sample_rate)
        
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 1:
            # Jitter como la variación relativa promedio entre períodos consecutivos
            jitter = np.mean(np.abs(np.diff(f0_voiced))) / np.mean(f0_voiced)
            features['jitter'] = jitter
        
        # Shimmer (variación de la amplitud)
        rms = librosa.feature.rms(y=audio,
                                frame_length=self.frame_length,
                                hop_length=self.frame_step)[0]
        if len(rms) > 1:
            # Shimmer como la variación relativa promedio de la amplitud
            shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
            features['shimmer'] = shimmer
        
        # Harmonics-to-Noise Ratio (HNR)
        # Aproximado usando la relación entre energía armónica y ruido
        S = np.abs(librosa.stft(audio, n_fft=self.frame_length, 
                               hop_length=self.frame_step, 
                               window=self.window))
        
        # Calcular la energía armónica y el ruido
        harmonic, percussive = librosa.decompose.hpss(S)
        harmonic_energy = np.mean(harmonic**2)
        noise_energy = np.mean(percussive**2)
        
        if noise_energy > 0:
            hnr = 10 * np.log10(harmonic_energy / noise_energy)
            features['hnr'] = hnr
        
        return features
    
    def process_audio(self, audio):
        """
        Procesa un audio y extrae todas sus características.
        """
        if len(audio) > self.sample_rate:
            audio = audio[:self.sample_rate]
        elif len(audio) < self.sample_rate:
            audio = np.pad(audio, (0, self.sample_rate - len(audio)))
        return self.extract_features(audio)
