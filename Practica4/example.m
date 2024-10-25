clc;
clear;
[audio, Fs] = audioread('perros2.wav'); 
N = length(audio);  % Número de muestras
t = (0:N-1)/Fs;  % Vector de tiempo para graficar la señal en segundos

figure;
subplot(2,1,1); 
plot(t, audio);
title('Señal de Audio en el Tiempo');
xlabel('Tiempo [s]');
ylabel('Amplitud');

%%

X = 2*randn(size(Fs)) + audio;

plot(1000*t,X)
title("Signal Corrupted with Zero-Mean Random Noise")
xlabel("t (milliseconds)")
ylabel("X(t)")

%% 



figure;
subplot(2,1,1); 
plot(t, audio);
title('Señal de Audio en el Tiempo');
xlabel('Tiempo [s]');
ylabel('Amplitud');

% FFT
fourier = fft(audio);  % Transformada de Fourier
frecuencias = (0:floor(N/2)-1)*(Fs/N);  % Vector de frecuencias para la mitad positiva

% Graficar la FFT
subplot(2,1,2);  
plot(frecuencias, abs(fourier / N), 'k', 'LineWidth', 1.5);
title('Transformada de Fourier del Audio');
xlabel('Frecuencia [Hz]');
ylabel('Magnitud');

% Encontrar la frecuencia fundamental
[magnitud_max, idx] = max(abs(fourier));  % Encuentra el pico máximo
frecuencia_fundamental = frecuencias(idx);  % Obtiene la frecuencia correspondiente

disp(['La frecuencia fundamental es: ', num2str(frecuencia_fundamental), ' Hz']);
