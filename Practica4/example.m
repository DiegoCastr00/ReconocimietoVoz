clc;
clear;

% Cargar el archivo de audio
[audio, Fs] = audioread('perros3.wav'); 
N = length(audio);         % Número de muestras
t = (0:N-1) / Fs;          % Vector de tiempo para graficar la señal en segundos

% Graficar la señal de audio en el dominio del tiempo
figure;
subplot(2,1,1); 
plot(t, audio);
title('Señal de Audio en el Tiempo');
xlabel('Tiempo [s]');
ylabel('Amplitud');

% Aplicar la Transformada de Fourier al audio
Y = fft(audio);
P2 = abs(Y / N);           % Espectro de amplitud de dos lados
P1 = P2(1:N/2+1);          % Convertir a espectro de un solo lado
P1(2:end-1) = 2 * P1(2:end-1);  % Escalar las amplitudes

% Definir el dominio de frecuencia para el espectro de un solo lado
f = Fs * (0:(N/2)) / N;

% Graficar el espectro de amplitud de un solo lado
subplot(2,1,2);
plot(f, P1, 'LineWidth', 2);
title('Espectro de Amplitud de un Solo Lado del Audio');
xlabel('Frecuencia [Hz]');
ylabel('|P1(f)|');
