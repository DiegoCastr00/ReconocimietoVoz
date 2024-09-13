clc;
clear;

A = 2; % Amplitud de la señal
des = deg2rad(45); % Desplazamiento de fase
f = 125; % Frecuencia de la señal
fs = 1000; % Frecuencia de muestreo
tiempo = 10; % Tiempo total de grabación (milisegundos)

T = tiempo/1000; % mili a segundos
Ts = 1/fs; % Periodo de muestreo
t = 0:Ts:T-Ts; % Vector de tiempo (desde 0 hasta T, con pasos de Ts)

% Ecuación de la señal muestreada
x = A * sin(2*pi*f*t + des); 

disp('Valores de las muestras:');
disp(x);

figure;
stem(t, x, 'r', 'LineWidth', 1.5); 
hold on;
plot(t, x, 'b'); 
title('Señal muestreada');
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on;
