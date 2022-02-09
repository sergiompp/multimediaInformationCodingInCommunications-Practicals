% 2021
% -------------------------------------------------------------------------
% PREGUNTA 1 - VOCODER 
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------

clc
clear all

% Codifique la señal almacenada en prueba8.wav utilizando su función 
% analysis.m con los siguientes parámetros: orden de predicción=20, 
% alen=128, ulen=64.

[audio, Fs] = audioread('prueba8.wav');
ventanaAudio = hamming(256);    

M = 20;
alen = 128;
ulen = 64;

[E, ZC, V, A, naf] = analysis(audio, alen, ulen, M);

P = pitch(audio, alen, ulen);

% Reconstruya la señal utilizando la función synthesis.m y guárdela en un 
% vector llamado x_dec.

U = ulen;
[x_dec] = synthesis(E, V, A, P, U);

% Cree un vector de flags sonoro/sordo que tenga la misma longitud que el 
% original devuelto por su función analysis.m, pero con todos sus valores a 0.

V_nuevo = zeros(naf, 1);

% Reconstruya la señal utilizando la función synthesis.m con el nuevo 
% vector de flags. Guárdela en un vector llamado x_dec_2.

[x_dec_2] = synthesis(E, V_nuevo, A, P, U);
plot_analysis(x_dec_2, E, ZC, V, A, P, naf);

% Escuche ambas señales y comente el impacto que ha tenido modificar el 
% vector de flags. ¿A qué se debe? Justifique su respuesta.

% Señal Original - DMOS 5
% sound(audio, Fs)

% Señal Reconsrtuida - DMOS 4
 sound(x_dec_2, Fs)

 % Guarde x_dec y x_dec_sordo en un fichero llamado a5_decodificados.mat y 
 % añádalo al .zip que subirá al entregador al terminar el exámen.

 filename = 'a5_decodificados.mat';
 save(filename, 'x_dec', 'x_dec_2');
 
%% 
% -------------------------------------------------------------------------
% PREGUNTA 2 - 
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------

clc
clear all

% Utilice los siguientes comandos de Matlab para leer el audio 'prueba8.wav' 
% incluído con el material de la práctica 1 y seleccionar dos fragmentos 
% del mismo (x1,x2).
% 

[x, Fs] = audioread('prueba8.wav');
x1 = x(8800:9800-1);
x2 = x(12800:13800-1);


% Un fragmento es sonoro y el otro es sordo. ¿Puede identificar mediante sus 
% espectros cuál es cuál?. Justifique su respuesta.


figure(1)
subplot(2,1,1);
plot(x1)
title('audio 1')
xlabel('Frecuencia [Hz]');

subplot(2,1,2);
plot(x2)
title('audio 2')
xlabel('Frecuencia [Hz]');
 

% Genere una gráfica en el dominio de la frecuencia en la que se superponga
% el espectro de la trama sonora de la señal de voz y su envolvente espectral. 
% Para ello elija un orden de predicción y una longitud de ventana que 
% considere apropiados. No se olvide de etiquetar correctamente el eje de 
% frecuencias en Hz y de representar exclusivamente el eje de frecuencias 
% positivo. Guarde su imagen como 'actividad2.jpg' y súbala al final de la 
% práctica utilizando el entregador que encontrará junto a este cuestionario.

N = 256;        % Longitud de la trama = Longitud de un fonema
S = 12800; 
M = 12;

xf = x(S:S+N-1).*hamming(N); %Audio multiplicado por ventana de Hamming
transformFourier = fft(xf);   %transformada de Fourier del audio

ejeFrecuencias = linspace(0, Fs/2, N/2 + 1);
figure(3)
plot(ejeFrecuencias, 10*log10(abs(transformFourier(1:N/2+1)).^2));
title('DFT de la señal "Audio" en frecuencias');
xlabel('Frecuencia [Hz]');
ylabel('dB');

c = xcorr(xf, xf, M);               
[a, e]= levinson(c(M+1:2*M+1));

% a es un vector cuyo primer elemento es siempre 1.
a = a(:);                           % Necesitamos que sea un vector columna

% Tracto Vocal
h = zeros(N,1);
for k=0:N-1
    h(k+1) = 1 / (a'*exp(-1i*2*pi*k/N*(0:M)'));
end


figure(3)
hold on
plot(ejeFrecuencias, 10*log10(e*abs(h(1:N/2+1)).^2), 'r');
title('DFT de la señal "Audio" en frecuencias');
legend('DFT del tramo sonoro', 'Envolvente de la Señal');
xlabel('Frecuencia [Hz]');
ylabel('dB');
hold off


%% 
% -------------------------------------------------------------------------
% PREGUNTA 3 - Efectos en frecuencia del enventanado
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------

clc
clear all
% 
% Utilice el siguiente comando en Matlab para generar una señal mezcla de 
% dos tonos puros (dos senos con frecuencias distintas):

y = sin(2.5133*(0:15000-1)') + sin(2.8274*(0:15000-1)');

% asuma una frecuencia de muestreo de Fs=200 Hz (muestras/segundo). 
% Su tarea consiste en averiguar las frecuencias de los dos senos que 
% componen la señal 'y' mediante el análisis del espectro en frecuencia de 
% la señal. Para ello, le proponemos la siguiente estrategia:


% 1. Genere una ventana hamming de 20 muestras y extraiga un segmento 
% enventanado de 'y'. Llame al fragmento enventanado 'yf'.
N = 20;
ventanaAudio = hamming(N);                    % Ventana de 20 muestras
Fs = 200;

S = 1;

yf = y(S:S+N-1).*ventanaAudio;
 
% 2. Calcule la FFT de 'yf' y representela en una Figura. Represente 
% exclusivamente el eje de frecuencias positivo. Asegúrese de que la 
% magnitud de los coeficientes está en unidades logarítmicas.

transformFourier = fft(yf);

figure(1); clf;

% Se expresa la DFT en dB
t=linspace(0, Fs/2, N/2+1);
subplot(2,1,1);
plot(t,10*log10(abs(transformFourier(1:N/2+1)).^2));
title('DFT de la señal');
xlabel('Muestras (Mitad de la longitud N)');
ylabel('dB');

% 3. Modifique el eje x de la Figura anterior para que muestre frecuencias en 
% lugar de muestras.

% Pasándolo a frecuencias
ejeFrecuencias = linspace(0, Fs/2, N/2 + 1);
figure(1)
subplot(2,1,2);
plot(ejeFrecuencias, 10*log10(abs(transformFourier(1:N/2+1)).^2));
title('DFT de la señal en frecuencias');
xlabel('Frecuencia [Hz]');
ylabel('dB');

% 4. Modifique ahora el tamaño de la ventana hamming a 200 muestras y 
% vuelva a repetir los 3 apartados anteriores. Guarde esta última figura 
% con el nombre 'actividad1.jpeg' y súbala al final de la práctica al 
% entregador.

N_200 = 200;
ventanaAudio_200 = hamming(N_200);                   % Ventana de 200 muestras

yf_200 = y(S:S+N_200-1).*ventanaAudio_200;

transformFourier_200 = fft(yf_200);

figure(2); clf;

% Se expresa la DFT en dB
t_200=linspace(0, Fs/2, N_200/2+1);
subplot(2,1,1);
plot(t_200,10*log10(abs(transformFourier_200(1:N_200/2+1)).^2));
title('DFT de la señal con N = 200');
xlabel('Muestras (Mitad de la longitud N)');
ylabel('dB');

% Pasándolo a frecuencias
ejeFrecuencias_200 = linspace(0, Fs/2, N_200/2 + 1);
figure(2)
subplot(2,1,2);
plot(ejeFrecuencias_200, 10*log10(abs(transformFourier_200(1:N_200/2+1)).^2));
title('DFT de la señal en frecuencias');
xlabel('Frecuencia [Hz]');
ylabel('dB');

figure(3)
plot(abs(transformFourier_200));
title('Valor Absoluto de la FFT de la señal "Audio de Teléfono" en frecuencias');
xlabel('Frecuencia [Hz]');
ylabel('dB');


%% 
clc
clear all

% -------------------------------------------------------------------------
% PREGUNTA 4 - El espectrograma.
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------

[audio, Fs] = audioread('prueba8.wav');

x = audio;       % Tramo Sonoro de la señal
alen = 256;                         % Longitud de la ventana de análisis
ulen = 32;                          % Despalzamiento de actualización

N = length(x);
naf = floor((N-alen+ulen)/ulen); %Número de ventanas

f = linspace(0,Fs/2,alen/2+1);
S = zeros(alen/2+1, naf);
t = linspace(0,naf*20e-3,naf);

Espectro = CIMC_spectrogram(x, alen, ulen);


% Ploteado de Espectrograma
figure(1)

imagesc(t,f,Espectro); axis xy
title('Espectrograma');
xlabel('Tiempo (seg)');
ylabel('Frecuencia (Hz)')
colorbar

%% 

clc
clear all

% -------------------------------------------------------------------------
% PREGUNTA 5  -Estimación de parámetros 
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------


%% 

function S = CIMC_spectrogram(x, alen, ulen)
    
    %x es la señal de voz
    %alen es la longitud de la ventana de análisis
    %ulen es el desplazamiento de actualización

    % una vez tenga el espectrograma en S, puede utilizar imagesc para
    % visualizarlo. (Fs frecuencia muestreo).
    % imagesc( (ulen/Fs)*(1:naf), (Fs/alen)*(0:alen/2), S );
    
    N = length(x);
    naf = floor((N - alen+ulen)/ulen);      % Número de ventanas
    n1 = 1;
    n2 = alen;
    
    %tiempo = 0:1/(Fs):alen/Fs;
    %tiempo = tiempo(1:end-1);
    %frecuencia = 0:Fs/alen:Fs/2;

    
    S = zeros(alen/2+1, naf);
    
    for n=1:naf                     % Contador sobre el número de ventana
        xf = x(n1:n2).*hanning(alen);
        X = fft(xf);
        
       n1 = n1 + ulen;             % Desplaza la ventana (muestra inicial)
        n2 = n2 + ulen;             % Desplaza la ventana (muestra final)
        
        S(:,n) = 10*log10(abs(X(1:alen/2+1)).^2);
    end
    
    
end

function [E, ZC, V, A, naf] = analysis(x, alen, ulen, M)
    
    N = length(x);
    naf = floor((N-alen+ulen)/ulen); %Número de ventanas

    % Initialization
    E = zeros(naf, 1);      % Energía de la señal
    ZC = zeros(naf, 1);     % Número de cruces por 0
    V = zeros(naf, 1);      % Decisor (0 = Sordo, 1 = Sonoro)
    A = zeros(naf, M+1);    % Coeficientes de Predición Lineal
                            % M es el orden de predicción.
                            % M+1 permite que guardemos el 1 inicial como en la actividad 2.
                            % Dentro del bucle
    n1 = 1;
    n2 = alen;

    for n = 1:naf %Contador sobre el número de ventana
        xf = x(n1:n2).*hanning(alen);
        
        
        E(n,1) = (1/alen)*sum(abs(xf).^2);
        for k = 1:alen-1
            if ((xf(k)>=0 && xf(k+1)<0) || (xf(k)<=0 && xf(k+1)>0))
                ZC(n,1) = ZC(n,1) + 1;
            end
        end

        ZC(n,1) = (1/alen)*ZC(n,1);

        if (ZC(n,1) > 0.100)%0.375
            V(n,1) = 1;
        end

        c = xcorr(xf, xf, M); % M es el orden de prediccion
        [A(n,:), ~] = levinson(c(M+1:2*M+1));

        n1 = n1 + ulen;
        n2 = n2 + ulen;
    end

end

function plot_analysis(x, E, ZC, V, A, P, naf)

    figure(15);
    clf;
    subplot(3,2,1)
    plot(x) % Señal de entrada
    axis([1 length(x) min(x) max(x)]);
    title('Señal Entrada');
    xlabel('Muestras');
    ylabel('Amplitud');

    subplot(3,2,2)
    plot(sqrt(E)) % Desviación estándar de la energía
    axis([1 length(E) min(sqrt(E)) max(sqrt(E))]);
    title('Desv. Estándar Energía');
    xlabel('Frecuencia');
    ylabel('Energía');

    subplot(3,2,3)
    plot(V) % Decisión sonoro/sordo
    axis([1 length(V) 0 1]);
    title('Decisión Sonoro / Sordo');
    xlabel('Tiempo');
    ylabel('Frecuencia');

    subplot(3,2,4)
    plot(ZC) % Número de cruces por cero
    axis([1 length(ZC) min(ZC) max(ZC)]);
    title('Num. Cruces Por 0');
    xlabel('Frecuencia');

    subplot(3,2,5)
    F = 8000./P;
    plot(F) % Frecuencia fundamental en Hz
    axis([1 length(F) 0 600]);
    title('Frecuencia Fundamental');
    xlabel('Frecuencia [Hz]');
    ylabel('Frecuencia [Hz]');

    subplot(3,2,6)
    S = zeros(512, naf);
    for n=1:naf
        S(:,n) = 20*log10(abs(freqz(1,A(n,:),512)));
    end
    S = flipud(S);
    imagesc(flipud(-S));
    %colormap(gray);
    imagesc(S); % Envolvente spectral en estilo espectrograma
    title('Envolvente espectral (espectograma)');
    xlabel('Tiempo');
    ylabel('Frecuencia [Hz]');
end
