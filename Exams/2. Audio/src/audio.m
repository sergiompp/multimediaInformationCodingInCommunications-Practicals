% 2021
% Examen Lab. 2 - Codif. Audio
% -------------------------------------------------------------------------
% PREGUNTA 1
% -------------------------------------------------------------------------

clc
clear all

% 1.1) Cargue el audio con audioread y pase las muestras a enteros
archivo = 'audio3.wav';
[N, Fs, Nb,audio, ye] = cargaInfoAudio(archivo);

L = 400; % Muestras por trama 
% Para recorrer las tramas, se llama a la función "recorreTramas"
[mae_ei, ma_xfi, minMAE,energiaSenalOriginal, energiaResiduo, senalE, selPred]=recorreTrama(ye, L);

figure
subplot(1,2,1);
plot(energiaSenalOriginal);
hold on
plot(energiaResiduo);
legend('Energia - Señal Original','Energia - Residuo');
hold off
title('Energías');

subplot(1,2,2);
plot(mae_ei);
hline = refline([0 ma_xfi]);
hline.Color = 'r';
legend('Mean Absolute Error para cada predictor','Media Absoluta de Tramas');
title('Valores Medidos vs Valores Errrores Medios');

y=linspace(1,sfdr(round(audio)),50); 
figure
subplot(1,2,1);
histogram(round(audio));
title('Frecuencia de los valores de la señal original');

subplot(1,2,2);
histogram(senalE);
title('Frecuencia de los valores del residuo');

%% 
% -------------------------------------------------------------------------
% PREGUNTA 2 
% -------------------------------------------------------------------------

x1 = golomb(90,6);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 - Codificación de Golomb-Rice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xf = ye(1:1000);
[E_golomb, mae_e, ma_xf] = lp(xf);

k = ceil(abs(ceil(log2(mean(((E_golomb)))))));
k = ceil(mean(k));
if k<0
    k = 0;
end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 - Codificador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.1
[audio1, Fs_audio1] = audioread('audio1.wav');
[audio3, Fs_audio3] = audioread('audio3.wav');

%sound(audio1, Fs_audio1)
%sound(audio3, Fs_audio3)
figure
plot(audio1);

figure
plot(audio3);

%% 3.2
inputFile   = audio1;
codecFile   = 'audio1.codec';
decodedFile = 'audio1_decoded.wav'; 


L1 = 10; 
L2 = 400; 
L3 = 20000;


% Codificar y decodificar la señal
[N, fs, Nb, ye, K, P]  = codificador(inputFile, codecFile, L2);
yeD = decodificador(codecFile, decodedFile, L2, N, fs, Nb); 
predictorElegido = P;
KPorTrama = K;

%% 3.3


%%
% Funciones


function [E, mae_e, ma_xf] = lp(xf)

    % Los cuatro predictores lineales se pueden modelar como filtros FIR. 
    % Para ello, necesitamos generar un vector con los coefficientes 
    % asociados a cada muestra de la señal de entrada y aplicar la función 
    % 'filter' de la siguiente forma:
    % 
    % y[n] = a*x[n] - b*x[n-1] + c*x[n-2] + ... + d*x[n-z]
    % 
    % coefs = [a, -b, c, ..., d]
    % y = filter( coefs, 1, x )
    %
    % Si un determinado x[n-i] no se utilizar en el filtro su coef = 0
    %
    % La función devuelve:
    % E:     error de predicción para cada muestra y predictor
    % mae_E: mean absolute error para cada predictor
    % ma_xf: mean absolute value of xf

   
    
    % Coeficientes
    
    filtros_1= [0];               
    filtros_2 = [0 1];
    filtros_3 = [0 2 -1];
    filtros_4= [0 3 -3 1];

    L = length(xf);         % Longitud de trama de la señal de audio
    E = zeros(L,4);         % Matriz Lx4 son los residuos

    xpred_1 = filter(filtros_1,1,xf);
    E(:,1) = xf - xpred_1;
    xpred_2 = filter(filtros_2,1,xf);
    E(:,2) = xf - xpred_2;
    xpred_3 = filter(filtros_3,1,xf);
    E(:,3) = xf - xpred_3;
    xpred_4 = filter(filtros_4,1,xf);
    E(:,4) = xf - xpred_4;
    
    
    mae_e = mean(abs(E)); 
    ma_xf = mean(abs(xf));

end

function [xb] = golomb(xn, k)

    % inputs
    %   xn: una muestra individual (en nuestro caso del residuo)
    %   k : el parémtro k calculado para el algoritmo de Golomb-Rice
    %
    % outputs:
    %   xb: la codificación binaria de la muestra xn de acuerdo con el
    %   codec que estamso utilizando en formato char, ej: '101101110'.
    
    % Además de en el enunciado, el proceso está descrito en el documento
    % lossless_compression.pdf. Puede utilizar la función dec2bin() para
    % pasar números decimales a binarios
   
    xb = 0;
    
    % 1 - Bit del signo
    if xn < 0     
        bitSigno = 1;
    elseif xn > 0    
        bitSigno = 0;
    end
        
    % 2 - Parte unaria y parte binaria (lossless_compression.pdf)
    m=2.^k;
    
    % x(n) =  2m + resto
    
    unaryPart = str2num(dec2bin((abs(xn)./m)));
    binPart = str2num(dec2bin(mod(abs(xn),m)));

    % 3 - Finalmente el stop bit
    stopBit = 1;

    % 4 - Concatene los chars de bits en el orden CORRECTO para generar la
    % codificación binaria para la muestra xb
    xb = [bitSigno unaryPart binPart stopBit];
end

function [N, fs, Nb, ye, K, P]  = codificador(inputFile, outputFile, L) 
    % inputs:
    %   inputFile:  the name of the .wav audio file to encode;
    %   outputFile: the name of the encoded audio file to create;
    %   L : the length of the window, e.g. for a 25 ms length 25e-3 * fs;
    %
    % outputs:
    %   N:  the number of samples in the input file
    %   fs: the sampling rate
    %   Nb: the number of bits per sample in the input file
    %   ye: the input file integer-encoded and casted to double
    %   K:  the K value for the Golomb-Rice coging for each frame
    %   P:  the index of the best linear predictor for each frame

    % lectura del fichero de audio
    [N, fs, Nb, ~, ye] = cargaInfoAudio(inputFile);
    
    % inicialización indices y variables
    L = 25e-3 * fs;

    % Para recorrer las tramas de muestras
    N = length(ye);
    alen = 128;
    ulen = alen;
    naf = floor((N-alen+ulen)/ulen); % Numero de ventanas
    n1 = 1;
    n2 = alen;

    f  = fopen(outputFile, 'wb');

    for n = 1:naf
        % selecciona la siguiente trama
        xf = ye(n1:n2);

        % predicción lineal y extracción de mejor predictor y mejor residuo
        
            % Para esta trama, calcula los predictores lineales
            [Ei, mae_ei, ~] = lp(xf);

            % Seleciona el mejor predictor basado en el mar_e del residuo
            [~, idx] = min(mae_ei);
            bestP = idx;                   % idx = Mejor Predictor
        fwrite( f, bestP - 1, 'ubit2' );

        % calcular k para esta trama
        k = ceil(log2(mean((abs(Ei(:,bestP))))));
        k = k-2;
        if k < 0
            k = 0;
        end
        fwrite( f, k, 'ubit4' );

        % almacena el valor de k y el predictor elegido para esta trama
        K(n,1) = k;
        P(n) = bestP;
        
        % calculamos la codificación para cada muestra de la trama
        for i = 1:alen
           xb = golomb(xf, k);
           fwrite( f, logical(xb(:)'-'0'), 'ubit1' );
        end
        
        n1 = n1 + ulen;
        n2 = n2 + ulen;
    end

    fclose(f);
    
end

function [pi,edges] = calculoProb(source)
    % inputs:
    %   source:  información a ser tratada;
    %
    % outputs:
    %   pi:  probabilidades de la informacion
    %   edges: alfabeto 
   
    edges = unique(source);     % Crea el alfabeto separando cada uno de los
                                % elementos que componen la información
    counts = histcounts(source(:),edges);    % Cuenta cada uno de los elementos
    pi = transpose(counts./length(source));     % Calcula la probafbilidad 
                                                % de aparición de los
                                                % elementos
end

function [N, Fs, Nb,audio, ye] = cargaInfoAudio(ficheroAudio)
    % inputs:
    %   ficheroAudio:  the name of the .wav audio file;
    %
    % outputs:
    %   N:  the number of samples in the input file
    %   Fs: the sampling rate
    %   Nb: the number of bits per sample in the input file
    %   audio: the output file
    %   ye: the input file integer-encoded and casted to double
   

    [audio, Fs] = audioread(ficheroAudio);
    infoAudio = audioinfo(ficheroAudio);
    
    N = infoAudio.SampleRate;
    Nb = infoAudio.BitsPerSample;    % num Bits = 16 de audio info

    ye = uencode(audio,Nb,1,'signed');
    ye = double(ye);
end

function [mae_ei, ma_xfi, minMAE, energiaSenalOriginal, energiaResiduo, senalE, selPred]=recorreTrama(ye, L)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % inputs:
    %   ye:  trama con muestras a enteros;
    %   L: Longitud de las tramas
    % outputs:
    %   mae_ei: 
    %   ma_xfi: 
    %   minMAE: 
    %   energiaSenalOriginal:  Vector - Energía de la señal original
    %   energiaResiduo: Vector - Energía de los residuos
    %   senalE: Señal Residuo
    %   selPred: Mejor Predictor para cada trama (según criterio mae del residuo)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Para recorrer las tramas de muestras
    
    n1 = 1;
    n2 = L;
    n_tramas = ceil(length(ye)/L);

    energiaSenalOriginal = zeros(n_tramas, 1);
    energiaResiduo = zeros(n_tramas, 1); 
    selPred = zeros(n_tramas, 1); 
    senalE = ye*0;

    for i = 1:n_tramas

        % Corta la siguiente trama    
        xf = ye(n1:min(n2, length(ye)));

        % Para esta trama, calcula los predictores lineales
        [Ei, mae_ei, ma_xfi] = lp(xf);

        % Seleciona el mejor predictor basado en el mae_i del residuo
        [minMAE, idx] = min(mae_ei);
        selPred(i) = idx;                   % idx = Mejor Predictor

        % Calcula la energía del mejor residuo y de la señal y almacenadas
        energiaSenalOriginal(i) = (xf)'*(xf)/L;
        energiaResiduo(i) = (Ei(:,idx))' * (Ei(:,idx)) / L;

        % Se puede concatenar las muestras del mejor residuo
        senalE(n1:min(n2, length(ye))) = Ei(:,idx); % Almacena la señal residuo
        n1 = n1 + L;
        n2 = n2 + L;

    end
end
