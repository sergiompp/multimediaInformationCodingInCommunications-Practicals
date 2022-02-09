%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actividad 1: representación
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

addpath('imagenes');
addpath('programas');
load('imagen2');

% 1. Mostrar imágenes del directorio
figure(1)
imshow(imagen, map);
title('Imagen Original')

res_orig = 255;

% RESOLUCIONES EN AMPLITUD
figure(2)
resolucion = 4;

% Se cuantifica la imagen para obtener diferentes resoluciones en amplitud.
% La imagen es una matriz de 255x3, por lo que se establece un término
% medio. Lo que haya por debajo del término es 0 (blanco) y por encima es
% 255 (negro)
% Imagen a resoluciones (niveles de grises);

imR = imagen / (res_orig/resolucion)-1;
imshow(imR,map);
xlabel(['Resolución aplicada - x',num2str(resolucion)]);    

% RESOLUCION ESPACIAL
figure(3)
escala = 2;

% Se realiza un diezmado de la imagen.
imEscala=imagen(1:escala:end, 1:escala:end);
imshow(imEscala,map);
xlabel(['Escala aplicada - 1:',num2str(escala)]);

Eq = [4,1/2];
[Nbits_original, Nbits_dec, Tasa_compr, MSE_total,~] = DCTCuantEntrop(imagen,map, Eq);
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actividad 2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

addpath('ejemplos_dct');
MSE=zeros(8,1);
for ind = 1:8
    load(sprintf('b%d',ind))
    [img_transformed, MSE(ind)] = Extractor_DCT(imagen, map); % podéis hacer los plots dentro de Extractor
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actividad 3 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

Eq = [0.1, 1, 10, 100];

for ind =1:length(Eq)
    Codificador(nombre_imagen, Eq(ind));
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actividad 4: Codificación entrópica 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

addpath('imagenes');
Eq =1;

for ind=1:3
    load(sprintf('imagen%d',ind))
    figure(ind)
    subplot(2,1,1);
    imshow(imagen, map);
    xlabel('Imagen Original');
    
    [Nbits_original(ind), Nbits_dec(ind), Tasa_compr(ind), MSE_total(ind),imagen_dec] = DCTCuantEntrop(imagen,map, Eq);
    
    subplot(2,1,2);
    imshow(imagen_dec, map);
    xlabel('Imagen Decodificada');
end

%% Funciones
function [Nbits_original, Nbits_dec, Tasa_compr, MSE_total, imagen_dec] = DCTCuantEntrop(imagen,map, Eq)

   
    imagen = double(imagen);

    MSE_total = zeros(1,length(Eq));
    % Proporcion_ceros = zeros(1,length(Eq));
    Nbits_dec = zeros(1,length(Eq));
    Tasa_compr = zeros(1,length(Eq));

    [n, m] = size(imagen);
    Nbits_original = m*n*8;
    %Nº de bloques 8x8 en cada coordenada
    Bx = floor(m/8);
    By = floor(n/8);
    %energia = zeros(Bx,By);
    dct_q_bloques = zeros(size(imagen));
    %Bucle de cuantificacion
    for k = 1:length(Eq)
        Q = qjpeg(Eq(k));
        %Bucle en la coordenada vertical
        for i = 1:By
            %Índices coordenada vertical
            ny1 = (i - 1)*8 + 1;
            ny2 = i*8;
            %Bucle en la coordenada horizontal
            for j = 1:Bx
                %Índices coordenada horizontal
                nx1 = (j - 1)*8 + 1;
                nx2 = j*8;
                bloque = imagen(ny1:ny2 , nx1:nx2);
                %Continúa aplicando la DCT sobre el bloque
                [imagen_dct, ~] = Extractor_DCT(bloque, map);
                
                dct_q_bloques(ny1:ny2, nx1:nx2) = round(imagen_dct./Q);
                
            end
        end

        [y,Nbits_dec(k)] = entropia(dct_q_bloques);
        Tasa_compr(k) = 100*(Nbits_original-Nbits_dec(k))/Nbits_original;

        imagen_dec = decodificador(y,n,m,Eq(k));
        MSE_total(k) = mean(mean((imagen_dec-imagen).^2));
    end
end

function [img_transformed, MSE] = Extractor_DCT(imagenOriginal, map)
    imagenOriginal = double(imagenOriginal);
    % 1. Restar 128 a la imagen (se centra y va entre -128 y 128)
    imagenCentrada = imagenOriginal - 128;
    
    % 2. Calcula la dct2 del bloque
    img_transformed = dct2(imagenCentrada); % img_transformed = dct_coeffsImagen
    
    % 3. Recupera el bloque aplicando idct2 y devuelve el rango de valores
    % al original (sumando 128)
    imagenReconstruida = idct2(img_transformed) + 128;
    
    % 4. Calcula el MSE entre la imagen original y la recuperada
    MSE = mean(mean(imagenOriginal - imagenReconstruida).^2);
    % MSE = immse(imagenReconstruida, img_patch);

    % ¿Qué debería salir?
    
    % ¿Es la DCT una transformada con pérdidas?
    
    % 5. Visualizar la imagen original y su transformada dct
%     figure
%     subplot(1,2,1);
%     imshow(imagenOriginal,map);
%     xlabel('Imagen Original')
%     subplot(1,2,2);
%     imshow(img_transformed,map);
%     xlabel('Valores Coeficientes asociados a la imagen');
end

function Codificador(nombre_imagen, Eq)
    

    %Aquí  se hace la lectura de la imagen, que se guarda en img
    img = nombre_imagen;
    Q = qjpeg(Eq);
        
    %Calculamos el tamaño de la imagen
    [H, W]=size(img);
    %N de bloques 8x8 en cada coordenada
    Bx=floor(W/8);
    By=floor(H/8);

    for i=1:1:By
        %Índices coordenada vertical
        ny1=(i-1)*8+1;
        ny2=i*8;
        %Bucle en la coordenada horizontal
        for j=1:1:Bx
            %Índices coordenada horizontal
            nx1=(j-1)*8+1;
            nx2= j*8;
            bloque=img(ny1:ny2,nx1:nx2);
            %Continúa aplicando la dct sobre el bloque
            [img_transformed, ~] = Extractor_DCT(bloque, map);
            
            dct_bloques_q(ny1:ny2,nx1:nx2) = round(img_transformed./Q);
           
        end
    end
    
    imagen_dec = dec_dctQ(dct_bloques_q, Eq);
    MSE_q= mean(mean((imagen_dec-nombre_imagen).^2)); 
end
