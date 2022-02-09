function [c_dec] = dentropia(y, N, M)

%[c_dec] = dentropia(y, N, M)
%Entradas:
%  y     a column vector of non-negative integers (bytes) representing 
%           the code, 0 <= y(i) <= 255. 
% NxM es el numero de pixeles de la imagen
%Salidas:
% c_dec es la matriz decodificada con coeficientes de la transformacion cuantificados

w = JPEGlike(y,64,N*M/64);
c_dec=reordenacion(w, N, M);