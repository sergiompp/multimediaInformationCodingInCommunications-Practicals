function [y,Nbits] = entropia(im_dct_q)

%[y_cod,Res] = entropia(dctq)
%Entradas:
% im_dct_q es la matriz con coeficientes de la transformacion cuantificados 
%Salidas:
%  y     a column vector of non-negative integers (bytes) representing 
%           the code, 0 <= y(i) <= 255. 
%  Nbits Number of bits resulting in the compressed image

c=ordenacion(im_dct_q);
[y,Res] = JPEGlike(0,c);
Nbits=sum(Res([2:4,6:8]));