function newim=image_resize(im)

%[newim] = image_resize(im)
%Entradas:
% im imagen original 8x8 
%Salidas:
%  newim Imagen grande a 64x64

newim=zeros(64,64);
for i=1:1:8
    for j=1:1:8
        newim((i-1)*8+1:i*8,(j-1)*8+1:j*8)=im(i,j);
    end
end
