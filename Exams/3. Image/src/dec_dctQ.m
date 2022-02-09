function im_d=dec_dctQ(im_dctq,Eq)
%dec_dctQ Función que decodifica una imagen que ha sido transformada (DCT) y
% cuantificada linealmente con una escala e cuantificación Eq>0
%[im_d]=dec_dctQ(im_dctQ,Eq) 
% Inputs:   
%       im_dctQ: DCT de la imagen cuantificada linealmente con escalón q
%       Eq: escala de cuantificación
% Outputs:
%       im_d: imagen decodificada

    if(Eq<0)
       error('El escalón de cuantificación debe ser mayor que 0.')
    end
    q=qjpeg(Eq);
    im_d=zeros(size(im_dctq));
    [H W]=size(im_dctq);
    Bx=W/8;
    By=H/8;
    for i=1:1:By
        for j=1:1:Bx
            im_rq=im_dctq((i-1)*8+1:i*8,(j-1)*8+1:j*8).*q;
            im_d((i-1)*8+1:i*8,(j-1)*8+1:j*8)=idct2(im_rq);
        end
    end
im_d=im_d+128;