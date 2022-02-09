function im_dec=decodificador(im_cod,H,W,escala)
fprintf('Decoding the image\n')
[c_dec] = dentropia(im_cod, H, W);
im_dec=dec_dctQ(c_dec,escala);

