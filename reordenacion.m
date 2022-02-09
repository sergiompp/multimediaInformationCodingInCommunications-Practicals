function c=reordenacion(w, N, M);
[Nw,Mw]=size(w);
if (Nw~=64)|(rem(Mw,N*M/64)~=0)
    disp('Error');
    return;
end
c=zeros(N,M);
cont=0;
for i=1:8:N
    for j=1:8:M
        cont=cont+1;
        c(i:i+7,j:j+7)=dezz(w(:,cont)');
    end
end