function w=ordenacion(c);
[N,M]=size(c);
if (rem(N,8)~=0)|(rem(M,8)~=0)
    disp('Error');
    return;
end
w=zeros(64,(M*N)/64);
cont=0;
for i=1:8:N
    for j=1:8:M
        cont=cont+1;
        w(:,cont)=zz(c(i:i+7,j:j+7))';
    end
end