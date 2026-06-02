function [Pbeamscan] = beamscan(fc,c,M,Rxx)

derad=pi/180;
dr=c/fc/2;
angle=-90:0.1:90;
for m = 1:length(angle)
%     angle(m)=((m-1)/(num/180)-90);
% Rxx=cov(sig);
    phim=derad*angle(m);
Aq = exp(1i*2*pi*fc/c*sin(phim)*dr*(0:M-1)');
Pbeamscan(m)=real(sum((Aq'*Rxx).*Aq.',2));


end
end
