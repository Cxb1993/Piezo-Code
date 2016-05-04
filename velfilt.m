function [u,v,a,b]=velfilt(u,v)
%a ist Betrag vom Vektorfeld als Zelle
%b ist die Phase vom Vektorfeld als Zelle
%n ist Anzahl Zellelemente
n=numel(u(1,1,:));
end_1=size(u(:,:,1),2);
end_2=size(v(:,:,1),1);

for p=1:n
    a(:,:,p)=sqrt(u(:,:,p).^2+v(:,:,p).^2);
    b(:,:,p)=atan2(v(:,:,p),u(:,:,p));
end
 for l=1:n
   for i=2:(end_1-1)
       for k=2:(end_2-1)

            a(k,i,l)=nanmedian(nanmedian([a(k-1,i-1,l) a(k-1,i,l) a(k-1,i+1,l);...
            a(k,i-1,l) a(k,i,l) a(k,i+1,l);a(k+1,i-1,l) a(k+1,i,l) a(k+1,i-1,l)]));

            
            b(k,i,l)=nanmedian(nanmedian([b(k-1,i-1,l) b(k-1,i,l) b(k-1,i+1,l);...
            b(k,i-1,l) b(k,i,l) b(k,i+1,l);b(k+1,i-1,l) b(k+1,i,l) b(k+1,i-1,l)]));
        
            u(k,i,l)=a(k,i,l).*cos(b(k,i,l));
            v(k,i,l)=a(k,i,l).*sin(b(k,i,l));
       end
   end
end

end