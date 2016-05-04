function [A]=nanmedfilt(A,n)

end_1=size(A,2);
end_2=size(A,1);

for i=2:(end_1-1)
       for k=2:(end_2-1)
           B=[A(k-1,i-1) A(k-1,i) A(k-1,i+1);...
            A(k,i-1) A(k,i) A(k,i+1);A(k+1,i-1) A(k+1,i) A(k+1,i-1)];
           if sum(isnan(B(:)))>=n
               A(k,i)=NaN;
           else
               A(k,i)=nanmedian(nanmedian(B));
           end

       end
end
end