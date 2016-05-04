x=squeeze(xy_grid(:,:,1));
y=squeeze(xy_grid(:,:,2));
a=545/40; %[px/mm]
dt=[2 0.5]; %[ms]

clear l i k
%Pixel in Velocity [m/s]
for jj=1:numel(filename)
    for i=1:anzahl
        uv(:,:,i,jj)=uvec(:,:,i,jj)./dt(jj)/a;
        vv(:,:,i,jj)=vvec(:,:,i,jj)./dt(jj)/a;
    end
end
for jj =1:numel(filename)
    for i=1:anzahl
        p(:,:,i,jj)=peak(:,:,i,jj)./maps(i,jj).nmaps;
    end
end
peak(isinf(peak))=NaN;
%Total Displacement
for jj=1:numel(filename)
    for l=1:anzahl
        for i=1:length(uv(1,:,l,jj))
            
            for k=1:length(uv(:,1,l,jj))
                tot(k,i,l,jj)=sqrt(uv(k,i,l,jj).^2+vv(k,i,l,jj).^2);
                phase(k,i,l,jj)=atan2(vv(k,i,l,jj),uv(k,i,l,jj));
            end
            
        end
    end
end
for jj=1:numel(filename)
for i=1:anzahl
   [p(:,:,i,jj)]=nanmedfilt(peak(:,:,i,jj)./maps(i,jj).nmaps,4);
end
end
%Fusion
for b=1:anzahl
    for i=1:size(peak(:,:,1,1),1)
        for l=1:size(peak(:,:,1,1),2)
            if (p(i,l,b,2) >= p(i,l,b,1)&& p(i,l,b,1) <= 0.5) || isnan(p(i,l,b,1)) || isnan(p(i,l,b,2))
                peak(i,l,b,3)=p(i,l,b,2);
            else
                peak(i,l,b,3)=p(i,l,b,1);
            end
            if peak(i,l,b,3)==p(i,l,b,1)
                uv(i,l,b,3)=uv(i,l,b,1);
                vv(i,l,b,3)=vv(i,l,b,1);
            else
                uv(i,l,b,3)=uv(i,l,b,2);
                vv(i,l,b,3)=vv(i,l,b,2);
            end
            
        end
        
    end
    tot(:,:,b,3)=sqrt(uv(:,:,b,3).^2+vv(:,:,b,3).^2);
    phase(:,:,b,3)=atan2(vv(:,:,b,3),uv(:,:,b,3));
end

%Filtering Vectordata
[uvm,vvm,totm,phm]=velfilt(uv(:,:,:,3),vv(:,:,:,3));

figure('units','normalized','outerposition',[0 0 1 1])
subplot(221)
histogram(uvec(:,:,:,1),100)
subplot(222)
histogram(vvec(:,:,:,1),100)
subplot(223)
histogram(uvec(:,:,:,2),100)
subplot(224)
histogram(vvec(:,:,:,2),100)
%Logscalierung zum Sehen der Aussenströmung
for l=1:anzahl
    totl(:,:,l)=sign(tot(:,:,l)).*abs(log(abs(tot(:,:,l))));
for k=1:size(uv(:,:,l),1)
    for i=1:size(uv(:,:,l),2)
        uvm(k,i,l)=totl(k,i,l).*cos(phase(k,i,l));
        vvm(k,i,l)=totl(k,i,l).*sin(phase(k,i,l));
    end
end
end
figure('units','normalized','outerposition',[0 0 1 1])
md=zeros(50,2);
for jj=1:length(uv(1,1,1,:))
    for i=1:anzahl
        if jj<=2
            md(i,jj)=median(nanmedian(tot(14:17,2:6,i,jj),2));
        else
            md(i,jj)=median(nanmedian(totm(14:17,2:6,i),2));
        end
    end
end

% for jj=1:length(uv(1,1,1,:))
    for i=1:anzahl
 %         subplot(121)
%         quiver(x,y,uvm(:,:,i),vvm(:,:,i))
        %           quiver(x,y,uv(:,:,i,jj),vv(:,:,i,jj))
%         axis([0 x(end) 0 y(end)])
%         drawnow
        
%         F1(i)=getframe;
        %     subplot(122)
            contourf((totm(:,:,i)))
        %     drawnow
        %     pause(1/6)
        %     view(2)
        %     colorbar
        %     colormap jet
        %     drawnow
            F2(i)=getframe;
        %     F3(i)=getframe;
    end
% end

figure('units','normalized','outerposition',[0 0 1 1])
% subplot(121)

m1=smooth(smooth(smooth(md(:,1))));
m2=smooth(smooth(smooth(md(:,2))));
m3=smooth(smooth(smooth(md(:,3))));
m4=smooth(smooth(smooth(mean(md,2))));

avel=mean(m1);
sigma=std(m1);
t=0:0.0067/(anzahl-1):0.0067;
s=-sigma*cos(2*pi*150*(t+0.005))+avel;
plot(s)
hold on
plot(1:anzahl,m1)
hold on
plot(1:anzahl,m2)
hold on
plot(1:anzahl,m3)
hold on
plot(1:anzahl,m4)
legend('-cos(t)','2 ms','500 us','Kombi','Average')
xlim([1 anzahl])
% subplot(122)
% matlab2tikz('filename','Schwingung.tex','width','0.8\textwidth')
clear i l k