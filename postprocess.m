% Variablen Deklaration
sol=cell(1,numel(peak));
for k=1:numel(peak)     %Zelle zum speichern der gemeinsamen peaks
   sol{k}=zeros(31,39); 
end
dt=[0.3 0.5 1];
a=545/40; %[px/mm] Breite Lüfter Kante in px durch 40 mm
uvm=cell(1,25); %u Geschw. für gemeinsame Strömung 300us mit 1ms 
vvm=cell(1,25); %v Geschw. für gemeinsame Strömung 300us mit 1ms
uv=cell(1,25); %u 300us
vv=cell(1,25); %v 300us
uv1=cell(1,25); %u 1ms
vv1=cell(1,25); %v 1ms
% Normieren nach auf Geschwindigkeit m/s
for i=1:numel(peak)
   uv{i}=uvec{i}./dt(1)/a;
   vv{i}=vvec{i}./dt(1)/a;
   uv1{i}=uvec3{i}./dt(3)/a;
   vv1{i}=vvec3{i}./dt(3)/a;
end
%Betrag vom Vektorfeld
for l=1:numel(peak)
for k=1:size(peak{1},1) 
    for i=1:size(peak{1},2)
        tot1{l}(k,i)=sqrt(uv{l}(k,i).^2+vv{l}(k,i).^2);
        tot3{l}(k,i)=sqrt(uv1{l}(k,i).^2+vv1{l}(k,i).^2);
    end 
end
end
 %Zusammenfügen 1ms mit 300us Strömung
for b=1:numel(peak) 
for i=1:size(peak{1},1)
    for l=1:size(peak{1,1},2)
         if ((peak{b}(i,l) >= peak3{b}(i,l))&&((tot1{b}(i,l)>=tot3{b}(i,l))||(tot3{b}(i,l)>=0.5)))||isnan(tot3{b}(i,l))
             sol{b}(i,l)=peak{b}(i,l);
         else
             sol{b}(i,l)=peak3{b}(i,l);
         end
         if sol{b}(i,l)==peak{b}(i,l)
             uvm{b}(i,l)=uv{b}(i,l);
             vvm{b}(i,l)=vv{b}(i,l);
         else
             uvm{b}(i,l)=uv1{b}(i,l);
             vvm{b}(i,l)=vv1{b}(i,l);
         end
            totm{b}(i,l)=sqrt(uvm{b}(i,l)^2+vvm{b}(i,l)^2);
            phm{b}(i,l)=atan2(vvm{b}(i,l),uvm{b}(i,l));
    end
  
end
end
figure('units','normalized','outerposition',[0 0 1 1])
for i=1:numel(peak)
    subplot(121)
    peak3{i}(isnan(peak3{i}))=0;
    surf(sol{i}-peak3{i})
    title('Differenz zwischen gemeinsamen Peaks und Peaks 1ms')
    subplot(122)
    C = categorical(sol{i}-peak3{i},[0],{'yes'});
    D = categorical(isundefined(C),[0 1],{'yes','no'});
    hist(D(:))
    ylim([0 850])
    title('Anzahl Peaks von 1ms')
%     subplot(133)
%     surf(isundefined(C))
%     view(2)
%     F2(i)=getframe;
end
% matlab2tikz('width','0.8\textwidth','Peakmerge.tex')
% %Filter
% [uvm,vvm,totm,phm]=velfilt(uvm,vvm);
% %Logscalierung zum Sehen der Aussenströmung
% for l=1:numel(peak)
%     totm{l}=sign(totm{l}).*abs(log(abs(totm{l})));
% for k=1:size(peak{1},1) 
%     for i=1:size(peak{1},2)
%         uvm{l}(k,i)=totm{l}(k,i).*cos(phm{l}(k,i));
%         vvm{l}(k,i)=totm{l}(k,i).*sin(phm{l}(k,i));
%     end
% end
% end
%Mittel über einen Zyklus
for k=1:numel(peak)
%     uvm{k}(isnan(uvm{k}))=0;
%     vvm{k}(isnan(vvm{k}))=0;
    if k==1
    w=uvm{1};
    q=vvm{1};
    else
    w=w+uvm{k};
    q=q+vvm{k};
    end
end
w=w/numel(peak);    
q=q/numel(peak);
ab=sqrt(w.^2+q.^2);
avel=median(ab(15,:));
figure('units','normalized','outerposition',[0 0 1 1])

[X,Y] = meshgrid(1:39,1:31);
col_quiver(X,Y,flipud(w),-flipud(q))
title('Arithmetisch gemitteltes Strömungsfeld')
% matlab2tikz('width','0.8\textwidth','Meanquiv.tex')
%Displacement Verteilung im Mittel pro ms
figure('units','normalized','outerposition',[0 0 1 1])
subplot(121)
histogram(w(:)*a,100)
title('Verschiebung px/ms in x-Richtung')
subplot(122)
histogram(q(:)*a,100)
title('Verschiebung px/ms in y-Richtung')
% matlab2tikz('width','0.8\textwidth','Distrib.tex')
figure('units','normalized','outerposition',[0 0 1 1])
contourf(ab)
title('Geschwindigkeitsbetrag arithmetrisch gemittelt')
colorbar
colormap(jet)
% matlab2tikz('width','0.8\textwidth','Meancon.tex')
figure('units','normalized','outerposition',[0 0 1 1])
x=xy_grid(1,end,1);
y=xy_grid(end,1,2);
X=(xy_grid(:,:,1));
Y=(xy_grid(:,:,2));
for i=1:numel(peak)
   md1(i)=(median(tot1{i}(15,:),2));
   quiver(X,Y,flipud(uvm{i}),-flipud(vvm{i}))
   axis([1 x 1 y])
   F1(i)=getframe;
end
t=0:0.0067/25:0.0067;
s=-0.035*cos(2*pi*150*(t+0.005))+1.1;
plot(s)
hold on
plot(smooth(smooth(smooth(md1))))
title('Median Geschwindigkeit in der Mitte vs Schwingung')
figure('units','normalized','outerposition',[0 0 1 1])
% matlab2tikz('width','0.8\textwidth','MeanvsSinus.tex')
% movie(F,4,5)
