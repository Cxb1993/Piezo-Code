% PIV Analyse
clear all;clc
%Eingabe der Files für PIV 
filename={'2*.tif' '500*.tif'};

for i=1:numel(filename)
    files(:,i)=dir(filename{i});
end

anzahl=50; %Anzahl Bilder pro Sweep
n=numel(files(:,1))/anzahl; %Anzahl Mittelungen

for i=1:numel(filename)
    min_image= [];
    for k=10:numel(files(:,1))
        if isempty(min_image)
            min_image=imread(files(k,i).name);
        else
            min_image=min(min_image,imread(files(k,i).name));
        end
    end
    if i==1
        min_image1=min_image;
    else
        min_image2=min_image;
        clear min_image
    end
end

for i=1:numel(filename)
    for k=1:numel(files(:,1))
        if i==1
            image=imread(files(k,i).name)-min_image1;
        else
            image=imread(files(k,i).name)-min_image2;
        end
        a1(:,:,k,i)=image(1:1024,:);
        a2(:,:,k,i)=image(1025:2048,:);
    end
end

for i=1:numel(filename)
    for k=1:numel(files(:,1))
        a1(:,:,k,i)=medfilt2(a1(:,:,k,i));
        a2(:,:,k,i)=medfilt2(a2(:,:,k,i));
    end
end
%PIV_base
for jj=1:numel(filename)
    
    for i=0:(anzahl-1)
        loopdata= [];
        for k=(1+i):anzahl:numel(files(:,1))
            if isempty(loopdata)
                [xy_grid,uv_vecs,peaks,valid,loopdata] = PIV_base_1 (a1(:,:,k,jj),a2(:,:,k,jj),1,[64,64],[16,16],[0,0],[32,32],[],0.5);
            else
                [xy_grid,uv_vecs,peaks,valid,loopdata] = PIV_base_1 (a1(:,:,k,jj),a2(:,:,k,jj),1,[64,64],[16,16],[0,0],[32,32],loopdata,0.5);
            end
        end
        uvec(:,:,i+1,jj)=squeeze(uv_vecs(:,:,1));
        vvec(:,:,i+1,jj)=squeeze(uv_vecs(:,:,2));
        peak(:,:,i+1,jj)=peaks;
        maps(i+1,jj)=struct('cmaps',loopdata.cmaps,'nmaps',loopdata.nmaps);
    end
end
clear i k jj