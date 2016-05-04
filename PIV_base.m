function [xgrid,ygrid,uvecs,vvecs,peaks,valid,cmaps] = PIV_base (I1,I2,mode,wxy,sxy,oxy,gxy,avgmaps,threshold)

%   Last Update: 29-NOW-09

%   ------ List of Changes ------
% [xgrid,ygrid,uvecs,vvecs,peaks,valid,cmaps] = PIV_base (Amean,Bmean,1,[62,62],[15,15],[0,0],[16,16],[],0);
%   -----------------------------

%   PIV 2-component, 2-D Particle Image Velocimetry
%
%   I1,I2:          images (no stacked slices allowed - use frame averaging to combine data)
%
%   mode:           0  = PIV with SINC subpixel interpolation
%                   1  = PIV with GAUSS subpixel interpolation
%                   -1 = PIV WITHOUT subpixel interpolation
%
%   wxy:            [wx,wy]     interrogation window size
%   sxy:            [sx,sy]     max search size in x and y 
%   oxy:            [ox,oy]     window offset (constant across image) 
%                   [nx,ny,2]	window offset for individual grid points
%   gxy:            [gx,gy]     window shift
%                   [nx,ny,2]	window positions in x and y
%
%   avgmaps:        composite image of correlation maps to be averaged with
%                   current data (leave empty if not needed)
%
%   xgrid,ygrid:    grid location of output results
%   uvecs, vvecs:   (map averaged) u- and v- velocities 
%   peaks:          map of peak heights 
%   valid:          map of flags: -1=invalid, 0=integer shifts ok, 1=subpixel shifts ok
%   cmaps:          output composite image of correlation maps to be averaged with
%                   subsequent data

%----------------------- global variables -----------------------------------

clear global CSZX CSZY
clear global QMAT QQ
clear global PHASEX PHASEY PHASEXX PHASEXY PHASEYY
clear global MAXITER

global CSZX CSZY
CSZX= 2*sxy(1) + 1;                     % x-size of correlation map, always odd
CSZY= 2*sxy(2) + 1;                     % y-size of correlation map, always odd

% matrix for Gaussian peak fit
global QMAT QQ
QMAT= [ [ 1 -1 -1  1  1  1]; ...
        [ 1  0 -1  0  0  1]; ...
        [ 1  1 -1  1 -1  1]; ...
        [ 1 -1  0  1  0  0]; ...
        [ 1  0  0  0  0  0]; ...
        [ 1  1  0  1  0  0]; ...
        [ 1 -1  1  1 -1  1]; ...
        [ 1  0  1  0  0  1]; ...
        [ 1  1  1  1  1  1] ];
QQ= QMAT' * QMAT;

% compute phase mask for FFT shifts
global PHASEX PHASEY PHASEXX PHASEXY PHASEYY
PHASEX= zeros(1,CSZX);
PHASEX(1:sxy(1)+1)= 2*pi*(0:sxy(1))./CSZX;
PHASEX(sxy(1)+2:CSZX)= 2*pi*(-sxy(1):-1)./CSZX;
PHASEX= repmat(PHASEX,CSZY,1);

PHASEY= zeros(1,CSZY);
PHASEY(1:sxy(2)+1)= 2*pi*(0:sxy(2))./CSZY;
PHASEY(sxy(2)+2:CSZY)= 2*pi*(-sxy(2):-1)./CSZY;
PHASEY= repmat(PHASEY',1,CSZX);

PHASEXX= PHASEX.*PHASEX;
PHASEXY= PHASEX.*PHASEY;
PHASEYY= PHASEY.*PHASEY;

global MAXITER
MAXITER= 20;

%-----------------------------------------------------------------------------

if size(I1) ~= size(I2)
    error ('Images should be same size ...');
end

if abs(mode) > 1
    error ('Select proper subpixel processing mode: 0=Sinc, 1=Gauss, -1=None ...');
end

if wxy(1) < 1 || wxy(2) < 1
    error ('Wrong window size wxy ...');
end

if sxy(1) < 1 || sxy(2) < 1
    error ('Search range sxy too small ...');
end

if 2*sxy(1) >= wxy(1) || 2*sxy(2) >= wxy(2)
    error ('Search range sxy too large ...');
end

if gxy(1) < 1 || gxy(2) < 1
    error ('Wrong window spacing dxy ...');
end

[height,width,nslice]=size(I1);

if nslice > 1
    disp ('Only processing first slice of image stack !');
end

% make sure image data are floating point
I1=double(I1);
I2=double(I2);

% fill in arrays of interrogation grid points
if numel(gxy) ==2
    % grid spacing is prescribed
    nx= floor((width-wxy(1))/gxy(1)) + 1;
    ny= floor((height-wxy(2))/gxy(2)) + 1;
    [xgrid,ygrid]= meshgrid((0:nx-1)*gxy(1)+1,(0:ny-1)*gxy(2)+1);
    xgrid= xgrid + wxy(1)/2 - 0.5;
    ygrid= ygrid + wxy(2)/2 - 0.5;
else
    % full grid is prescribed
    [ny,nx,nz]= size(gxy);
    xgrid= gxy(:,:,1);
    ygrid= gxy(:,:,2);
end

% size of smaller interrogation window
wszx= wxy(1)-CSZX+1;
wszy= wxy(2)-CSZY+1;
norm= wszx*wszy;
            
% initialize output arrays
uvecs= NaN(ny,nx);
vvecs= NaN(ny,nx);
peaks= NaN(ny,nx);
valid= -int8(ones(ny,nx));
cmaps= zeros(ny*CSZY,nx*CSZX);

% compute edge coordinates for PIV grid
if  numel(oxy) == 2
    ox2= repmat(round(oxy(1)/2),ny,nx);
    oy2= repmat(round(oxy(2)/2),ny,nx);
elseif  numel(oxy) == numel(gxy)
    ox2= round(oxy(:,:,1)/2);
    oy2= round(oxy(:,:,2)/2);
else
    error ('Incompatible specification for oxy ...');
end

% precompute sliding means and variances (is faster than for individual
% subwindows)
sumAn= slidesum(I1,wxy(2)-CSZY+1,wxy(1)-CSZX+1);
varAn= slidesum(I1.^2,wxy(2)-CSZY+1,wxy(1)-CSZX+1) - (sumAn.^2)./norm;
sumBn= slidesum(I2,wszy,wszx);
varBn= slidesum(I2.^2,wszy,wszx) - (sumBn.^2)./norm;

% loop over all subwindows
for rcindx=1:nx*ny

    try

        % Extract subwindows from the images
        xoff= xgrid(rcindx) - wxy(1)/2 + 0.5;
        yoff= ygrid(rcindx) - wxy(2)/2 + 0.5;

        % extract smaller subwindow from first image
        x1= xoff-ox2(rcindx)+sxy(1);
        y1= yoff-oy2(rcindx)+sxy(2);
        A= I1(y1:y1+wxy(2)-CSZY,x1:x1+wxy(1)-CSZX);

        % extract full subwindow from second image
        x2= xoff+ox2(rcindx);
        y2= yoff+oy2(rcindx);
        B= I2(y2:y2+wxy(2)-1,x2:x2+wxy(1)-1);

        % insert previous map estimate if available
        rr= mod(rcindx-1,ny)*CSZY;
        cc= fix((rcindx-1)/ny)*CSZX;
        if not(isempty(avgmaps))
            newmap= avgmaps(rr+1:rr+CSZY,cc+1:cc+CSZX);
        else
            newmap= zeros(CSZY,CSZX);
        end

        % compute normalizing denominator
        denom= sqrt(varAn(y1,x1) .* varBn(y2:y2+CSZY-1,x2:x2+CSZX-1));

        if denom ~= 0

            % compute covariance for windows
            fft_A= fft2(A,wxy(2),wxy(1));   % padded for size of subwindow B
            fft_B= fft2(B);
            corrAB= ifft2(conj(fft_A).*fft_B,'symmetric');

            % compute (co-)variances
            covAB= corrAB(1:CSZY,1:CSZX) - sumAn(y1,x1).*sumBn(y2:y2+CSZY-1,x2:x2+CSZX-1)./norm;

            % compute correlation map
            normcorr= covAB ./ denom;
            if max(normcorr(:)) >= threshold
                newmap= newmap + normcorr;
            end

        end

        % store updated correlation map
        cmaps(rr+1:rr+CSZY,cc+1:cc+CSZX)= newmap;

        % peak analysis only if there are non-zero elements in cmaps
        if newmap ~= 0

            % Find value and position (xpeak, ypeak) of the correlation peak
            peaks(rcindx)= max(newmap(:));
            [ypeak,xpeak]= find(newmap==peaks(rcindx));

            % analyze further only if peak value is unique                
            if numel(xpeak) == 1
 
                % compute shift index
                uvecs(rcindx)= xpeak - sxy(1) - 1 + 2*ox2(rcindx);
                vvecs(rcindx)= ypeak - sxy(2) - 1 + 2*oy2(rcindx);
                valid(rcindx)= 0;
                
                % correlation maximum is not an a edge
                if mode >= 0 && xpeak~=1 && xpeak~=CSZX && ypeak~=1 && ypeak~=CSZY
                    % subpixel interpolation
                    if mode == 0
                        [intpeak,shift,iter]= sincpeak (newmap,[xpeak,ypeak]);
                    else
                        [intpeak,shift,iter]= gausspeak (newmap,[xpeak,ypeak]);
                    end
                    if iter < MAXITER && intpeak/peaks(rcindx) > 0.98 && intpeak/peaks(rcindx) < 2.0
                        % good data: final interpolated shifts superseedes integer shift estimate
                        uvecs(rcindx)= shift(1) - sxy(1) - 1 + 2*ox2(rcindx);
                        vvecs(rcindx)= shift(2) - sxy(2) - 1 + 2*oy2(rcindx);
                        peaks(rcindx)= intpeak;
                        valid(rcindx)= 1;
                    end
                end
            end
        end

    catch
        % array indexing error (window out of bounds): do nothing, move on
    end

end

return

%----------------------------------------------------------------------------------------

function ss= slidesum(a,n,m)
% sliding window summation; averaging window size is [n,m]
[na,ma]= size(a);
aa= [zeros(1,ma+1); [zeros(na,1),a]];
a1= cumsum(aa,1);
a2 = a1(n+1:na+1,:)-a1(1:na+1-n,:);
a3 = cumsum(a2,2);
ss = a3(:,m+1:ma+1)-a3(:,1:ma+1-m);
return

%----------------------------------------------------------------------------------------

function [peak,shift,it]= gausspeak (cmap,start)

% (pseudo-)Gaussian peak fit: p(x,y) = exp(a + b*x + c*y + d*x*x + e*x*y + f*y*y)

% last modified 23-02-03

global QMAT QQ MAXITER

peak= 0;
shift= [0,0];

ind= 1;
for j=-1:1
    for i=-1:1
        rhs(ind)= log(max(cmap(start(2)+j,start(1)+i),1.e-12));
        ind= ind + 1;
    end
end
if var(rhs) == 0
    % right hand side was clipped everywhere: no good
    it= MAXITER;
    return
end
qr= QMAT' * rhs';
coeffs= QQ \ qr;
mmat= [ [2*coeffs(4) coeffs(5)]; [coeffs(5) 2*coeffs(6)]];
mrhs= [-coeffs(2); -coeffs(3)];
qvec= (mmat \ mrhs)';

if norm(qvec) > 1
    % interpolated displacement is too large: no good
    it= MAXITER;
    return
end

it= 1;
shift= qvec + start;
    
% compute surface curvature criterion for Gaussian peak
peak= exp(coeffs(1) + coeffs(2)*qvec(1) + coeffs(3)*qvec(2) + ...
          coeffs(4)*qvec(1)*qvec(1) + coeffs(5)*qvec(1)*qvec(2) + coeffs(6)*qvec(2)*qvec(2));

return

%----------------------------------------------------------------------------------------

function [peak,shift,it]= sincpeak (cmap,start)

% FFT based optimum (sinc) interpolation   

% last modified 23-02-03

global CSZX CSZY
global PHASEX PHASEY PHASEXX PHASEXY PHASEYY
global MAXITER

% must subtract [1,1] to work with Fourier shift indices
fcorr= fft2(cmap);
low= start - 1 - 0.75;
high= start - 1 + 0.75;

% Newton iteration to find subpixel shift
new= [1,1];
shift= start - 1;
it=0;
while norm(new) > 1.e-6 && it < MAXITER
    it= it + 1;
    [peak, grad, hess]= fgheval (shift,CSZX,CSZY,fcorr,PHASEX,PHASEY,PHASEXX,PHASEXY,PHASEYY);
    % compute determinant of Hessian (local curvature)
    det= hess(1,1)*hess(2,2)-hess(1,2)*hess(2,1);
    % compute inverse Hessian (determinant inserted later)
    ihess(1,1)= hess(2,2);
    ihess(1,2)= -hess(1,2);
    ihess(2,1)= -hess(1,2);
    ihess(2,2)= hess(1,1);
    % update shift estimate (abs(det) used to prevent runaways)
    new= (ihess*(grad'))/abs(det);
    shift= shift - new';
    % limit to acceptable coordinate range
    shift= min(high,max(low,shift));
end        
peak= peak / (CSZY*CSZX);

% correct for Fourier indexing
shift= shift + 1;

return

%----------------------------------------------------------------------------------------

function [f, g, h]= fgheval (shift,nx,ny,fcorr,phasex,phasey,phasexx,phasexy,phaseyy)

% compute x-, y- and total phase factors
totphase= phasey .* repmat(shift(2),ny,nx) + phasex .* repmat(shift(1),ny,nx);

% compute product of spectrum and phase factors
ff= fcorr .* exp(i*totphase);
realf= real(ff);
imagf= imag(ff);

% function evaluation 
f= sum(realf(:));

% evaluate function gradient: vector
g(1)= dot(imagf(:),phasex(:));
g(2)= dot(imagf(:),phasey(:));

% evaluate Hessian matrix
h(1,1)= dot(realf(:),phasexx(:));
h(1,2)= dot(realf(:),phasexy(:));
h(2,1)= h(1,2);
h(2,2)= dot(realf(:),phaseyy(:));

return