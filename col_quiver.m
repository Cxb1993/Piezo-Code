function col_quiver(x,y,u,v,s,maxuv)

% Arrow head parameters
alpha = 0.33; % Size of arrow head relative to the length of the vector
beta = 0.33;  % Width of the base of the arrow head relative to the length
autoscale = 1; % Autoscale if ~= 0 then scale by this.

if nargin > 4 & ~isempty(s) 
  autoscale = s;
end

% Scalar expand u,v
if prod(size(u))==1, u = u(ones(size(x))); end
if prod(size(v))==1, v = v(ones(size(u))); end

% make colormap (here: 64 level "jet" scheme)
cmap(1:64,:)= colormap(jet(64));
if nargin < 6 | isempty(maxuv)
    maxuv= sqrt(max(u(:).^2+v(:).^2));
end

% Base autoscale value on average spacing in the x and y
% directions.  Estimate number of points in each direction as
% either the size of the input arrays or the effective square
% spacing if x and y are vectors.
delx = diff([min(x(:)) max(x(:))])/max(size(x(:)));
dely = diff([min(y(:)) max(y(:))])/max(size(y(:)));
del = delx.^2 + dely.^2;
if del>0
    len = sqrt((u.^2 + v.^2)/del);
    maxlen = max(len(:));
else
    maxlen = 0;
end
up = u*autoscale; vp = v*autoscale;

ax = newplot;
next = lower(get(ax,'NextPlot'));
hold_state = ishold;

% Make velocity vectors
x = x(:).'; y = y(:).';
up = up(:).'; vp = vp(:).';
uu = [x;x+up;repmat(NaN,size(up))];
vv = [y;y+vp;repmat(NaN,size(vp))];

% compute color index for each vector
colind= min(64,max(1,floor(64 * sqrt(u.^2+v.^2)./(maxuv*1.0001))+1));

% plot arrow stems
for k=1:size(up,2)
    if k == 2, hold on; end
    if isfinite (colind(k))
        plot(uu(:,k),vv(:,k),'Color',cmap(colind(k),:));
    end
end
hold off

% Make arrow heads and plot them
hu = [x+up-alpha*(up+beta*(vp+eps));x+up; ...
    x+up-alpha*(up-beta*(vp+eps));repmat(NaN,size(up))];
hv = [y+vp-alpha*(vp-beta*(up+eps));y+vp; ...
    y+vp-alpha*(vp+beta*(up+eps));repmat(NaN,size(vp))];
hold on
for k=1:size(up,2)
    if isfinite (colind(k))
        plot(hu(:,k),hv(:,k),'Color',cmap(colind(k),:));
    end
end
hold off

box on;
tickstep= 10^round(log10(maxuv/10));
nticks= maxuv / tickstep;
while nticks > 10
    tickstep= tickstep * 2;
    nticks= maxuv / tickstep;
end
while nticks < 5
    tickstep= tickstep / 2;
    nticks= maxuv / tickstep;
end
    
tickval= 0;
i= 0;
while tickval <= maxuv
    i= i + 1;
    tickpos(i)= tickval/maxuv;
    string{i}= sprintf('%6.1f',tickval);
    tickval= tickval + tickstep;
end
colorbar('YTick',tickpos,'YTickLabel',string);

if ~hold_state, hold off, view(2); set(ax,'NextPlot',next); end

