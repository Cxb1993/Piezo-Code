clear all;clc
file1='s300\analyse300.mat';
% file2='s500\analyse500.mat';
file3='s1\analyse1.mat';


m1=matfile(file1);
% m2=matfile(file2);
m3=matfile(file3);

peak=m1.peak; peak3=m3.peak; %maps1=m1.maps; maps2=m3.maps;
uvec=m1.uvec; uvec3=m3.uvec;
vvec=m1.vvec; vvec3=m3.vvec;
xy_grid=m1.xy_grid;
