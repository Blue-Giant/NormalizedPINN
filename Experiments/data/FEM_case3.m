clear all
clc
close all
% 参数设置
ws = 0.00001;
T = 1800; % 时间范围
Z = 10;   % 深度范围
dt = 1.8;   % 时间步长
dz = 0.1; % 深度步长

% 网格设置
timesteps = 0:dt:T;
depths = 0:dz:Z;
nt = length(timesteps);
nz = length(depths);

% 初始化 c 矩阵
c = zeros(nz, nt);
c(:, 1) = 0; % 初始条件 c(z, 0) = 0
c(1, :) = 1; % 边界条件 c(0, t) = 1
c(end, :) = 0; % 边界条件 c(10, t) = 0

% 使用有限差分法进行数值求解
for i = 2:nt
    for j = 2:nz-1
        % 根据深度z计算ds值
        z = depths(j);
%         ds = 0.001 - 0.0009 * (z - 5)^2 / 25;
        ds = 0.001 * (z/5 - 1)^2;
        dds = 0.002/5*(z/5 - 1);
        % 中心差分求解
        dcdz = (c(j+1, i-1) - c(j-1, i-1)) / (2 * dz);
        d2cdz2 = (c(j+1, i-1) - 2 * c(j, i-1) + c(j-1, i-1)) / dz^2;
        
        % 更新 c(z, t) 值
        c(j, i) = c(j, i-1) + dt * (ds * d2cdz2 + dds*dcdz - ws * dcdz);
    end
end

% 可视化结果
[X, Y] = meshgrid(timesteps, depths);
mesh(Y, X, c);
ylabel('Time (t)');
xlabel('Depth (z)');
zlabel('Concentration (c)');
title('Convection-Diffusion Equation Numerical Solution');
x = X(:,(1:10:1001)); %这个是时间项
y = Y(:,(1:10:1001)); %这个是空间项
z = c(:,(1:10:1001)); % 这个是浓度
save('X_3.mat', 'y');
save('T_3.mat', 'x');
save('Z_3.mat', 'z');