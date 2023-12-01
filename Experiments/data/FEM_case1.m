clear all
clc
close all
% ��������
ws = 0.00001;
ds = 0.001;
T = 1800; % ʱ�䷶Χ
Z = 10;   % ��ȷ�Χ
dt = 18;   % ʱ�䲽��
dz = 0.1; % ��Ȳ���

% ��������
timesteps = 0:dt:T;
depths = 0:dz:Z;
nt = length(timesteps);
nz = length(depths);

% ��ʼ�� c ����
c = zeros(nz, nt);
c(:, 1) = 0; % ��ʼ���� c(z, 0) = 0
c(1, :) = 1; % �߽����� c(0, t) = 1
c(end, :) = 0; % �߽����� c(10, t) = 0

% ʹ�����޲�ַ�������ֵ���
for i = 2:nt
    for j = 2:nz-1
        % ���Ĳ�����
        dcdz = (c(j+1, i-1) - c(j-1, i-1)) / (2 * dz);
        d2cdz2 = (c(j+1, i-1) - 2 * c(j, i-1) + c(j-1, i-1)) / dz^2;
        
        % ���� c(z, t) ֵ
        c(j, i) = c(j, i-1) + dt * (ds * d2cdz2 - ws * dcdz);
    end
end

% ���ӻ����
[X, Y] = meshgrid(timesteps, depths);
% mesh(X, Y, c);
mesh(c);
ylabel('Time (t)');
xlabel('Depth (z)');
zlabel('Concentration (c)');
title('Convection-Diffusion Equation Numerical Solution');
% save('X_1.mat', "Y");
% save('T_1.mat', 'X');
% save('Z_1.mat', 'c');