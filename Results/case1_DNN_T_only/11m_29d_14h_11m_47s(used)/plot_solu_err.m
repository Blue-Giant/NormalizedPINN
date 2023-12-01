clear all
clc
close all
% ��������
ws = 0.00001;
T = 1800; % ʱ�䷶Χ
Z = 10;   % ��ȷ�Χ
dt = 1.8;   % ʱ�䲽��
dz = 0.1; % ��Ȳ���

% ��������
timesteps = 0:dt:T;
depths = 0:dz:Z;
nt = length(timesteps);
nz = length(depths);

data2solus = load('test_solus.mat');
solu2fdm = double(data2solus.Utrue);
solu2dnn = double(data2solus.Usin);


[meshX, meshY] = meshgrid(timesteps, depths);
surfX = meshX(:, 1:10:1001);
surfY = meshY(:, 1:10:1001);


% ���ӻ����
figure('name','DNN_Solu')
% surf(surfY, surfX, solu2dnn');
mesh(surfY, surfX, solu2dnn');
ylim([0,1800])
xlim([0,10])
zlim([-0.05, 1])
ylabel('Time [s]', 'Interpreter', 'latex');
xlabel('$z$ [m]', 'Interpreter', 'latex');
zlabel('C($z$, $t$)~~~[g/L]', 'Interpreter', 'latex');
% title('S-NPINN Solution');
hold on
% colorbar;
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 15);
set(gcf, 'Renderer', 'zbuffer');
hold on

figure('name','FDM_Solu')
% surf(surfY, surfX, solu2fdm');
mesh(surfY, surfX, solu2fdm');
ylim([0,1800])
xlim([0,10])
ylabel('Time [s]', 'Interpreter', 'latex');
xlabel('$z$ [m]', 'Interpreter', 'latex');
zlabel('C($z$, $t$)~~~[g/L]', 'Interpreter', 'latex');
% title('Analytical Solution');
% text('Interpreter','latex','String','g/L','Position',[0.0, 0.0 , 0.9],'FontSize',14);
hold on
% colorbar;
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 15);
set(gcf, 'Renderer', 'zbuffer');
hold on

abs_err = abs(solu2dnn - solu2fdm);
% figure('name','Absolute Error')
% surf(surfY, surfX, abs_err, 'EdgeColor', 'none');
% xlim([0,10])
% ylim([0,1800])
% ylabel('Time (s)');
% xlabel('z(m)');
% zlabel('C(z,t)');
% title('Absolute Error', 'position', [11,1000]);
% hold on
% cb2 = colorbar();
% set(get(cb2,'title'),'string','g/L','FontSize',16, 'position', [50, 230]);
% caxis([0 0.007])
% hold on
% set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 15);
% set(gcf, 'Renderer', 'zbuffer');
% hold on

figure('name','Absolute Error2')
surf(surfX, surfY, abs_err', 'EdgeColor', 'none');
xlim([0,1800])
ylim([0,10])
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('$z$ [m]', 'Interpreter', 'latex');
zlabel('C($z$, $t$)', 'Interpreter', 'latex');
title('Absolute Error~[g/L]', 'Interpreter', 'latex');
hold on
cb2 = colorbar();
% set(get(cb2,'title'),'string','g/L','FontSize',16, 'position', [50, 230]);
caxis([0 0.007])
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 15);
set(gcf, 'Renderer', 'zbuffer');
hold on

