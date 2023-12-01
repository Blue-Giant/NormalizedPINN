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

data2solus = load('test_solus.mat');
solu2fdm = double(data2solus.Utrue);
solu2dnn = double(data2solus.Usin);


[meshX, meshY] = meshgrid(timesteps, depths);
surfX = meshX(:, 1:10:1001);
surfY = meshY(:, 1:10:1001);

t_op1 = surfX(2,:);
z_0op1 = surfY(2,:);
ufdm_op1 = solu2fdm(2,:);
udnn_op1 = solu2dnn(2,:);

t_op2 = surfX(5,:);
z_0op2 = surfY(5,:);
ufdm_op2 = solu2fdm(5,:);
udnn_op2 = solu2dnn(5,:);

t_op3 = surfX(10,:);
z_0op3 = surfY(10,:);
ufdm_op3 = solu2fdm(10,:);
udnn_op3 = solu2dnn(10,:);

t_op4 = surfX(20,:);
z_0op4 = surfY(20,:);
ufdm_op4 = solu2fdm(20,:);
udnn_op4 = solu2dnn(20,:);

t_op5 = surfX(50,:);
z_0op5 = surfY(50,:);
ufdm_op5 = solu2fdm(50,:);
udnn_op5 = solu2dnn(50,:);

t_op6 = surfX(70,:);
z_0op6 = surfY(70,:);
ufdm_op6 = solu2fdm(70,:);
udnn_op6 = solu2dnn(70,:);

% 可视化结果
figure('name','Solus')
set(gcf,'position',[0 0 1000 750])
tsubplot(6,1,1, 'tight')
plot(t_op1,ufdm_op1, 'b-', 'linewidth', 2);
hold on
plot(t_op1,udnn_op1, 'r-.','linewidth', 3);
hold on
xlim([0,1800])
ylim([-0.01,1])
ylabel('C [g/L]');
xlabel('Time [s]');
title('FDM and T-NPINN Comparsion at Different Depths', 'Fontsize', 18)
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 12);
set(gcf, 'Renderer', 'zbuffer');
hold on

tsubplot(6,1,2, 'tight')
% subplot('position', [0.15, 0.65, 0.75, 0.15])
% hold on
plot(t_op2,ufdm_op2, 'b-', 'linewidth', 2);
hold on
plot(t_op2,udnn_op2, 'r-.','linewidth', 3);
hold on
xlim([0,1800])
ylim([-0.01,1])
ylabel('C [g/L]');
xlabel('Time [s]');
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 12);
hold on

tsubplot(6,1,3, 'tight')
plot(t_op3,ufdm_op3, 'b-', 'linewidth', 2);
hold on
plot(t_op3,udnn_op3, 'r-.','linewidth', 3);
hold on
xlim([0,1800])
ylim([-0.01,1])
ylabel('C [g/L]');
xlabel('Time [s]');
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 12);
hold on

tsubplot(6,1,4, 'tight')
plot(t_op4,ufdm_op4, 'b-', 'linewidth', 2);
hold on
plot(t_op4,udnn_op4, 'r-.','linewidth', 3);
hold on
xlim([0,1800])
ylim([-0.01,1])
ylabel('C [g/L]');
xlabel('Time [s]');
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 12);
hold on

tsubplot(6,1,5, 'tight')
plot(t_op5,ufdm_op5, 'b-', 'linewidth', 2);
hold on
plot(t_op5,udnn_op5, 'r-.','linewidth', 3);
hold on
xlim([0,1800])
ylim([-0.01,1])
ylabel('C [g/L]');
xlabel('Time [s]');
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 12);
hold on

tsubplot(6,1,6, 'tight')
plot(t_op6,ufdm_op6, 'b-', 'linewidth', 2);
hold on
plot(t_op6,udnn_op6, 'r-.','linewidth', 2.5);
hold on
xlim([0,1800])
ylim([-0.01,1])
ylabel('C [g/L]');
xlabel('Time [s]');
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 12);
hold on

text(800, 8.7, '$z=0.1$m','Fontsize', 15, 'Interpreter', 'latex')
text(800, 7.0, '$z=0.4$m','Fontsize', 15, 'Interpreter', 'latex')
text(800, 5.7, '$z=0.9$m','Fontsize', 15, 'Interpreter', 'latex')
text(800, 3.9, '$z=1.9$m','Fontsize', 15, 'Interpreter', 'latex')
text(800, 2.25, '$z=4.9$m','Fontsize', 15, 'Interpreter', 'latex')
text(800, 0.45, '$z=6.9$m','Fontsize', 15, 'Interpreter', 'latex')

lgd0=legend({'FDM','T-NPINN'},'orientation','horizontal','location','North');
set(lgd0,'FontSize',13);
lgd0.Position = [0.6  0.8  0.15  0.1];
legend boxoff;