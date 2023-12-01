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
        % 中心差分求解
%         ds = 0.001 * depths(j) / 10; % ds = 0.001*z/10
        ds = 0.001; % ds = 0.001*(10-z)
        dcdz = (c(j+1, i-1) - c(j-1, i-1)) / (2 * dz);
        d2cdz2 = (c(j+1, i-1) - 2 * c(j, i-1) + c(j-1, i-1)) / dz^2;
        
        % 更新 c(z, t) 值
        c(j, i) = c(j, i-1) + dt * (ds * d2cdz2 - ws * dcdz);
    end
end

% 可视化结果
[X, Y] = meshgrid(timesteps, depths);
mesh(Y, X, c);
ylabel('Time (t)');
xlabel('Depth (z)');
zlabel('Concentration (c)');
title('Convection-Diffusion Equation Numerical Solution');
x = X(:,[1:10:1001]); %这个是时间项
y = Y(:,[1:10:1001]); %这个是空间项
fdm = c(:,[1:10:1001]); % 这个是浓度


random = num2str(24259);

data2points = load('-mat',[file_path2,random,'/testData2XY.mat']);
data2solus = load('-mat',[file_path2,random,'/test_solus.mat']);
Points2XY = double(data2points.Points2XY);
M=101;
N=101;
z = double(data2solus.Usin);
true = double(data2solus.Utrue);
a=Points2XY(:,1);
b=Points2XY(:,2);
x = reshape(a,[M,N]);
y = reshape(b,[M,N]);
z = reshape(max(z,0),[M,N]);
true(1)=1;
z(1)=1;
true = reshape(true,[M,N]);
% map = addcolorplus(300);


figure(2)
mesh(x,y,true);
hTitle = title('Analytical Solution');
hXLabel = xlabel('Z (m)');
hYLabel = ylabel('Time (s)');
hZLabel = zlabel('C(z,t)');
% hTitle = title('Step response for w_s= 0.00001m/s, D_s=0.001m^2/s');
% set(hTitle, 'FontSize', 16);
% 坐标区调整
% axis equal
% axis equal
set(gca, 'Box', 'off', ...                                       % 边框
         'XGrid', 'off', 'YGrid', 'off', ...                      % 网格
         'TickDir', 'out', 'TickLength', [0.01 0.01], ...       % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...          % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])           % 坐标轴颜色
view(40.19,23.28);
% colorbar(gca,'Position',[0.13 0.24 0.04 0.18],...
%              'AxisLocation','in',...
%              'TickLength',0.08);
% 字体和字号
set(gca, 'FontName', 'Helvetica')
set([hXLabel, hYLabel, hZLabel], 'FontName', 'AvantGarde')
set(gca, 'FontSize', 18,'FontWeight' , 'bold')
set([hXLabel, hYLabel, hZLabel], 'FontSize', 18,'FontWeight' , 'bold')
set(hTitle, 'FontSize', 18,'FontWeight' , 'bold');
legend off;
% set(hTitle, 'FontSize', 18, 'FontWeight' , 'bold')
% colorbar(gca,'Position',[0.13 0.24 0.04 0.18],...
%              'AxisLocation','in',...
%              'TickLength',0.08);
% 背景颜色
set(gcf,'Color',[1 1 1])

hold on;

figure(3)
mesh(x,y,fdm');
hTitle = title('FDM Solution');
hXLabel = xlabel('Z (m)');
hYLabel = ylabel('Time (s)');
hZLabel = zlabel('C(z,t)');
% 坐标区调整
% axis equal
% axis equal
set(gca, 'Box', 'off', ...                                       % 边框
         'XGrid', 'off', 'YGrid', 'off', ...                      % 网格
         'TickDir', 'out', 'TickLength', [0.01 0.01], ...       % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...          % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])           % 坐标轴颜色
view(40.19,23.28);
% colorbar(gca,'Position',[0.13 0.24 0.04 0.18],...
%              'AxisLocation','in',...
%              'TickLength',0.08);
% 字体和字号
set(gca, 'FontName', 'Helvetica')
set([hXLabel, hYLabel, hZLabel], 'FontName', 'AvantGarde')
set(gca, 'FontSize', 18,'FontWeight' , 'bold')
set([hXLabel, hYLabel, hZLabel], 'FontSize', 18,'FontWeight' , 'bold')
set(hTitle, 'FontSize', 18, 'FontWeight' , 'bold')
% colorbar(gca,'Position',[0.13 0.24 0.04 0.18],...
%              'AxisLocation','in',...
%              'TickLength',0.08);
% 背景颜色
% zlim([0 1]);
set(gcf,'Color',[1 1 1])

tr2 = true.^2;
point_wise_error =true-fdm';
sq = point_wise_error.^2;
format long; % 设置输出格式为10位小数
mse = mean(sq(:));
tre2m = mean(tr2(:));
rel = mse./tre2m;
s=sprintf('MSE is %10f',mse);
d=sprintf('REL is %10f',rel);
disp(s);
disp(d);