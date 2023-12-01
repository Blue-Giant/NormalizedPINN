clc
clear all
close all

dataErr2case1_DNN = load('test_Err2sin_case1_DNN.mat');
% mse2case1_DNN = sqrt(dataErr2case1_DNN.mse);
% rel2case1_DNN = sqrt(dataErr2case1_DNN.rel);

mse2case1_DNN = (dataErr2case1_DNN.mse);
rel2case1_DNN = (dataErr2case1_DNN.rel);

dataErr2case1_T = load('test_Err2sin_cases1_T.mat');
% mse2case1_T = sqrt(dataErr2case1_T.mse);
% rel2case1_T = sqrt(dataErr2case1_T.rel);
mse2case1_T = (dataErr2case1_T.mse);
rel2case1_T = (dataErr2case1_T.rel);

dataErr2case1_X = load('test_Err2sin_cases1_X.mat');
% mse2case1_X = sqrt(dataErr2case1_X.mse);
% rel2case1_X = sqrt(dataErr2case1_X.rel);
mse2case1_X = (dataErr2case1_X.mse);
rel2case1_X = (dataErr2case1_X.rel);

dataErr2case1_Both = load('test_Err2sin_cases1_both.mat');
% mse2case1_Both = sqrt(dataErr2case1_Both.mse);
% rel2case1_Both = sqrt(dataErr2case1_Both.rel);
mse2case1_Both = (dataErr2case1_Both.mse);
rel2case1_Both = (dataErr2case1_Both.rel);

epoch = 1:51;

figure('name','error')
fig_rel2case1_DNN = plot(epoch,rel2case1_DNN,'r-*', 'linewidth',2);
hold on

bb = [0 0.55 0.8];
fig_rel2case1_T = plot(epoch, rel2case1_T,'-X', 'linewidth',2, 'color', bb);
hold on

fig_rel2case1_X = plot(epoch,rel2case1_X,'m:', 'linewidth',2);
hold on

fig_rel2case1_Both = plot(epoch,rel2case1_Both,'c-V', 'linewidth',2);
hold on

set(gca,'yscale','log')
set(gca, 'Fontsize', 18)
xlabel('epoch/1000', 'Fontsize', 18)
ylabel('REL', 'Fontsize', 18, 'Interpreter', 'latex')
xlim([0,51])
ylim([0.0001, 75])
hold on
% 
lgd0=legend([fig_rel2case1_DNN,fig_rel2case1_T],{'PINN','T-NPINN'},'orientation','horizontal','location','North');
set(lgd0,'FontSize',18.5);
lgd0.Position = [0.3  0.820  0.45  0.2];
legend boxoff;

ah1=axes('position',get(gca,'position'),'visible','off');
lgd1=legend(ah1,[fig_rel2case1_X,fig_rel2case1_Both],{'S-NPINN','ST-NPINN'},'orientation','horizontal','location','North');
set(lgd1,'FontSize',18.5);
lgd1.Position = [0.3  0.77  0.45  0.2];
legend boxoff;