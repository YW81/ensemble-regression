clear all; %close all;

addpath ../Ensemble_Regressors/
addpath Datasets/
addpath Datasets/ManualEnsembleDatasets/
dataset_name = 'DifferentRegressors_Friedman1';
%dataset_name = 'RandomForestTest_Friedman1';
%dataset_name = %'flights_BWI'%'SP500.mat'; 'flights_BOS'
load(dataset_name);% y_RidgeCV = y_RidgeCV';
tic;

y_true = y; clear y;
var_y = Ey2 - Ey.^2;
[m,n] = size(Z);

num_alphas = 100;
e=zeros(m,num_alphas); 
w=zeros(m,num_alphas);
%alphas=logspace(log10(1/var_y)-5,log10(1/var_y)+1,num_alphas); %100/var_y
alphas=linspace(0.1,1,num_alphas); 
for i=1:num_alphas; 
    [y_lrm, w_lrm, y_oracle_rho, w_oracle_rho,Cstar] = ...
        ER_LowRankMisfitCVX( Z, Ey, Ey2, y_true, alphas(i)); 
    
    w(:,i) = w_lrm;
    e(:,i) = eig(Cstar); 
    MSEs(i) = mean((y_true' - y_lrm).^2); 
    cnd(i) = cond(Cstar); 
end; 
toc;

%% Plot eigenvalues and eigenvalue ratios
e_ratio = e./repmat(e(end,:),m,1);
figure('Name',[dataset_name ' Rank']);
suptitle(['Oracle \rho MSE/var(Y)=' num2str(mean((y_true' - y_oracle_rho).^2) / var_y)]);
%figure('Name','Eigenvalues and MSE');
subplot(211);
[ax,line1,line2] = plotyy(alphas,e_ratio',alphas,MSEs/var_y);
title('Eigenvalues and MSE vs alpha');

for i=1:length(line1)
    line1(i).Marker = 'x';
end;
line2.Marker = '*';
line2.Color = 'k';
line2.LineWidth = 2;
ylabel(ax(1),'Eigenvalues');
ylabel(ax(2),'MSE / var(Y)');

grid on; xlabel('Alpha');
set(ax(1),'xscale','log');
set(ax(2),'xscale','log');

subplot(212);
[ax,line1,line2] = plotyy(alphas,e',alphas,MSEs/var_y);
title('Eigenvalues and MSE vs alpha');

for i=1:length(line1)
    line1(i).Marker = 'x';
end;
line2.Marker = '*';
line2.Color = 'k';
line2.LineWidth = 2;
ylabel(ax(1),'Eigenvalues');
ylabel(ax(2),'MSE / var(Y)');

grid on; xlabel('Alpha');
set(ax(1),'xscale','log');
set(ax(2),'xscale','log');


%%
figure('Name',dataset_name);
suptitle(['Oracle \rho MSE/var(Y)=' num2str(mean((y_true' - y_oracle_rho).^2) / var_y)]);
%figure('Name','Eigenvalues and MSE');
subplot(221);
[ax,line1,line2] = plotyy(alphas,e',alphas,MSEs/var_y);
title('Eigenvalues and MSE vs alpha');

for i=1:length(line1)
    line1(i).Marker = 'x';
end;
line2.Marker = '*';
line2.Color = 'k';
line2.LineWidth = 2;
ylabel(ax(1),'Eigenvalues');
ylabel(ax(2),'MSE / var(Y)');

grid on; xlabel('Alpha');
set(ax(1),'xscale','log');
set(ax(2),'xscale','log');

%figure('Name','Weights'); 
subplot(222);
plot(alphas,w','x-')
xlabel('Alpha');
ylabel('Weights');
grid on;
hold on; plot(alphas, repmat(w_oracle_rho,1,num_alphas)','k:'); hold off;
title('Weights');
set(gca,'xscale','log');

%figure('Name','MSE and Condition number');
subplot(223);
[ax,line1,line2] = plotyy(alphas,cnd,alphas,MSEs / var_y);
title('Condition number and MSE');
line1.Marker = 'x';
line2.Marker = 'x';
ylabel(ax(1),'Condition Number(Cstar)');
ylabel(ax(2),'MSE / var(Y)');
set(ax(1),'xscale','log');
set(ax(2),'xscale','log');
grid on;

%figure;
subplot(224);
dist = sum(abs((w-(ones(m,num_alphas)/m))));
plot(alphas, dist,'x-',alphas,max(dist)*(MSEs - min(MSEs)) / (max(MSEs) - min(MSEs)),'k.-'); 
set(gca,'xscale','log'); grid on;
xlabel('Alpha'); legend('|w - 1/m|', 'max(dist)*(MSE - min(MSE)) / (min(MSE) - max(MSE))');
% Green Lines: 1/var_y * [.1 .01 .001]
line(.1*ones(1,2)/var_y,get(gca,'YLim'),'Color',[0 0.5 0]);
line(0.01*ones(1,2)/var_y,get(gca,'YLim'),'Color',[0 0.8 0]);
line(0.001*ones(1,2)/var_y,get(gca,'YLim'),'Color',[0 1 0]);
% Purple/Pink Lines: 1/(var_y*m), 1/(var_y*m^2), 1/(var_y*m*2)
line(ones(1,2)/(var_y*m),get(gca,'YLim'),'Color',[0.4 0 0.3]);
line(ones(1,2)/(var_y*m^2),get(gca,'YLim'),'Color',[.7 0.1 0.5]);
line(ones(1,2)/(var_y*m^3),get(gca,'YLim'),'Color',[1 0.3 0.7]);
% Blue Line: 1/order of nuclear_norm(C*) = 1/sqrt(m*M_2^3)
line(ones(1,2)/(sqrt(m*Ey2^3)),get(gca,'YLim'),'Color',[0 0 1]);
title('L1 distance of weights from simple average');
