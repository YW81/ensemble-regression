clear all; %close all;
addpath Ensemble_Regressors;

%load 'Datasets/mlp_large_friedman1.mat';
load 'Datasets/mlp_large_friedman1_big.mat';
%load 'Datasets/auto_large_friedman1.mat';
%load 'Datasets/mlp_xlarge_friedman1.mat';
y_true = y; clear y; %Z = Z(1:20,:); Ztrain=Ztrain(1:20,:); 
%Z = Z(1:6,:); Ztrain = Ztrain(1:6,:);

% bias correction
Ey = mean([y_true ytrain]); mu = mean([Z Ztrain],2);
y_true = y_true - Ey; ytrain = ytrain - Ey;
Z = bsxfun(@minus, Z, mu); Ztrain = bsxfun(@minus,Ztrain,mu);

Ztest = Z(:,1:1000); ytest = y_true(1:1000); ytest = ytest - mean(ytest);
Z = Z(:,1001:end); y_true = y_true(1001:end); y_true = y_true - mean(y_true);

[m n] = size(Z);
Ey = mean([y_true ytrain]);
Ey2 = mean([y_true ytrain].^2);
var_y = Ey2 - Ey.^2;
mse = @(x) mean((y_true' - x).^2 / var_y);

%%
n_L_list = [100 150 200 300 ];%[50,80,120,200]; - this was what I sent boaz that was good
n_U_list = 2:1000:n;
for n_L_idx = 1:length(n_L_list);
    n_L = n_L_list(n_L_idx);
    for n_U_idx = 1:length(n_U_list)
        n_U = n_U_list(n_U_idx);
        mse = @(x) mean((ytest' - x).^2 / var_y);
        
        [y_oracle2,w_oracle2] = ER_Oracle_2_Unbiased(ytest,Ztest);
        [y_oraclerho,w_oraclerho] = ER_Oracle_Rho(ytest,Ztest);
        [y_gem,w_gem] = ER_PerroneCooperGEM(Ztrain(:,1:n_L), ytrain(1:n_L), Ztest);
        y_gem = y_gem - mean(y_gem);
        [y_ssgem,w_ssgem] = ER_SemiSupervisedGEM(Ztrain(:,1:n_L), ytrain(1:n_L), Z(:,1:n_U), Ztest, Ey, Ey2);
        [y_ssfpf,w_ssfpf] = ER_FixedPointFunctionalSemiSupervised(Ztrain(:,1:n_L), ytrain(1:n_L), Z(:,1:n_U), Ztest, 5,10); % lambda=1
        
        MSE_ORACLE2(n_L_idx, n_U_idx) = mean((y_oracle2' - ytest).^2/var_y);
        MSE_ORACLERHO(n_L_idx, n_U_idx) = mean((y_oraclerho' - ytest).^2/var_y);
        MSE_GEM(n_L_idx, n_U_idx) = mean((y_gem' - ytest).^2/var_y);
        MSE_SSGEM(n_L_idx, n_U_idx) = mean((y_ssgem' - ytest).^2/var_y);
        MSE_SSFPF(n_L_idx, n_U_idx) = mean((y_ssfpf' - ytest).^2/var_y);
    end;
end;

%% Plots
red = [1 0 0]; blue = [0 0 1]; mag = [1 0 1];
cmap = [.3*red; .5*red; .7*red; red; .3*blue; .5*blue; .7*blue; blue; .3*mag; .5*mag; .7*mag; mag];
set(groot,'DefaultAxesColorOrder',cmap);
set(groot,'DefaultLineLineWidth',2);
figure; 
plot(n_U_list, MSE_GEM, '-', n_U_list, MSE_SSGEM, '-', n_U_list, MSE_SSFPF, '-',n_U_list, MSE_ORACLERHO(1,:),'k-');%, n_U_list, MSE_ORACLE2,'k--'); 
legend_entries = [repmat({'GEM n_L='},4,1); repmat({'ss GEM n_L='},4,1); repmat({'ss FPF n_L='},4,1); {'Oracle \rho'}];
for i=1:4; 
    legend_entries{i} = [legend_entries{i} num2str(n_L_list(i))]; 
    legend_entries{i+4} = [legend_entries{i+4} num2str(n_L_list(i))]; 
    legend_entries{i+8} = [legend_entries{i+8} num2str(n_L_list(i))];     
end;
legend(legend_entries);
grid on; ylabel('MSE / Var(Y)','FontSize',20); xlabel('n_U','FontSize',20);



