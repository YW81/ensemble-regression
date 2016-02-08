%% Init
clc; clear all; close all;
addpath 'Ensemble Regressors'
addpath 'DataGeneration'

n_training_set = 200;
n = 1e5;
m = 15;
y_true = linspace(100,200,n + n_training_set);
Ey = 150;
max_var = mean(y_true)^2;
min_var = max_var/16;
min_bias = -.75*(max(y_true) - min(y_true));
max_bias = 1.5*(max(y_true) - min(y_true));

%% Generate Data
rng(0);
% [Zfull_orig,bias_real_orig,var_real_orig,Sigma_real] = ...
%      GenerateNormalData(m, n + n_training_set, y_true, min_bias, max_bias, min_var, max_var);
[Zfull_orig,bias_real_orig,var_real_orig,Sigma_real_orig] = ...
    GenerateIndependentData(m, n + n_training_set, y_true, min_bias, max_bias, min_var, max_var);
                                        
[Ztrain_orig, ytrain, Ztest_orig, ytest] = TrainTestSplit(Zfull_orig, y_true, ...
                                                          n_training_set / (n+n_training_set));

%% Go over parameter list
param_name = 'bias';
param_list = linspace(-1e3,1e3,5e2);
%param_name = 'variance';
%param_list = linspace(1,1e3,1e2);
tic
for i=1:numel(param_list)
    bias_real = bias_real_orig;
    var_real = var_real_orig;
    Sigma_real = Sigma_real_orig;
    if strcmp(param_name, 'bias')
        % Adapt data for param=bias        
        Ztrain = Ztrain_orig + param_list(i);
        Ztest = Ztest_orig + param_list(i);
        Zfull = Zfull_orig + param_list(i);
        bias_real = bias_real_orig + param_list(i);
    
    elseif strcmp(param_name, 'variance')
        % Adapt data for param=variance
%         Ztrain = Ztrain_orig * sqrt(param_list(i) / max_var);
%         Ztest = Ztest_orig * sqrt(param_list(i) / max_var);
%         Zfull = Zfull_orig * sqrt(param_list(i) / max_var);
%         var_real = var_real_orig * sqrt(param_list(i) / max_var);
%         Sigma_real = Sigma_real_orig .* sqrt(var_real * var_real');
        max_var = param_list(i);
        min_var = max_var / 16;
        [Zfull,bias_real,var_real,Sigma_real] = ...
            GenerateIndependentData(m, n + n_training_set, y_true, min_bias, max_bias, min_var, max_var);

        [Ztrain, ytrain, Ztest, ytest] = TrainTestSplit(Zfull_orig, y_true, ...
                                                                  n_training_set / (n+n_training_set));
    end;
    
    [y_hat,w,R,rho] = ER_FullySupervised(Ztrain, ytrain, Ztest);
    MSE(i) = mean((y_hat - ytest).^2); %#ok<SAGROW>
    fprintf(['MSE(' param_name ' = %g) = %g\n'],param_list(i),MSE(i));
    
    % Oracle
    [y_oracle,w_oracle,R_oracle,rho_oracle] = ...
        calculate_oracle(Ey,min(y_true),max(y_true),Sigma_real,bias_real,Ztest); 
    y_oracle = y_oracle';
    MSE_oracle(i) = mean((y_oracle - ytest).^2); %#ok<SAGROW>
end;
toc

%% Plotting
figure('Name',['MSE vs ' param_name]);
plot(param_list,MSE,'.-'); hold on; 
plot([min(param_list) max(param_list)],[mean(MSE) mean(MSE)],'--k');
%plot([min(param_list) max(param_list)],[MSE_oracle MSE_oracle],':r','linewidth',4);
plot(param_list,MSE_oracle ,':r','linewidth',4);
xlabel(param_name); ylabel('MSE'); grid on;
legend('Fully Supervised','Mean[Fully Supervised]','Oracle');
title({ 
    ['MSE vs ' param_name] , ...
    ['Min at ' param_name '= ' num2str(param_list(MSE == min(MSE)))] , ...
    ['Max at ' param_name '= ' num2str(param_list(MSE == max(MSE)))  ] 
      });

%% Compare rho
rho_empirical = mean(repmat(ytest,m,1) .* Ztest,2);
fprintf('\nrho = E[yf_i]\t/1e6\n====================\n');
fprintf('Full SV\t|Empir.\t|Oracle\t|\n');
fprintf('%.2f\t|%.2f\t|%.2f\t|\n',[rho rho_empirical rho_oracle] / 1e6)
norm(rho - rho_empirical,1)
norm(rho_oracle - rho_empirical,1)

%% Compare R
R_empirical = Ztest*Ztest' / n;
figure; 
subplot(311); imagesc(R); colorbar; colormap(gray); title('R - Fully SV');
subplot(312); imagesc(R_empirical); colorbar; colormap(gray); title('R - Empirical');
subplot(313); imagesc(R_oracle); colorbar; colormap(gray); title('R - Oracle');
norm(R - R_empirical,1)
norm(R_oracle - R_empirical,1)
