%   Comparing approaches for ensemble regression
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Init
addpath 'Ensemble_Regressors'
addpath 'DataGeneration'

if ~exist('A_outsideParams') || (~A_outsideParams)
    clc; clear all; close all;
    m = 15;
    n = 1000;
    num_iterations = 20;
    dontPlot = 1;

    %% Create data for simulation
    n_training_set = 200;
    y_true = linspace(100,200,n + n_training_set);
    Ey = mean(y_true);
    Ey2 = mean(y_true.^2);

    % min/max bias/variance
    min_bias = -.75*(max(y_true) - min(y_true)); % -75
    max_bias = 1.5*(max(y_true) - min(y_true));  %+150
    max_var = (max(y_true) - min(y_true))^2;     % 100^2
    min_var = max_var/4;                         % 
end;

% [Z,real_bias,real_var,real_Sigma] = GenerateNoBoazrmalData(m, n + n_training_set, y_true, ...
%                                             min_bias, max_bias, ...
%                                             min_var, max_var);
[Z,real_bias,real_var,real_Sigma] = GenerateDependentData(m, n + n_training_set, y_true, ...
                                            min_bias, max_bias, ...
                                            min_var, max_var);
                                        
[Ztrain, ytrain, Z, y_true] = TrainTestSplit(Z, y_true, n_training_set / (n+n_training_set));

% TODO:
%2. Generate the data in different ways (uncorrelated, block diagonal covariance, wishart, others).

%% Simulation Initializations
y_uncorr = cell(num_iterations,1);
y_uncentered_gem = cell(num_iterations,1);
y_gem = cell(num_iterations,1);
w_uncorr = cell(num_iterations+1,1);
w_uncentered_gem = cell(num_iterations+1,1);
w_gem = cell(num_iterations+1,1);
cost_func_uncenetered_gem = zeros(num_iterations,1);
cost_func_uncorr = zeros(num_iterations,1);
cost_func_gem = zeros(num_iterations,1);
C = cell(num_iterations,1);

% Initialize weights
w_uncorr{1} = ones(m,1) / m; % Init w for equal weights
w_uncentered_gem{1} = ones(m,1) / m;
w_gem{1} = ones(m,1) / m;
b_last = zeros(m,1); % init bias estimation to zero

%% Init estimations of bias and variance
b_hat = mean(Z,2) - Ey;
mu = mean(Z,2);    
Z_centered = Z - repmat(mu,1,n);    
Sigma = Z_centered * Z_centered' ./ (n-1); % unbiased. maximum likelihood is 1/n
R_hat = Sigma + mu*mu';


%% Perrone & Cooper Supervised GEM
[y_pcGEM, w_pcGEM, C_pcGEM] = ER_PerroneCooperGEM(Ztrain, ytrain, Z);

%% P&C Supervised GEM without covariance centering
[y_sgem_no_centering, w_sgem_no_centering, R_sgem_no_centering, rho_sgem_no_centering] = ...
    ER_SupervisedGEM_no_centering(Ztrain, ytrain, Z);
    
%% Oracle Prediction
[y_oracle, beta_oracle] = ER_linear_regression_oracle(y_true, Z);
% [y_oracle,w_oracle,R_real,rho_real] = ...
%     calculate_oracle(y_true,Ey,real_Sigma, real_bias,Z);single_run

%% Oracle 2 - Empirical R/rho
% mu_real = mu; %Ey+real_bias;
% R_real2 = R_hat; %real_Sigma + mu_real*mu_real';
% rho_real2 = sum(bsxfun(@times, Z,y_true),2) / n;
% w_oracle2 = inv(R_real2) * rho_real2;
% w_oracle2 = w_oracle2 ./ sum(w_oracle2);
% y_oracle2 = (Z - repmat(real_bias,1,n))' * w_oracle2;

%% 2nd Moment Estimator (assumes Ey, Ey^2 are given)
%[y_2me, beta_2me, rho_hat] = ER_SecondMoment(Z,Ey,Ey2,mu, b_hat, R_hat);
[y_2me, beta_2me] = ER_Boaz(Z,Ey,Ey2);

%% Iterate
for i=1:num_iterations
    %% Option 0: Assuming uncorrelated predictors
    [y_uncorr{i}, w_uncorr{i+1}] = ER_AssumeNoCorrelation(Z, b_hat, w_uncorr{i});

    %% Option 1: Uncentered GEM (estimate b,Sigma, calculate R, update prediction)
    [y_uncentered_gem{i}, w_uncentered_gem{i+1}] = ...
                                  ER_unsupervisedUncenteredGEM(Z, b_hat, R_hat, w_uncentered_gem{i});

    %% Option 2 = Unsupervised General Ensemble Method 
    [y_gem{i}, w_gem{i+1}, C{i}] = ER_unsupervisedGEM(Z, b_hat, w_gem{i});
end

%% Option 3* = Randomize Cross-Validation and choose the best subset of Z
% Randomly choose m/2 classifiers. Take their mean and treat it as the
% true value for y. Update the other m/2 classifiers based on that[1;real_var]
% value of y (with a learning rate). Idea is - if  3/4 of the
% classifiers are somewhat correct (low bias) there's a high
% probability of selecting them at each round, and thus eliminating the
% 

%% Calculate MSEs
% MSE_sgem_no_centering = mean((y_sgem_no_centering - y_true) .^2);
% fprintf('MSE[GEM no centering] = %g\n',MSE_sgem_no_centering);

MSE_pcGEM = mean((y_pcGEM - y_true).^2);
fprintf('MSE[P&C GEM] = %g\n',MSE_pcGEM);

% MSE_oracle2 = mean((y_oracle2 - y_true') .^2);
MSE_oracle = mean((y_oracle - y_true') .^2);
fprintf('--\nMSE[Oracle] = %g\n',MSE_oracle);

MSE_f_best = min(mean((Z - repmat(b_hat,1,n) - repmat(y_true,m,1)).^2,2)); % MSE of the best single predictor in the ensemble, given the estimated bias
fprintf('MSE[best predictor] = %g\n',MSE_f_best);

MSE_mean_f_i = mean((mean(Z - repmat(b_hat,1,n),1) - y_true) .^2);
fprintf('MSE[Mean f_i] = %g\n',MSE_mean_f_i);

% MSE_uncorr = mean((y_uncorr{i} - y_true').^2);
% fprintf('MSE[Uncorrelated] = %g\n',MSE_uncorr);

MSE_2me = mean((y_2me - y_true') .^2);
fprintf('--\nMSE[2nd Moment] = %g\n',MSE_2me);

MSE_gem = mean((y_gem{i} - y_true').^2);
fprintf('MSE[US GEM] = %g\n',MSE_gem);

MSE_uncentered_gem = mean((y_uncentered_gem{i} - y_true').^2);
fprintf('MSE[US Uncentered GEM] = %g\n',MSE_uncentered_gem);

%% Plotting
% if ~exist('A_dontPlot') || (~A_dontPlot)
% 
% ht = 3; wd = 2; % height and width of the plot (given in # of subplots)
% figure;
% 
% subplot(ht,wd,1); hold all;
% close all;
% end