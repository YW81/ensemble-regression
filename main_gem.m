%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Iterative approach for ensemble regression
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Init
addpath 'Ensemble Regressors'

if ~exist('outsideParams') || (~outsideParams)
    clc; clear all; close all;
    m = 15;
    n = 1000;
    num_iterations = 20;
    dontPlot = 1;

    %% Create data for simulation
    n_training_set = 200;
    y_true = linspace(100,200,n + n_training_set);
    Ey = mean(y_true);

    % min/max bias/variance
    min_bias = -.75*(max(y_true) - min(y_true)); % -75
    max_bias = 1.5*(max(y_true) - min(y_true));  %+150
    max_var = (max(y_true) - min(y_true))^2;     % 100^2
    min_var = max_var/4;                         % 
end;

[Z,real_bias,real_var,real_Sigma] = GenerateNormalData(m, n + n_training_set, y_true, ...
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


%% Fully Supervised
[y_supervised, w_supervised, R_supervised, rho_supervised] = ...
    ER_FullySupervised(Ztrain, ytrain, Z);
    
%% Perrone & Cooper Supervised GEM
[y_sgem, w_sgem, C_sgem] = ER_PerroneCooperGEM(Ztrain, ytrain, Z);

%% Oracle Prediction
[y_oracle,w_oracle,R_real,rho_real] = ...
    calculate_oracle(Ey,min(y_true),max(y_true),real_Sigma, real_bias,Z);

%% Oracle 2 - Empirical R/rho
% mu_real = mu; %Ey+real_bias;
% R_real2 = R_hat; %real_Sigma + mu_real*mu_real';
% rho_real2 = sum(bsxfun(@times, Z,y_true),2) / n;
% w_oracle2 = inv(R_real2) * rho_real2;
% w_oracle2 = w_oracle2 ./ sum(w_oracle2);
% y_oracle2 = (Z - repmat(real_bias,1,n))' * w_oracle2;

%% 2nd Moment Estimator (assumes Ey, Ey^2 are given)
[y_2me, w_2me, rho_hat] = ER_SecondMoment(Z,Ey,var(y_true)+Ey^2,mu, b_hat, R_hat);

%% Iterate
for i=1:num_iterations
    %% Option 0: assuming uncorrelated predictors
    % y_uncorr{i} = Z' * w_uncorr{i}
    y_uncorr{i} = Z' * w_uncorr{i} - repmat(w_uncorr{i}',n,1)*b_hat;
    
    % update weights
    %w_uncorr{i+1} = (Z * y_uncorr{i}) ./ sum( Z.^ 2 ,2); % denominator = sum f_i^2 of all samples
    w_uncorr{i+1} = ((Z - repmat(b_hat,1,n)) * y_uncorr{i}) ./ sum( (Z - repmat(b_hat,1,n)).^ 2 ,2); % denominator = sum f_i^2 of all samples
    w_uncorr{i+1} = w_uncorr{i+1} / sum(w_uncorr{i+1}); % keep sum(w) = 1.

    %% Option 1: Iterative approach - estimate b,Sigma, calculate R, update prediction
    % find y_hat
    %y_uncentered_gem{i} = Z' * w_uncentered_gem{i} - repmat(w_uncentered_gem{i}',n,1)*b_hat;
    y_uncentered_gem{i} = (Z - repmat(b_hat,1,n))' * w_uncentered_gem{i};
    
    % update w
    w_uncentered_gem{i+1} = inv(R_hat)*Z*y_uncentered_gem{i} / n;
    w_uncentered_gem{i+1} = w_uncentered_gem{i+1} / sum(w_uncentered_gem{i+1}); % keep sum(w) = 1. TODO: Is this needed?

    %% Option 2 = unsupervised General Ensemble Method 
    % get prediction
    y_gem{i} = (Z - repmat(b_hat,1,n))' * w_gem{i};

    % update weights
    misfit = repmat(y_gem{i}',m,1) - (Z - repmat(b_hat,1,n));
    if any(any(isnan(misfit)))
        C{i} = C{i-1}; % keep the original (mean) weighting
    else
        C{i} = misfit * misfit' / n;
    end;
    Cinv = pinv(C{i});
    w_gem{i+1} = sum(Cinv,1)' ./ sum(sum(Cinv)); % w_i = sum_j(Cinv_ij) / sum(sum(Cinv))

   
    %% Option 3* = Randomize Cross-Validation and choose the best subset of Z
    % Randomly choose m/2 classifiers. Take their mean and treat it as the
    % true value for y. Update the other m/2 classifiers based on that
    % value of y (with a learning rate). Idea is - if  3/4 of the
    % classifiers are somewhat correct (low bias) there's a high
    % probability of selecting them at each round, and thus eliminating the
    % 
    
    % Caluculate cost functions
    cost_func_uncenetered_gem(i) = w_uncentered_gem{i}'*R_hat*w_uncentered_gem{i} - 2*w_uncentered_gem{i}'*Z*y_uncentered_gem{i};
    cost_func_gem(i) = w_gem{i}'*C{i}*w_gem{i};        
    cost_func_uncorr(i) = w_uncorr{i}'*(R_hat .* eye(size(R_hat)))*w_uncorr{i} - 2*w_uncorr{i}'*Z*y_uncorr{i}; % like opt, only assuming no correlation
end

MSE_uncentered_gem = mean((y_uncentered_gem{i} - y_true').^2);
MSE_gem = mean((y_gem{i} - y_true').^2);
MSE_uncorr = mean((y_uncorr{i} - y_true').^2);
MSE_mean_f_i = mean((mean(Z - repmat(b_hat,1,n),1) - y_true) .^2);
MSE_supervised = mean((y_supervised - y_true) .^2);
MSE_oracle = mean((y_oracle - y_true') .^2);
% MSE_oracle2 = mean((y_oracle2 - y_true') .^2);
MSE_2me = mean((y_2me - y_true') .^2);
MSE_f_best = min(mean((Z - repmat(b_hat,1,n) - repmat(y_true,m,1)).^2,2)); % MSE of the best single predictor in the ensemble, given the estimated bias
MSE_sgem = mean((y_sgem - y_true).^2);


fprintf(['MSE[mean f_i] = %g\nMSE[supervised] = %g\nMSE[oracle] = %g\n' ...
         'MSE[uncentered gem] = %g\nMSE[gem] = %g\nMSE[uncorr] = %g\n' ...
         ...%'MSE[oracle 2] = %g\n' ...
         'MSE[best predictor] = %g\nMSE[2me] = %g\nMSE[Supervised GEM] = %g\n'], ...
    MSE_mean_f_i,MSE_supervised, MSE_oracle, ...
    MSE_uncentered_gem,MSE_gem,MSE_uncorr, ... 
    ... % MSE_oracle2,
    MSE_f_best,MSE_2me,MSE_sgem);

fprintf('\n----------------------\n');
fprintf('MSE[oracle] =\t%g\nMSE[SV GEM] =\t%g\nDiff =\t\t\t%g\n',MSE_oracle,MSE_sgem,MSE_sgem - MSE_oracle);
fprintf('\n----------------------\n');
fprintf('MSE[oracle] =\t%g\nMSE[Superv] =\t%g\nDiff =\t\t\t%g\n',MSE_oracle,MSE_supervised,MSE_supervised - MSE_oracle);

%% Plotting
if ~exist('dontPlot') || (~dontPlot)

ht = 3; wd = 2; % height and width of the plot (given in # of subplots)
figure;

subplot(ht,wd,1); hold all;
close all;
end