%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Iterative approach for ensemble regression
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Init
%clc; clear all; close all;

m = 15;
n = 100000;
num_iterations = 20;

%% Create data for simulation
training_set_size = 200;
y_true = linspace(100,200,n + training_set_size);
Ey = mean(y_true);

% min/max bias/variance
min_bias = -.75*(max(y_true) - min(y_true)); % -75
max_bias = 1.5*(max(y_true) - min(y_true));  %+150
min_var = (max(y_true) - min(y_true)) / 10;  %  50
max_var = (max(y_true) - min(y_true));       % 150

[Z,real_bias,real_var,real_Sigma] = GenerateNormalData(m, n + training_set_size, y_true, ...
                                            min_bias, max_bias, ...
                                            min_var, max_var);
                                        
[Ztrain, ytrain, Z, y_true] = TrainTestSplit(Z, y_true, training_set_size / (n+training_set_size));

% TODO:
%1. Add a supervised ensemble regressor
%2. Generate the data in different ways (uncorrelated, block diagonal covariance, wishart, others).

%% Simulation Initializations
y_uncorr = cell(num_iterations,1);
y_opt = cell(num_iterations,1);
y_gem = cell(num_iterations,1);
w_uncorr = cell(num_iterations+1,1);
w_opt = cell(num_iterations+1,1);
w_gem = cell(num_iterations+1,1);
cost_func_opt = zeros(num_iterations,1);
cost_func_uncorr = zeros(num_iterations,1);
cost_func_gem = zeros(num_iterations,1);
%b = cell(num_iterations,1);
%Sigma = cell(num_iterations,1);
%R = cell(num_iterations,1);
C = cell(num_iterations,1);

% Initialize weights
w_uncorr{1} = ones(m,1) / m; % Init w for equal weights
w_opt{1} = ones(m,1) / m;
w_gem{1} = ones(m,1) / m;
b_last = zeros(m,1); % init bias estimation to zero

%% Fully Supervised
R_supervised = Ztrain*Ztrain' / n;
rho_supervised = sum(bsxfun(@times, Ztrain,ytrain),2) / n;
w_supervised = inv(R_supervised) * rho_supervised;
y_supervised = w_supervised' * Z;

%% Init for option 1: estimate bias and variance
b_hat = mean(Z,2) - Ey;
mu = mean(Z,2);    
Z_centered = Z - repmat(mu,1,n);    
Sigma = Z_centered * Z_centered' ./ (n-1); % unbiased. maximum likelihood is 1/n
R_hat = Sigma + mu*b_hat' + b_hat*mu' + mu*mu';

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
    %y_opt{i} = Z' * w_opt{i} - repmat(w_opt{i}',n,1)*b_hat;
    y_opt{i} = (Z - repmat(b_hat,1,n))' * w_opt{i};
    
    % update w
    w_opt{i+1} = inv(R_hat)*Z*y_opt{i} / n;
    w_opt{i+1} = w_opt{i+1} / sum(w_opt{i+1}); % keep sum(w) = 1. TODO: Is this needed?

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
    cost_func_opt(i) = w_opt{i}'*R_hat*w_opt{i} - 2*w_opt{i}'*Z*y_opt{i};
    cost_func_gem(i) = w_gem{i}'*C{i}*w_gem{i};        
    cost_func_uncorr(i) = w_uncorr{i}'*(R_hat .* eye(size(R_hat)))*w_uncorr{i} - 2*w_uncorr{i}'*Z*y_uncorr{i}; % like opt, only assuming no correlation
end

MSE_opt = mean((y_opt{i} - y_true').^2);
MSE_gem = mean((y_gem{i} - y_true').^2);
MSE_uncorr = mean((y_uncorr{i} - y_true').^2);
MSE_mean_f_i = mean((mean(Z - repmat(b_hat,1,n),1) - y_true) .^2);
MSE_supervised = mean((y_supervised - y_true) .^2);
% fprintf('MSE[opt] = %g\n',MSE_opt);
% fprintf('MSE[gem] = %g\n',MSE_gem);
% fprintf('MSE[uncorr] = %g\n',MSE_uncorr);
% fprintf('MSE[mean f_i] = %g\n',MSE_mean_f_i);
% fprintf('MSE[supervised] = %g\n',MSE_supervised);
fprintf('MSE[opt] = %g\nMSE[gem] = %g\nMSE[uncorr] = %g\nMSE[mean f_i] = %g\nMSE[supervised] = %g\n',MSE_opt,MSE_gem,MSE_uncorr,MSE_mean_f_i,MSE_supervised);

%% Plotting
if ~exist('dontPlot') || (~dontPlot)

figure;
ht = 4; wd = 2; % height and width of the plot (given in # of subplots)

subplot(ht,wd,[1 3]); hold all; 
title('Estimation per sample','fontsize',22);
%plot([ y_opt{num_iterations} y_uncorr{num_iterations} ...
%       y_gem{num_iterations} mean(Z,1)']);
plot(y_opt{num_iterations},'x-'); 
plot(y_uncorr{num_iterations},'o-');
plot(y_gem{num_iterations},'.-'); 
plot(mean(Z - repmat(b_hat,1,n),1)); 
plot(y_true,'*');
legend('y-opt','y-uncorr','y-gem','E[f_i - b_i]','true y');

subplot(ht,wd,5); hold all;
title('Error per sample (after final iteration)','fontsize',22);
plot(y_opt{num_iterations} - y_true','x-');
plot(y_uncorr{num_iterations} - y_true','o-'); 
plot(y_gem{num_iterations} - y_true','.-'); 
plot((mean(Z - repmat(b_hat,1,n),1)) - y_true); 
legend('y-opt error','y-uncorr error','gem error','mean f_i - b_i error');

subplot(ht,wd,7); hold all;
title('Square Error per sample (after final iteration)','fontsize',22); 
plot((y_opt{num_iterations} - y_true').^2,'x-'); 
plot((y_uncorr{num_iterations} - y_true').^2,'o-'); 
plot((y_gem{num_iterations} - y_true').^2,'.-'); 
plot(((mean(Z - repmat(b_hat,1,n),1)) - y_true) .^2); 
legend('y-opt error^2','y-uncorr error^2','gem error^2','mean f_i - b_i error^2');
xlabel('sample #','fontsize',22);

subplot(ht,wd,[2 4]); imagesc(Z); c = colorbar('southoutside'); % colormap(gray);
xlabel('samples'); ylabel('ensemble predictors'); title('Ensemble Predictions');

subplot(ht,wd,[6 8]);
plot(w_opt{num_iterations},'x-'); hold all; 
plot(w_uncorr{num_iterations},'o-'); 
plot(w_gem{num_iterations},'.-'); 
plot((1/m) * abs(1./real_bias) / max(1./real_bias),'.-'); 
plot((1/m) * abs(real_var) / max(real_var),'.-'); xlabel('regressor');
bias_str = sprintf('1/m * 1/|bias_j| * %.1f', 1/max(1./real_bias));
var_str = sprintf('1/m * variance_j / %.1f', max(real_var));
title('Weights & Biases'); legend('opt weight','uncorr weight','gem weights',bias_str,var_str);

%% Some more plotting
ht = 3; wd = 2; % height and width of the plot (given in # of subplots)
figure;

subplot(ht,wd,1); hold all;
title('MSE by iteration');
plot(mean((cell2mat(y_opt')' - repmat(y_true,num_iterations,1)).^2,2), 'x-');
plot(mean((cell2mat(y_gem')' - repmat(y_true,num_iterations,1)).^2,2), 'o-');
plot(mean((cell2mat(y_uncorr')' - repmat(y_true,num_iterations,1)).^2,2));
plot(repmat(MSE_mean_f_i,1,num_iterations),'--');
plot(repmat(MSE_supervised,1,num_iterations),'--');
xlabel('iteration');
legend('opt','gem','uncorr','unbiased mean','supervised');

subplot(ht,wd,2); hold all;
title('Weights by predictor');
plot(w_opt{num_iterations},'x-');
plot(w_gem{num_iterations},'o-');
plot(w_uncorr{num_iterations});
xlabel('w_i');
legend('opt','gem','uncorr');

subplot(ht,wd,3); hold all;
title('Cost Functions change by iteration');
plot(cost_func_opt,'x-'); 
plot(cost_func_gem,'o-');
plot(cost_func_uncorr); 
legend('opt','gem','uncorr');

subplot(ht,wd,4); hold all;
title('Absolute total change in weights by iteration');
tmp = cell2mat(w_opt'); plot(sum(abs(tmp(:,2:end) - tmp(:,1:end-1))),'x-');
tmp = cell2mat(w_gem'); plot(sum(abs(tmp(:,2:end) - tmp(:,1:end-1))),'o-');
tmp = cell2mat(w_uncorr'); plot(sum(abs(tmp(:,2:end) - tmp(:,1:end-1))));
xlabel('iteration');
legend('opt','gem','uncorr');

subplot(ht,wd,5); 
imagesc(Sigma); colormap(gray); colorbar;
title('Empirical (uncentered) Covariance Matrix');

subplot(ht,wd,6);
imagesc(real_Sigma); colormap(gray); colorbar;
title('Population (centered) Covariance Matrix');
end