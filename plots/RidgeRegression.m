addpath Datasets
addpath ../Ensemble_Regressors/
clear all; close all;

% RIDGE REGRESSION MATLAB EXAMPLE
%% Init
tic;
close all;
clear all;
%load RidgeRegression_Friedman1; 
%load RidgeRegression_Friedman2; 
%load RidgeRegression_Friedman3; 
%load RidgeRegression_Friedman1_200k; y_RidgeCV = y_RidgeCV';
load RidgeRegression_Friedman2_200k; y_RidgeCV = y_RidgeCV';
%load RidgeRegression_Friedman3_200k; y_RidgeCV = y_RidgeCV';
y_true = y; clear y;
y_BiasedMean = mean(Z); %Z = Z(randperm(size(Z,1)),:);
y_BiasedMean = y_BiasedMean';

var_y = Ey2 - Ey.^2;

[m,n] = size(Z);

%%
Zfull = Z; y_true_full = y_true;
% m_min = 3;
% m_max = m;
n_max = n;
n = 2000;

rng;
%n_start = 1000; n_spacing = 1000; n_stop = 20000;
%for n=n_start:n_spacing:n_stop
n = 10000; n_spacing = n; n_stop = 200000;
for n_start = 1:n_spacing:n_stop

% % for m=m_min:m_max
%num_iters = 10;
%for iter=1:num_iters
%    idxs = randperm(n_max);
%    y_true_perm = y_true_full(1,idxs);
%    Zfullperm = Zfull(:,idxs);
    y_true = y_true_full(1,n_start:n_start+n-1);
    Z = Zfull(1:m,n_start:n_start+n-1);
    
%    y_true = y_true_perm(1,1:n);
    mse = @(x) (mean((y_true'-x).^2)) / var_y;
%    Z = Zfullperm(1:m,1:n);

    %% Ensemble Methods
    % Oracle
    [y_oracle, beta_oracle] = ER_linear_regression_oracle(y_true, Z);

    % Calculate mean predictions
    [y_mean, beta_mean] = ER_MeanWithBiasCorrection(Z, Ey);

    % Find best predictor in the ensemble
    [y_best, w_best] = ER_BestRegressor(y_true, Z);

    % Calculate varw predictions
    [y_qd, w_qd] = ER_UnsupervisedDiagonalGEM(Z,Ey);
    [y_varw,beta_varw] = ER_VarianceWeightingWithBiasCorrection(Z,Ey);

    % Calculate predictions dropping the assumption of a diagonal covariance matrix (assumes rho=c*ones(m,1) is uni-directional)
    Z0 = Z - mean(Z,2)*ones(1,n); warnstate=warning('off','MATLAB:rankDeficientMatrix'); a=mvregress(Z0', y_true'); warning(warnstate); disp(a/sum(abs(a)));
    [y_invc, beta_invc] = ER_UnsupervisedGEM(Z,Ey,Ey2);
    w_invc = beta_invc(2:end);

    % Calculate predictions with Spectral Method (L2 regularization on w)
    [y_sm, w_sm, t_sm] = ER_SpectralApproach(Z,Ey,Ey2);

    % Calculate predictions for the low rank misfit covariance method
    [y_lrm, w_lrm, y_rho_oracle, w_rho_oracle] = ER_LowRankMisfit( Z, Ey, Ey2, y_true );
    [y_lrmcvx, w_lrmcvx, y_rho_oracle2, w_rho_oracle2] = ER_LowRankMisfitCVX( Z, Ey, Ey2, y_true );
    
    disp([beta_oracle(2:end)/sum(abs(beta_oracle(2:end))) w_rho_oracle/sum(abs(w_rho_oracle)) w_invc/sum(abs(w_invc))]);    
    disp([sum(abs(beta_oracle(2:end))) sum(abs(w_rho_oracle)) sum(abs(w_invc))]);    

    %% Calculate MSEs
    num_predictors = 10;        
    results = { ...
        %'RandomForest', mse(y_RandomForest); ...
        'RidgeCV', mse(y_RidgeCV(n_start:n_start+n-1)); ...
        'Oracle', mse(y_oracle); ...
        'OracleRho', mse(y_rho_oracle); ...        
        'Mean', mse(y_mean); ...
        'Best', mse(y_best); ...
        'uDGEM', mse(y_qd); ...
...%        'VarW', mse(y_varw); ...
        'uGEM', mse(y_invc); ...
        'uPCRstar', mse(y_sm); ...
        'LowRankMisfit', mse(y_lrm); ...
        'LowRankMisfitCVX', mse(y_lrmcvx); ...

        'n', n; ...
        'm', m
        };
%%

    if ~exist('results_summary','var')
        %results_summary = cell(num_iters*length(n_start:n_spacing:n_stop), size(results,1));
        results_summary = cell(length(1:n_spacing:n_stop), size(results,1));
    end;
    for i=1:length(results)
        results_summary{floor(n_start / n_spacing)+1,i} = results{i,2};
    end;
 
% end; % for m
end; % for n

%%
cols = {results{:,1}};
out=cell2table(results_summary, 'VariableNames', cols) % 'RowNames', cellstr(num2str([m_min:m_max]')), 

results_table = cell2mat(results_summary);

%% PLOTTING ONLY FOR THE FIRST 1000 SAMPLES (ONLY LOOP OVER M)
figure('Name',Description);
%plt=plot(results_table(1:num_iters:size(results_table,1),num_predictors+1),results_table(1:num_iters:size(results_table,1),1:num_predictors),'x-'); 
plt=plot(1:size(results_table,1),results_table(1:size(results_table,1),1:num_predictors),'x-'); 
mrk={'o','s','x','d','v','^','*','+','>','<'}.'; % <- note: transpose...
col=num2cell(jet(num_predictors),2);
set(plt,{'marker'},mrk,{'markerfacecolor'},col,{'Color'},cellfun(@(x){x*.5},col));

%for n=n_start:n_spacing:n_stop; biased_mean_mse(floor((n-n_start)/n_spacing)+1) = mean((mean(Z(:,1:n)) - y_true(1:n)).^2) / var_y; end;
%hold on; plot(n_start:n_spacing:n_stop,biased_mean_mse,'k--');
for n_start=1:n_spacing:n_stop; biased_mean_mse(floor(n_start/n_spacing)+1) = mean((mean(Zfull(:,n_start:n_start+n-1)) - y_true_full(n_start:n_start+n-1)).^2) / var_y; end;
hold on; plot(1:n_stop/n_spacing,biased_mean_mse,'k--');

grid on; grid minor; axis auto; legend([cols(1:num_predictors), 'Ridge Regression mean (without bias correction)']);
%%title('\bf Ridge Regression: Comparing Unsupervised Ensemble Methods');
%xlabel('Number of samples (n)');
xlabel('Iteration');
ylabel('MSE / Var(Y)'); ylim([0 1]);
hold off; 
T = get(gca,'tightinset'); set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]);
drawnow;

%% BOX PLOTS - M PREDICTORS, 10 ITERATIONS PER METHOD
figure('Name',Description); hold on;

colors = num2cell(jet(num_predictors),2);
for i=1:num_predictors
    boxplot(results_table(:,i), results_table(:,num_predictors+1), 'positions', results_table(:,num_predictors+1) -25 + i*5, 'widths', 5, 'colors', .8*colors{i},'outliersize',4,'symbol','x');
    pause(0.5);
end;
title('\bf Ridge Regressors: Comparing Unsupervised Ensemble Methods');
xlabel('Number of sampels (n)');
ylabel('MSE / Var(Y)');
ylim([0 6.5]); grid minor;
xlim([min(results_table(:,num_predictors+1)) - 25, max(results_table(:,num_predictors+1)) + num_predictors*5]);
drawnow; pause(0.5);
hLegend = legend(findall(gca,'Tag','Box'), cols(1:num_predictors));

    % Among the children of the gca, find all boxes
    box_vars = findall(gca,'Tag','Box');

    % Set the horizontal lines to the right colors
    for i=1:num_predictors
        idx = find(not(cellfun('isempty', strfind(cols, box_vars(i).DisplayName))));
        set(box_vars(idx),'Color',colors{i}*.8)
    end;
T = get(gca,'tightinset'); set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]);
hold off;

%% BOX PLOT FOR N=10000
wanted_n = 10000;
wanted_results = results_table(results_table(:,num_predictors+1) == wanted_n,:);
figure('Name',Description); hold on;

labels = {'RidgeCV', 'Oracle', 'Oracle Rho', 'Mean', 'Best', 'Uns. D-GEM','Uns. GEM','Uns. PCR*', 'Low Rank Misfit', 'Low Rank Misfit CVX'};
idxs = [2 3 1 5 8 4 6 7 9 10]; idxs = idxs(end:-1:1);
colors = num2cell(jet(num_predictors),2);
boxplot(wanted_results(:,idxs),'labels',labels(idxs),'widths',.5,'orientation','horizontal'); 
grid on; grid minor; %ylim([0,1]);
%title('\bf Ridge Regressors: Comparing Unsupervised Ensemble Methods');
ylabel('MSE / Var(Y)');
grid on; grid minor;
set(gca,'xscale','log');
ticks = [0:.05:.45 .5:.5:3];
set(gca,'xtick',ticks); set(gca,'XTickLabel',ticks)
T = get(gca,'tightinset'); set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]);
toc