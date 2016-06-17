%% Init
clear all; close all; 
clc;
addpath './Ensemble_Regressors'
[~,hostname] = system('hostname'); hostname = strtrim(hostname);
%ROOT = './Datasets/auto_mlp5/';
ROOT = './Datasets/auto/';
%ROOT = 'Datasets/auto_friedman1_test2/';
file_list = dir([ROOT '*.mat']);
datasets = cell(1,length(file_list));
for i=1:length(file_list)
    if ~strcmp(file_list(i).name, 'n_samples_10000000.mat')
        datasets{i}=[ROOT file_list(i).name];
    end;
end;
datasets = datasets(~cellfun('isempty',datasets)); % Call Built-in string

%% Load dataset
%datasets = {datasets{16}};

%% For each dataset
subplot_index = 0;
for dataset_name=datasets
    subplot_index = subplot_index + 1;
    if mod(subplot_index,12) == 1
        fig = figure; subplot_index = 1;
        subplot(3,4,1); subplot_index = subplot_index + 1;
        plot(0,0, 0,0, 0,0,'*r', 0,0,'xk' , 0,0,'sk', 0,0,'vk');
        legend({'MSE','Var($$\hat{y}$$)', 'min MSE','$$s_{max}$$','Var($$\hat{y}$$) = Var(y)','min reg err'},'interpreter','latex','location','north');
%         plot(0,0, 0,0, 0,0, 0,0,'*r', 0,0,'xk' , 0,0,'sk', 0,0,'vk');        
%         legend({'MSE','MSE K2','Var($$\hat{y}$$)', 'min MSE','$$s_{max}$$','Var($$\hat{y}$$) = Var(y)','min reg err'},'interpreter','latex','location','north');
    end;

    %%
    load(dataset_name{1})

if isempty(strfind(dataset_name{1},'MVGaussianDepData')) % if ~strcmp(dataset_name{1},'./Datasets/MVGaussianDepData.mat') % 
    y_true = double(y); clear y; % rename y to y_true and make sure both y and y_true are double
    ytrain = double(ytrain);     % (in the sweets dataset y is integer)
end;
y = y_true;
[m,n] = size(Z);
var_y = Ey2 - Ey.^2;
fprintf(['\n' dataset_name{1} '\t n = ' num2str(n) '\t var(y) = ' num2str(var_y) '\n']);

%% Oracle Prediction
[y_oracle, beta_oracle] = ER_linear_regression_oracle(y_true, Z);

%% Supervised Methods
[y_pc, w_pc, C_pc] = ER_PerroneCooperGEM(Ztrain, ytrain, Z);
[y_breiman, w_breiman, ~] = ER_BreimanStar(Ztrain, ytrain, Z);
[y_pcrstar,w_pcrstar, Kpcrstar] = ER_PCRstar(Ztrain, ytrain, Z); set(gcf,'Name',[regexprep(regexprep(dataset_name{1},'.*auto\/',''),'\.mat','') ' (n=' num2str(n) ')']);
[y_semi_s, w_semi_s, s_semi_s] = ER_SemiSupervised_S_SelectionSpectralApproach(Ztrain, ytrain, Z);
y_semi_comp = ER_SemiSupervisedWeightsComposition(Ztrain, ytrain, Z);

mse_oracle = mean((y_true'-y_oracle).^2);
mse = @(x) (mean((y_true'-x).^2)) / mse_oracle; % show MSE relative to oracle.

%% Do stuff
    %% Preparations
    b_hat = mean(Z,2) - Ey;
    Zc = Z - b_hat * ones(1,n);
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i    
    C = cov(Z'); % cov(Z') == cov(Zc') == cov(Z0')...
    mu = mean(Z,2);
    
    %% Calculate mean predictions
    [y_mean, beta_mean] = ER_MeanWithBiasCorrection(Z, Ey);
    
    %% Find best predictor in the ensemble
    [y_best, w_best] = ER_BestRegressor(y_true, Z);
  
    %% Calculate varw predictions
    var_i = var(Zc,[],2);
    w_varw = var_i / sum(var_i);
    y_varw = Zc'*w_varw;
    beta_varw = [0;w_varw];
    
    %% Calculate predictions dropping the assumption of a diagonal covariance matrix (assumes rho=c*ones(m,1) is uni-directional)
    w_rho_uni = (C\ones(m,1)) / sum(sum(pinv(C)));
    y_rho_uni = Ey + Z0' * w_rho_uni;
    
    %% Calculate predictions with oracle rho
    r_true = Z0*y_true' / n;
    lambda_oracle_rho = ( (ones(1,m)*(C\r_true) ) - 1) / (ones(1,m)*(C\ones(m,1)));
    w_rho_oracle = C \ (r_true - lambda_oracle_rho * ones(m,1));
    y_rho_oracle = Ey + Z0' * w_rho_oracle;
    
    %% Calculate predictions with Spectral Method (L2 regularization on w)
    
    % Preparations
    [V,D] = eigs(C,5,'lm'); % columns of V are possible w's. Take 5 largest magnitude eigvals
    eigvals = diag(D);
    lambda_1 = eigvals(1); v_1 = V(:,1); 

    % at s = s_max / s(MSE = 0), if y = f(x) non-stochastic and realizable, 
    % then this should be the exact solution
    s_max = var_y / lambda_1;
    y_SM_s_max = Ey + Z0' * (v_1 * s_max);
    mse_s_max = mse(y_SM_s_max);
    var_s_max = var(y_SM_s_max);
    
    % w = t*v_1 with s=t^2. Requiring sum(w)=1 means t=1/sum(v_1)
    s_wsum1 = (1/sum(v_1))^2;
    y_SM_s_wsum1 = Ey + Z0' * (v_1 * s_wsum1);
    mse_s_wsum1 = mse(y_SM_s_wsum1);
    var_s_wsum1 = var(y_SM_s_wsum1);
    
    % scan s, find s at: minMSE, var_y=var(y_hat), 
    s_scan = linspace(0,max(.75,1.5*s_max),200); 
    scan_mse = zeros(size(s_scan)); scan_cross_var = zeros(size(s_scan)); 
    scan_avg_regressor_err = zeros(size(s_scan));
    y_SM_scan = zeros(n,numel(s_scan));
    for i=1:length(s_scan); 
        y_SM_scan(:,i) = Ey + Z0' * v_1 * s_scan(i); 
        scan_mse(i) = mse(y_SM_scan(:,i)); 
        scan_cross_var(i) = var(y_SM_scan(:,i));
        scan_avg_regressor_err(i) = mean(mean((Z0 - repmat(y_SM_scan(:,i)',m,1)).^2,2));
    end;
    s_best_mse = s_scan(scan_mse == min(scan_mse));
    s_cross_var = s_scan(abs(var_y - scan_cross_var) == min(abs(var_y - scan_cross_var)));
    s_min_reg_err = s_scan(scan_avg_regressor_err == min(scan_avg_regressor_err));
    
    y_SM_s_best_mse = y_SM_scan(:,s_scan == s_best_mse);
    y_SM_s_cross_var = y_SM_scan(:,s_scan == s_cross_var);
    y_SM_s_min_reg_err = y_SM_scan(:,s_scan == s_min_reg_err);
    
    mse_s_best_mse = mse(y_SM_s_best_mse);
    mse_s_cross_var = mse(y_SM_s_cross_var);
    mse_s_min_reg_err = mse(y_SM_s_min_reg_err);
    
    var_s_best_mse = var(y_SM_s_best_mse) / var_y;
    var_s_cross_var = var(y_SM_s_cross_var) / var_y;
    var_s_min_reg_err = var(y_SM_s_min_reg_err) / var_y;
    
    % Test 2 leading eigenvectors
%     s_scan_K2 = linspace(0,max(4,2*s_max),50); 
%     lambda_2 = eigvals(2); v_2 = V(:,2);
%     scan_mse_K2 = zeros(size(s_scan)); y_SM_scan_K2 = zeros(n,numel(s_scan)); 
%     scan_mse_K2_inner = zeros(size(s_scan_K2)); y_SM_scan_K2_inner = zeros(n,numel(s_scan_K2));
%     for i=1:length(s_scan); 
%         for j=1:length(s_scan_K2)
%             y_SM_scan_K2_inner(:,j) = Ey + Z0' * (v_1*s_scan(i) + v_2*s_scan_K2(j));
%             scan_mse_K2_inner(j) = mse(y_SM_scan_K2_inner(:,j)); 
%         end;
%         min_j = find(scan_mse_K2_inner == min(scan_mse_K2_inner),1);
%         y_SM_scan_K2(:,i) = y_SM_scan_K2_inner(:,min_j);
%         scan_mse_K2(i) = scan_mse_K2_inner(min_j);
%         fprintf('s_scan for v_2 = %g == 0 ?\n',s_scan_K2(min_j));
%     end;
    

    %% Plotting
    %figure('Name',dataset_name{1});
%    subplot(6,ceil(numel(datasets)/6),subplot_index); subplot_index = subplot_index + 1;
    set(0,'currentfigure', fig);
    subplot(3,4,subplot_index);
%    hold on;
%     plot(s_scan,scan_mse,'.-'); grid minor;
%     plot(s_scan, scan_cross_var,'.-');    

%     [ax,~,~] = plotyy([s_scan' s_scan'],[(scan_mse / var_y)', (scan_mse_K2 / var_y)'],s_scan, scan_cross_var / var_y); grid minor;
    [ax,~,~] = plotyy(s_scan,scan_mse / var_y,s_scan, scan_cross_var / var_y); grid minor;
    xlabel('s');
    title([regexprep(regexprep(dataset_name{1},'.*auto\/',''),'\.mat','') '  (n=' num2str(n) ')'],'interpreter','none');
    
    hold(ax(1),'on');
    ylabel(ax(1), 'MSE/Var(Y)');
    plot(ax(1), s_best_mse, mse_s_best_mse / var_y, '*r');
    plot(ax(1), s_max, mse_s_max / var_y, 'xk');
    plot(ax(1), s_cross_var, mse_s_cross_var / var_y, 'sk');
    plot(ax(1), s_min_reg_err, mse_s_min_reg_err / var_y, 'vk');
    
    hold(ax(2),'on');
    ylabel(ax(2),'Var($$\hat y$$/Var(y)','interpreter','latex');
    plot(ax(2), s_best_mse, var_s_best_mse / var_y, '*r');
    plot(ax(2), s_max, var_s_max / var_y, 'xk');
    plot(ax(2), s_cross_var, var_s_cross_var / var_y, 'sk');
    plot(ax(2), s_min_reg_err, var_s_min_reg_err / var_y, 'vk');
    
    %legend([mat2cell(num2str(eigvals), ones(5,1));'Var(Y)']);
    axis(ax, 'tight');
    drawnow;

  
    %% Collect results
    fprintf('\nWeights:\t(oracle bias = %g)\n========\n',beta_oracle(1));
    format short e;
    disp(array2table([beta_oracle(2:end) w_pc v_1*s_best_mse, w_semi_s, v_1*s_cross_var, v_1*s_max], ...
        'VariableName',{'Oracle','PC','s_BestMSE','s_SemiSupervised', 's_cross_var','s_max'}))
    
    % Plot weights correlations between the different methods
%     if s_best_mse > 0  % The case where w_best = 0 kills corrplot, just bypass this problem for now
%         weightstable = array2table([beta_oracle(2:end) w_pc w_breiman w_pcrstar v_1*s_best_mse w_semi_s v_1*s_cross_var, v_1*s_max], ...
%                                    'VariableName',{'LR','PC','Breiman','PCRStar','BestMSE','SSemi', 'CrossVar','sMax'})
%         corrplot(weightstable);
%         set(gcf,'Name',[regexprep(regexprep(dataset_name{1},'.*auto\/',''),'\.mat','') '  (n=' num2str(n) ')']);
%     end;
    
    % Add the results of this iteration to the table
    format short; fprintf('\n');
    results_summary_current_dataset = ...
           {'n', n; 'var_y', var_y; ...
            'delta', mse_oracle / var_y; ...
            'oracle',   mse_oracle ; ...
            'rho_oracle', mse(y_rho_oracle) ; ...
            'PC',       mse(y_pc) ; ...
            'BreimanStar',   mse(y_breiman) ; ...
            'PCRStar',   mse(y_pcrstar) ; ...            
            'SemiSupervised_S_Selection', mse(y_semi_s); ...
            'SemiSupervisedComposition', mse(y_semi_comp); ...
            'Best',     mse(y_best) ; ...
            'MSE_s_minMSE', mse(y_SM_s_best_mse) ; ...            
            'Mean',     mse(y_mean) ; ...
            'VarW',     mse(y_varw) ; ...
            'rho_unidirectional',    mse(y_rho_uni) ; ...
            'MSE_s_max', mse(y_SM_s_max) ; ...            
            'MSE_s_cross_var', mse(y_SM_s_cross_var) ; ...
            'MSE_s_min_reg_err', mse(y_SM_s_min_reg_err) ; ...
            'MSE_s_w_sum_1', mse(y_SM_s_wsum1) ; ...            
            's_minMSE', s_best_mse ; ...           
            's_semi_s', s_semi_s; ...
            's_max', s_max ; ...            
            's_cross_var', s_cross_var ; ...
            's_min_reg_err', s_min_reg_err ; ...
            's_w_sum_1', s_wsum1 ; ...
            'VAR_s_minMSE', var_s_best_mse ; ...
            'VAR_s_max', var_s_max ; ...
            'VAR_s_cross_var', var_s_cross_var ; ...
            'VAR_s_min_reg_err', var_s_min_reg_err ; ...
            'VAR_s_w_sum_1', var_s_wsum1 ; ...
            'PCRstar_K', Kpcrstar ; ...
            };
        
    if ~exist('results_summary','var')
        results_summary = cell(length(datasets), size(results_summary_current_dataset,1));
    end;
    for i=1:length(results_summary_current_dataset)
        fprintf('%25s \t%g\n',results_summary_current_dataset{i,1}, results_summary_current_dataset{i,2});
        results_summary{find(strcmp(dataset_name,datasets)),i} = results_summary_current_dataset{i,2};
    end;

%%
end; % for datasets

cols = {results_summary_current_dataset{:,1}};
for i=1:length(cols); cols{i}=strrep(cols{i},' ','_');end;
out=cell2table(results_summary, 'RowNames', regexprep(regexprep(datasets,'.*auto\/',''),'\.mat','')', 'VariableNames', cols)
cell2table(results_summary', 'VariableNames', regexprep(regexprep(datasets,'.*auto\/',''),'\.mat','')', 'RowNames', cols)
writetable(out, ['results/s-tmp-' hostname '.csv'],'WriteRowNames',true)