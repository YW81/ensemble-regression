%% Init
clear all; close all; 
clc;
addpath './Ensemble_Regressors'
[~,hostname] = system('hostname'); hostname = strtrim(hostname);
%ROOT = './Datasets/auto_mlp5/';
%ROOT = './Datasets/auto/';
ROOT = './Datasets/auto_repeat/';
%ROOT = 'Datasets/auto_friedman1_test3/';
%ROOT = 'Datasets/auto_friedman1_test_corr/';
%ROOT = 'Datasets/mlp_friedman1_test_corr/';
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
        legend({'MSE / Var(Y)','Var($$\hat{y}$$) / Var(Y)', 'min MSE','$$t_{max}$$','Var($$\hat{y}$$) = Var(y)','min reg err'},'interpreter','latex','location','north');
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
var_y = Ey2 - Ey.^2;

% Z = Z([1 2 4 5 6],:); Ztrain = Ztrain([1 2 4 5 6],:); % work-around, remove NW kernel regression from ensemble
% Z(6,:) = Ey + 1000*var_y*randn(1,size(Z,2));
% Ztrain(6,:) = Ey + 10000*var_y*randn(1,size(Ztrain,2));

% clear Z Ztrain;
% Z([1 2 3],:) = [0.8;1.15;1.05]*y;
% Z(4:10,:) = Ey + .5*var_y*randn(7,length(y));
% Ztrain([1 2 3],:) = [0.8;1.15;1.05]*ytrain;
% Ztrain(4:10,:) = Ey + .5*var_y*randn(7,length(ytrain));

[m,n] = size(Z);
fprintf(['\n' dataset_name{1} '\t n = ' num2str(n) '\t var(y) = ' num2str(var_y) '\n']);

%% Oracle Prediction
[y_oracle, beta_oracle] = ER_linear_regression_oracle(y_true, Z);

%% Supervised Methods
[y_pc, w_pc, C_pc] = ER_PerroneCooperGEM(Ztrain, ytrain, Z);
[y_breiman, w_breiman, ~] = ER_BreimanStar(Ztrain, ytrain, Z);
[y_pcrstar,w_pcrstar, Kpcrstar] = ER_PCRstar(Ztrain, ytrain, Z); set(gcf,'Name',[regexprep(regexprep(dataset_name{1},'.*auto\/',''),'\.mat','') ' (n=' num2str(n) ')']);
y_semi_comp = ER_SemiSupervisedWeightsComposition(Ztrain, ytrain, Z);

mse_oracle = mean((y_true'-y_oracle).^2);
mse = @(x) (mean((y_true'-x).^2)) / var_y; %mse_oracle; % show MSE relative to oracle.

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
    [y_qd, w_qd] = ER_UnsupervisedDiagonalGEM(Z,Ey);
    [y_varw,beta_varw] = ER_VarianceWeightingWithBiasCorrection(Z,Ey);
    
    %% Calculate predictions dropping the assumption of a diagonal covariance matrix (assumes rho=c*ones(m,1) is uni-directional)
    [y_invc, beta_invc] = ER_UnsupervisedGEM(Z,Ey,Ey2);
    w_invc = beta_invc(2:end);
    
%     %% Calculate predictions with oracle rho
%     r_true = Z0*y_true' / n;
%     lambda_oracle_rho = ( (ones(1,m)*(C\r_true) ) - 1) / (ones(1,m)*(C\ones(m,1)));
%     w_rho_oracle = C \ (r_true - lambda_oracle_rho * ones(m,1));
%     y_rho_oracle = Ey + Z0' * w_rho_oracle;

    %% Calculate preditions with Low Rank Misfit Covariance Method
    [y_lrm, w_lrm, y_rho_oracle, w_rho_oracle] = ER_LowRankMisfitCVX(Z,Ey,Ey2,y_true);    
    
    %% Calculate predictions with Spectral Method (L2 regularization on w)
    [y_SM_t_max, w_tmax, t_max] = ER_SpectralApproach(Z,Ey,Ey2);
    mse_t_max = mse(y_SM_t_max);
    var_t_max = var(y_SM_t_max) / var_y;
    t_sign = sign(t_max);
    
    [v_1,lambda_1] = eigs(C,1,'lm'); % columns of V are possible w's. Take 5 largest magnitude eigvals
    % w = t*v_1 with s=t^2. Requiring sum(w)=1 means t=1/sum(v_1), use the correct sign for t.
    t_wsum1 = (1/sum(v_1)) * t_sign;
    y_SM_t_wsum1 = Ey + Z0' * (v_1 * t_wsum1);
    mse_t_wsum1 = mse(y_SM_t_wsum1);
    var_t_wsum1 = var(y_SM_t_wsum1);
    
    % scan s, find s at: minMSE, var_y=var(y_hat), 
    t_scan = linspace(-max(2,2*t_max),max(2,2*t_max),20); %400 % [0-max] 200
    scan_mse = zeros(size(t_scan)); scan_cross_var = zeros(size(t_scan)); 
    scan_avg_regressor_err = zeros(size(t_scan));
    y_SM_scan = zeros(n,numel(t_scan));
    for i=1:length(t_scan); 
        y_SM_scan(:,i) = Ey + Z0' * v_1 * t_scan(i); 
        scan_mse(i) = mse(y_SM_scan(:,i)); 
        scan_cross_var(i) = var(y_SM_scan(:,i));
        scan_avg_regressor_err(i) = mean(mean((Z0 - repmat(y_SM_scan(:,i)',m,1)).^2,2));
    end;
    t_best_mse = t_scan(scan_mse == min(scan_mse));
    t_min_reg_err = t_scan(scan_avg_regressor_err == min(scan_avg_regressor_err));
    %t_cross_var = t_scan(abs(var_y - scan_cross_var) == min(abs(var_y - scan_cross_var)));

    midway = floor(numel(scan_cross_var)/2);
    if t_min_reg_err < 0 % variance crossing can happen on the positive AND negative sides of t-axis. 
                         % Choose the one which presumes the average regressor isn't bad.
        t_cross_var = t_scan(find(abs(scan_cross_var(1:midway)-var_y) == min(abs(scan_cross_var(1:midway) - var_y))));
    else
        t_cross_var = t_scan(midway + find(abs(scan_cross_var(midway+1:end) - var_y) == min(abs(scan_cross_var(midway+1:end) - var_y))));
    end;
    
    y_SM_t_best_mse = y_SM_scan(:,t_scan == t_best_mse);
    y_SM_t_cross_var = y_SM_scan(:,t_scan == t_cross_var);
    y_SM_t_min_reg_err = y_SM_scan(:,t_scan == t_min_reg_err);
    
    mse_t_best_mse = mse(y_SM_t_best_mse);
    mse_t_cross_var = mse(y_SM_t_cross_var);
    mse_t_min_reg_err = mse(y_SM_t_min_reg_err);
    
    var_t_best_mse = var(y_SM_t_best_mse) / var_y;
    var_t_cross_var = var(y_SM_t_cross_var) / var_y;
    var_t_min_reg_err = var(y_SM_t_min_reg_err) / var_y;
    
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
%         fprintf('s_scan for v_2 = %g == 0 ?\n',s_scan_K2(min_j));[commandchars=\\\{\}]
%     end;
    

    %% Plotting
    %figure('Name',dataset_name{1});
    set(0,'currentfigure', fig);
    subplot(3,4,subplot_index);

    [ax,~,~] = plotyy(t_scan,scan_mse / var_y,t_scan, scan_cross_var / var_y); grid minor;
    xlabel('s');
    title([regexprep(regexprep(dataset_name{1},'.*auto\/',''),'\.mat','') '  (n=' num2str(n) ')'],'interpreter','none');
    
    hold(ax(1),'on');
    ylabel(ax(1), 'MSE/Var(Y)');
    plot(ax(1), t_best_mse, mse_t_best_mse / var_y, '*r');
    plot(ax(1), t_max, mse_t_max / var_y, 'xk');
    plot(ax(1), t_cross_var, mse_t_cross_var / var_y, 'sk');
    plot(ax(1), t_min_reg_err, mse_t_min_reg_err / var_y, 'vk');
    
    hold(ax(2),'on');
    ylabel(ax(2),'Var($$\hat y$$/Var(y)','interpreter','latex');
    plot(ax(2), t_best_mse, var_t_best_mse, '*r');
    plot(ax(2), t_max, var_t_max, 'xk');
    plot(ax(2), t_cross_var, var_t_cross_var, 'sk');
    plot(ax(2), t_min_reg_err, var_t_min_reg_err, 'vk');
    
    %legend([mat2cell(num2str(eigvals), ones(5,1));'Var(Y)']);
    axis(ax, 'tight');
    drawnow;
  
    %% Collect results
    fprintf('\nWeights:\t(oracle bias = %g)\n========\n',beta_oracle(1));
    format short e;
    disp(array2table([beta_oracle(2:end) w_pc v_1*t_best_mse, v_1*t_cross_var, v_1*t_max], ...
        'VariableName',{'Oracle','PC','s_BestMSE','s_cross_var','s_max'}))
    
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
            'cos_dist', pdist([w_tmax w_rho_oracle]','cosine'); ...
            'oracle',   mse_oracle ; ...
            'rho_oracle', mse(y_rho_oracle) ; ...
            'PC',       mse(y_pc) ; ...
            'BreimanStar',   mse(y_breiman) ; ...
            'PCRStar',   mse(y_pcrstar) ; ...            
            'SemiSupervisedComposition', mse(y_semi_comp); ...
            'Best',     mse(y_best) ; ...
            'MSE_t_minMSE', mse(y_SM_t_best_mse) ; ...            
            'Mean',     mse(y_mean) ; ...
            'QuickDirty',mse(y_qd) ; ...            
            'VarW',     mse(y_varw) ; ...
            'InvC',    mse(y_invc) ; ...
            'LRM',      mse(y_lrm); ...
            'Spectral', mse(y_SM_t_max) ; ...            
            'MSE_t_cross_var', mse(y_SM_t_cross_var) ; ...
            'MSE_t_min_reg_err', mse(y_SM_t_min_reg_err) ; ...
            'MSE_t_w_sum_1', mse(y_SM_t_wsum1) ; ...            
            't_minMSE', t_best_mse ; ...           
            't_max', t_max ; ...            
            't_cross_var', t_cross_var ; ...
            't_min_reg_err', t_min_reg_err ; ...
            't_w_sum_1', t_wsum1 ; ...
            'VAR_t_minMSE', var_t_best_mse ; ...
            'VAR_t_max', var_t_max ; ...
            'VAR_t_cross_var', var_t_cross_var ; ...
            'VAR_t_min_reg_err', var_t_min_reg_err ; ...
            'VAR_t_w_sum_1', var_t_wsum1 ; ...
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
out=cell2table(results_summary, 'RowNames', regexprep(regexprep(datasets,ROOT,''),'\.mat','')', 'VariableNames', cols)
writetable(out, ['results/s-tmp-' hostname '.csv'],'WriteRowNames',true)
cell2table(results_summary', 'VariableNames', regexprep(regexprep(datasets,ROOT,''),'\.mat','')', 'RowNames', cols)

%% Box Plots
RES = cell2mat(results_summary); % length(datasets) x number of variables (columns)
idx_oracle_mse = find(strcmpi(cols,'oracle'));
idx_varw = find(strcmpi(cols,'VarW'));
idx_invc = find(strcmpi(cols,'InvC'));
idx_tmax = find(strcmpi(cols,'Spectral'));
idx_lrm = find(strcmpi(cols,'LRM'));
%RES_minor = RES(:,[idx_varw idx_invc idx_tmax]);
data_type = regexprep(regexprep(datasets,ROOT,''),'_.*$',''); data_type = data_type{1};
rownames = regexprep(regexprep(regexprep(datasets,ROOT,''),'_iter\#\d+\.mat',''),[data_type '_overlap_'],'')';

%pos = str2num(char(rownames));
%pos = [pos; 10 + pos; 20 + pos];
%boxplot(RES_minor, repmat(rownames,3,1), 'positions', pos, 'colors','rgb')

pos = str2num(char(rownames));
figure; hold on;
% boxplot(RES(:,idx_varw) .* RES(:,idx_oracle_mse), pos, 'positions', pos + 3, 'widths', 1, 'colors', 'b');
boxplot(RES(:,idx_invc) .* RES(:,idx_oracle_mse), pos, 'positions', pos - 1, 'widths', 1, 'colors','r'); % for 'auto' C is singular
boxplot(RES(:,idx_tmax) .* RES(:,idx_oracle_mse), pos, 'positions', pos + 1, 'widths', 1, 'colors','k');
boxplot(RES(:,idx_lrm) .* RES(:,idx_oracle_mse), pos, 'positions', pos + 3, 'widths', 1, 'colors','b');
xlabel('overlap (#samples, n=1000)');  ylabel('actual MSE');
title({'Ensemble regression with increased dependency';['Friedman1 data, ''' data_type ''' ensemble with n=' num2str(n) ', and n_{train}=' num2str(size(Ztrain,2))]});
%legend('VarW','Spectral'); %set(gca,'xscale','log');
ylim([0 30]); grid on; grid minor;