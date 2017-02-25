addpath ../Ensemble_Regressors/
clear all; close all;

% define prank: pseudo-rank returns the rank of a matrix containing 95% of the variability
% LRM is a good approach when prank(Cstar) is low.
iif = @(varargin) varargin{2 * find([varargin{1:2:end}], 1, 'first')}(); % inline-if. Refer to http://blogs.mathworks.com/loren/2013/01/10/introduction-to-functional-programming-with-anonymous-functions-part-1/
prank = @(X) iif( any(isnan(X(:))), @() Inf,  ...
                  true,             @() find(cumsum(flipud(eig(X))) / sum(eig(X)) > .95,1));

% VARIOUS REGRESSORS MATLAB EXAMPLE
%% Init
tic;
[~,hostname] = system('hostname'); hostname = strtrim(hostname);
%ROOT = './Datasets/ManualEnsembleDatasets/10MLP/';
%ROOT = './Datasets/auto_repeat/';
ROOT = './Datasets/auto_large/';

file_list = dir([ROOT '*.mat']);
datasets = cell(1,length(file_list));
for i=1:length(file_list)
    if (~strcmp(file_list(i).name, 'n_samples_10000000.mat')) && (~strcmp(file_list(i).name, 'MVGaussianDepData.mat'))
        datasets{i}=[ROOT file_list(i).name];
    end;
end;
datasets = datasets(~cellfun('isempty',datasets)); % Call Built-in string

%% Load dataset
%datasets = {datasets{9}};

%% For each dataset
file_idx = 0; m = 9;
Wstar = zeros(m,length(file_list)); Wrstar = zeros(m,length(file_list));
Wnnstar = zeros(m,length(file_list)); Wnnsum1star = zeros(m,length(file_list));
MSE_orig = zeros(m,length(file_list));
VAR_orig = zeros(m,length(file_list));

for dataset_name=datasets
    file_idx = file_idx + 1;
    load(dataset_name{1}); y=double(y);
    dataset_str = regexprep(regexprep(dataset_name{1},ROOT,''),'.mat','');
    rng;

    idxs=randperm(length(y));
    y = y(idxs);
    %filter max(MSE) > 10, filter min(var) < 1e-5
    Z = Z([3:7 9:end],idxs); %Z = Z(:,idxs);
    Z = Z([1 2 4 5 8:end],:);
    
    y_true = y; clear y;
    y_BiasedMean = mean(Z); %Z = Z(randperm(size(Z,1)),:);
    y_BiasedMean = y_BiasedMean';

    var_y = Ey2 - Ey.^2;
    [m,n] = size(Z);

    %%hold on; plot(1:floor(n_max/n_spacing),biased_mean_mse,'k--');
    Zfull = Z; y_true_full = y_true;
    n_max = n;

    %n = 2000; n_spacing = n; % run on every 2000 samples
    n = n_max; n_spacing = n; % run once for every dataset
    avg_prankCstar = 0; avg_cosDist = 0;
    for n_start = 1:n_spacing:n_max

        n_stop = n_start+n-1;
        if n_stop > n_max
            break;
        end;
            
        y_true = y_true_full(1,n_start:n_stop);
        Z = Zfull(1:m,n_start:n_stop);

        mse = @(x) (mean((y_true'-x).^2)) / var_y;
        
        for i=1:m
            MSE_orig(i,file_idx) = mse(Z(i,:)');
            VAR_orig(i,file_idx) = var(Z(i,:)') /var_y;
        end;

        %% Ensemble Methods
        % Oracle
        [y_oracle, beta_oracle] = ER_linear_regression_oracle(y_true, Z); 
        [y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z);Wstar(:,file_idx) = w_oracle2;
        [y_oracle_rho, w_oracle_rho] = ER_Oracle_Rho(y_true, Z);Wrstar(:,file_idx) = w_oracle_rho;
        [y_oracle_nn, w_oracle_nn] = ER_Oracle_2_NonNegWeights(y_true, Z);Wnnstar(:,file_idx) = w_oracle_nn;
        [y_oracle_nnsum1, w_oracle_nnsum1] = ER_Oracle_2_NonNegSum1Weights(y_true, Z);Wnnsum1star(:,file_idx) = w_oracle_nnsum1;

        % Calculate mean predictions
        [y_mean, beta_mean] = ER_MeanWithBiasCorrection(Z, Ey);
        y_median = ER_MedianWithBiasCorrection(Z,Ey);

        % Find best predictor in the ensemble
        [y_best, w_best] = ER_BestRegressor(y_true, Z);

        % Calculate varw predictions
        [y_udgem, w_udgem] = ER_UnsupervisedDiagonalGEM(Z,Ey);
        [y_udgem2, w_udgem2] = ER_UnsupervisedDiagonalGEMTwice(Z,Ey);
        [y_varw,beta_varw] = ER_VarianceWeightingWithBiasCorrection(Z,Ey);

        % Calculate predictions dropping the assumption of a diagonal covariance matrix (assumes rho=c*ones(m,1) is uni-directional)
        [y_ugem, beta_ugem] = ER_UnsupervisedGEM(Z,Ey,Ey2); w_ugem = beta_ugem(2:end);
        [y_ugem_with_rho_est, w_ugem_with_rho_est] = ER_UnsupervisedGEM_with_rho_estimation(Z,Ey);
        [y_ugem_with_rho_est_from_dgem, w_ugem_with_rho_est_from_dgem] = ER_UnsupervisedGEM_with_rho_estimation_from_DGEM(Z,Ey);

        % Calculate predictions with Spectral Method (L2 regularization on w)
        [y_sm, w_sm, t_sm] = ER_SpectralApproach(Z,Ey,Ey2);

        % Calculate predictions with rank-1 misfit covariance method
        [y_rank1, w_rank1] = ER_Rank1Misfit(Z,Ey,Ey2);
        
        % Calculate predictions for the low rank misfit covariance method
        %[y_lrm, w_lrm, y_rho_oracle, w_rho_oracle] = ER_LowRankMisfit( Z, Ey, Ey2, y_true );
        %[y_lrmcvx, w_lrmcvx, y_rho_oracle2, w_rho_oracle2,Cstar] = ER_LowRankMisfitCVX( Z, Ey, Ey2, y_true );
        
%         % Calculate predictions for matrix of preset low ranks
%         try
%             y_rank1m = Inf * ones(size(y_true))';
%             y_rank2m = Inf * ones(size(y_true))';
%             y_rank3m = Inf * ones(size(y_true))';
%             y_rank4m = Inf * ones(size(y_true))';            
%             [y_rank1m, w_rank1m] = ER_Rank1Misfit( Z, Ey, Ey2, y_true);
%             [y_rank2m, w_rank2m] = ER_Rank1Misfit( Z, Ey, Ey2, y_true, 2);
%             %[y_rank3m, w_rank3m] = ER_Rank1Misfit( Z, Ey, Ey2, y_true, 3);
%             %[y_rank4m, w_rank4m] = ER_Rank1Misfit( Z, Ey, Ey2, y_true, 4);
%             %fprintf('Low-Rank weights (1,2,3,4):\n');
%             %disp([w_rank1m, w_rank2m, w_rank3m, w_rank4m]);
%             fprintf('Low-Rank weights (1,2):\n');
%             disp([w_rank1m, w_rank2m]);
%             
%         catch ME
%             disp(ME.identifier);
%         end

        %% Calculate MSEs
        num_predictors = 15;
        results = { ...
            'Oracle', mse(y_oracle); ...
            'OracleRho', mse(y_oracle_rho); ...        
            'OracleNN', mse(y_oracle_nn); ...
            'OracleNNSum1', mse(y_oracle_nnsum1); ...
            'Best', mse(y_best); ...
            'Mean', mse(y_mean); ...            
            'Median', mse(y_median); ...                        
            'uDGEM', mse(y_udgem); ...
            'uDGEM2', mse(y_udgem2); ...
            'uGEM', mse(y_ugem); ...
            'uGEMwithRhoEst', mse(y_ugem_with_rho_est); ...            
            'uGEMwithRhoEstFromDGEM', mse(y_ugem_with_rho_est_from_dgem); ...
            'uPCRstar', mse(y_sm); ...
            'Rank1', mse(y_rank1); ...
            %'LowRankMisfit', mse(y_lrm); ...
            %'LowRankMisfitCVX', mse(y_lrmcvx); ...            
            'BiasedMean', mse(y_BiasedMean(n_start:n_stop)); ...
            %'Rank1Misfit', mse(y_rank1m); ...
            %'Rank2Misfit', mse(y_rank2m); ...
            %'Rank3Misfit', mse(y_rank3m); ...
            %'Rank4Misfit', mse(y_rank4m); ...            

            'n', n; ...
            'm', m; ...
            'delta', mse(y_oracle)/var_y; ...
            'SMCosDist', pdist([w_sm w_oracle_rho]','cosine'); ...
%            'LRMApproxRankCstar', getApproxRank(Cstar); ... 
            'Dataset', dataset_str
            }; 
    %%
%        avg_prankCstar = avg_prankCstar + prank(Cstar)/floor(n_max/n_spacing);
        avg_cosDist = avg_cosDist + pdist([w_sm w_oracle_rho]','cosine') / floor(n_max/n_spacing);

        if ~exist('results_summary','var')
            %results_summary = cell(length(1:n_spacing:n_stop) * , size(results,1));
            current_results = 1;
        else
            current_results = current_results+1;
        end;
        for i=1:length(results)
            %results_summary{floor(n_start / n_spacing)+1,i} = results{i,2};
            results_summary{current_results,i} = results{i,2};
        end;

    end; % for n
    
    results_table = cell2mat(results_summary(:,1:find(strcmp(results(:,1),'n'))));
    wanted_results = results_table(strcmp(results_summary(:,find(strcmp(results(:,1),'Dataset'))), dataset_str),:);

    %% BOX PLOT FOR CURRENT DATASET
%     if size(wanted_results,1) < 5
%         continue
%     end;
% 
%     figure('Name',dataset_str); hold on;
%     labels = {'Oracle', 'Oracle Rho', 'Best', 'Mean', 'Uns. D-GEM','Uns. GEM','Uns. PCR*', 'Low Rank Misfit', 'LRM CVX','Biased Mean'};
%     idxs = [1 2 3 7 4 5 6 8 9 10]; idxs = idxs(end:-1:1);
%     colors = num2cell(jet(num_predictors),2);
%     boxplot(wanted_results(:,idxs),'labels',labels(idxs),'widths',.5,'orientation','horizontal'); 
%     title({[dataset_str ': Manually Constructed Regressors']; ...
%            ['m = ' num2str(m) ', n = ' num2str(n) ', delta*=' num2str(mse(y_oracle)/var_y) ', ' num2str(floor(n_max / n_spacing)) ' iterations']; ...
%            ['mean(cosDist)=' num2str(avg_cosDist) ', mean(prank(Cstar))=' num2str(avg_prankCstar)]}, ...
%            'interpreter','none');
%     ylabel('MSE / Var(Y)'); grid on; grid minor; %ylim([0,1]);    
%     %set(gca,'xscale','log');
%     %ticks = [0:.05:.45 .5:.5:3];
%     %set(gca,'xtick',ticks); set(gca,'XTickLabel',ticks)
%     T = get(gca,'tightinset'); set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]);
%     hold off;
%     pause(0.5);
end; % for dataset
    
%%
cols = {results{1:end-1,1}};
%out=cell2table(results_summary(:,1:end-1), 'VariableNames', cols) % 'RowNames', cellstr(num2str([m_min:m_max]')), 
out = cell2table(results_summary(:,1:end), 'VariableNames', {results{1:end,1}}')

results_table = cell2mat(results_summary(:,1:end-1));
writetable(out);

%% Plot uPCR* MSE vs cos-dist
figure('Name','Cos-dist vs MSE');
[vals,idxs] = sort(out{:,'SMCosDist'});
plot(vals,out{idxs,'uPCRstar'},'x-')
xlabel('Cos-dist.'); ylabel('MSE(Uns. PCR*) / Var(Y)');
grid on; 

%% PLOTTING ONLY FOR THE FIRST 1000 SAMPLES (ONLY LOOP OVER M)
% figure; hold on; 
% plt=plot(1:size(results_table,1),results_table(1:size(results_table,1),1:num_predictors),'x-'); 
% mrk={'o','s','x','d','v','^','*','+','>','<','p','h','.','.'}.'; % <- note: transpose...
% mrk=mrk(1:num_predictors);
% col=num2cell(jet(num_predictors),2);
% set(plt,{'marker'},mrk,{'markerfacecolor'},col,{'Color'},cellfun(@(x){x*.5},col));
% 
% if n_spacing < n_max
%     for n_start=1:n_spacing:n_max; n_stop=n_start+n-1; if n_stop > n_max; break; end; biased_mean_mse(floor(n_start/n_spacing)+1) = mean((mean(Zfull(:,n_start:n_stop)) - y_true_full(n_start:n_stop)).^2) / var_y; end;
%     plot(1:floor(n_max/n_spacing),biased_mean_mse,'k--');
%     legend([cols(1:num_predictors), 'Regressor mean (without bias correction)']);    
% else
%     legend(cols(1:num_predictors));    
% end;
% 
% grid on; grid minor; axis auto; 
% xlabel('Iteration');
% ylabel('MSE / Var(Y)'); ylim([0 1]);
% hold off;
% pause(1.0);

%% BOX PLOTS - M PREDICTORS, 10 ITERATIONS PER METHOD
% figure; hold on;
% 
% colors = num2cell(jet(num_predictors),2);
% for i=1:num_predictors
%     boxplot(results_table(:,i), results_table(:,find(strcmp(results(:,1),'n'))), 'positions', results_table(:,find(strcmp(results(:,1),'n'))) -25 + i*5, 'widths', 5, 'colors', .8*colors{i},'outliersize',4,'symbol','x');
%     pause(0.5);
% end;
% title('\bf Ridge Regressors: Comparing Unsupervised Ensemble Methods');
% xlabel('Number of sampels (n)');
% ylabel('MSE / Var(Y)');
% ylim([0 6.5]); grid minor;
% xlim([min(results_table(:,num_predictors+1)) - 25, max(results_table(:,num_predictors+1)) + num_predictors*5]);
% drawnow; pause(0.5);
% 
% % Among the children of the gca, find all boxes
% box_vars = findall(gca,'Tag','Box');
% 
% hLegend = legend(box_vars(1:num_predictors), cols(1:num_predictors));
% 
% 
%     % Set the horizontal lines to the right colors
%     for i=1:num_predictors
%         %idx = find(not(cellfun('isempty', strfind(cols, box_vars(i).DisplayName))));
%         idx = find(strcmp(cols, box_vars(i).DisplayName));
%         set(box_vars(idx),'Color',colors{i}*.8)
%     end;
%     
% hold off;
% 
toc;
Wnnsum1star( abs(Wnnsum1star) < 1e-10) = 0;
a=results_table(:,1:4)'
x=a(1,:);
%figure(5); clf; plot(x,x,'b-',x,a(2,:),'rs',x,a(3,:),'ko',x,a(4,:),'mp','markersize',10);

out.Properties.VariableNames,
A = results_table(:,1:num_predictors)
x = A(:,1)
