when re-running this, remember that IND code was changed.
Now, after getting rho, instead of using U-GEM weights, it uses the optimal solution w=C\rho.

addpath Datasets
addpath ../Ensemble_Regressors/
addpath ../HelperFunctions//
clear all; close all;

%% Init
tic;
close all;
clear all;
%load ../Datasets/final/rf/friedman1.mat
%load RandomForestTest_Friedman1
%load RandomForestTest_Friedman2
%load RandomForestTest_Friedman3
%load ../Datasets/final/rf/diabetes.mat

full_results = {};

% ROOT = '../Datasets/final/rf/';
% files = dir([ROOT '*.mat']);%files = dir([ROOT '*.mat']); %diabetes, blog_feedback
ROOT = '../Datasets/final/repeat/rf/';
files = dir([ROOT 'blog_feedback*.mat']);%files = dir([ROOT '*.mat']); %abalone, flights_AUS, SP500
%load tmp.mat
for file_idx=1:length(files)
    load([ROOT files(file_idx).name]);
    fprintf('FILE: %s\n', files(file_idx).name);

y_true = double(y); clear y;
y_RandomForest = y_RandomForest';

var_y = Ey2 - Ey.^2;

[m,n] = size(Z);

%%
Zfull = Z; y_true_full = y_true;
m_min = 3;
m_max = m;

results_summary = {};
m_list = 3:50; %[3:2:10 10:10:50];
for m=m_list
    y_true = y_true_full(1,:);
    mse = @(x) (mean((y_true'-x).^2)) / var_y;
    Z = Zfull(1:m,:);

    %% Ensemble Methods
    [y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z);
    [y_oracle_rho, w_oracle_rho] = ER_Oracle_Rho(y_true,Z);
    [y_oracle_nonneg, w_oracle_nonneg] = ER_Oracle_2_NonNegWeights(y_true,Z);
    [y_oracle_nonnegsum1, w_oracle_nonnegsum1] = ER_Oracle_2_NonNegSum1Weights(y_true,Z);
    [y_mean,w_mean] = ER_MeanWithBiasCorrection(Z, Ey);
    y_median = ER_MedianWithBiasCorrection(Z, Ey);
    [y_dgem,w_dgem] = ER_UnsupervisedDiagonalGEM(Z, Ey);
    [y_gem,w_gem] = ER_UnsupervisedGEM(Z, Ey,Ey2);
    %[y_gem_with_rho_estimation,w_gem_with_rho_estimation] = ER_UnsupervisedGEM_with_rho_estimation(Z, Ey);
    [y_gem_with_rho_estimation,w_gem_with_rho_estimation] = ER_UnsupervisedGEM_with_rho_estimation_from_DGEM(Z, Ey);
    [y_spectral,w_spectral] = ER_SpectralApproach(Z, Ey, Ey2);
    [y_spectralgivend,w_spectralgivend] = ER_SpectralApproachGivenDeltaStar(Z, Ey, Ey2,mse(y_oracle2));
    [y_spectralminRE,w_spectralminRE] = ER_SpectralApproachDeltaMinMRE(Z, Ey, Ey2,mse(y_oracle2));    
    [y_spectralminWRE,w_spectralminWRE] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2,mse(y_oracle2));    
    [y_indepmisfit,w_indepmisfit,rho_indepmisfit] = ER_IndependentMisfits(Z,Ey, Ey2);
%    [y_indepmisfitnn,w_indepmisfitnn, rho_indepmisfitnn] = ER_IndependentMisfitsNonNegResiduals(Z,Ey, Ey2);
%    [y_indepmisfitl1,w_indepmisfitl1] = ER_IndependentMisfits(Z,Ey, Ey2,'l1');
%    [y_indepmisfithuber,w_indepmisfithuber] = ER_IndependentMisfits(Z,Ey, Ey2,'huber');
    %[y_lrm,~,y_oracle_rho] = ER_LowRankMisfitCVX(Z, Ey, Ey2, y_true);
    %[y_rank1,w_rank1,Cstar_rank1_offdiag,rho_rank1] = ER_Rank1Misfit(Z,Ey, Ey2); Wrank1(:,file_idx) = w_rank1; 
    %[y_rank1,w_rank1,Cstar_rank1_offdiag,rho_rank1] = ER_Rank1MisfitConstrainedRho(Z,Ey, Ey2, mse(y_oracle2)); Wrank1(:,file_idx) = w_rank1; 
    %[y_rank1nn,w_rank1nn,Cstar_rank1nn_offdiag,rho_rank1nn] = ER_Rank1MisfitNonNegResiduals(Z,Ey, Ey2); Wrank1nn(:,file_idx) = w_rank1nn; 
    %[y_rank2,w_rank2,Cstar_rank2_offdiag] = ER_Rank2Misfit(Z,Ey, Ey2);

        %% Calculate MSEs
    results = {m, 'best',min(mean((Z - repmat(y_true,m,1)).^2,2))/var_y}; % best individual regressor
    for alg=who('y_*')'
        if (~strcmp(alg{1}, 'y_true')) && (~strcmp(alg{1}, 'y_true_full'))
            results = [results; {m, alg{1}, mse(eval(alg{1}))}];
        end;
    end;
    results_summary = [results_summary; results];

end; % for m
% end; % for n

%% Plot Single Experiment
% t =pivottable(results_summary,1,2,3,@sum);
% fprintf('Column Names: \n'); t(1,2:end),
% a =cell2mat(t(2:end,2:end));
% 
% fig = figure('Name',files(file_idx).name); msize = 8; 
% set(gca,'fontsize',14); 
% hold on; grid on; 
% idx_end = size(a,2); idx_orc = 9; 
% idx_mean = idx_orc-2; idx_med = idx_orc-1; idx_pcr = idx_end-2; idx_dgem = 3; idx_ind=6; idx_best=1;
% x = m_list;
% plot(x,a(:,idx_mean),'ko-','markerfacecolor','k','color','k');
% plot(x,a(:,idx_med),'k>-','markersize',msize,'markerfacecolor','k','color','k');
% plot(x,a(:,idx_dgem),'d-','markersize',msize,'linewidth',2,'color',[.9 0 0],'markeredgecolor',[.9 0 0]);
% plot(x,a(:,idx_ind),'v-','markersize',msize,'linewidth',2,'color',[0 .7 0],'markeredgecolor',[0 .7 0]);
% plot(x,a(:,idx_pcr),'ms-','markersize',msize+1,'markerfacecolor','m','color','m');
% %plot(x,a(:,idx_best));
% legend('MEAN','MED','D-GEM','IND','U-PCR','Location','NorthEast'); 
% %plot(x,a(:,idx_orc),'b--');
% axis tight; %([0 50 .5 1]);
% 
% xlabel('number of trees, m'); ylabel('MSE/Var(Y)');
% set(fig,'PaperPositionMode','auto');
% set(fig,'Position',[574 656 1045 378]);
% set(gca,'Position', [.08 .22 .91 .75]);
% saveas(fig,['tmp/rf50/' files(file_idx).name '.fig'],'fig'); saveas(fig,['tmp/rf50/' files(file_idx).name '.jpg'],'jpg');
% 
% drawnow;

full_results = [full_results; results_summary];
end; 

%% Plot Repeating Experiment
t =pivottable(full_results,1,2,3,@median);
fprintf('Column Names: \n'); t(1,2:end),
a =cell2mat(t(2:end,2:end));

fig = figure('Name',files(file_idx).name); msize = 8; 
set(gca,'fontsize',14); 
hold on; grid on; 
idx_end = size(a,2); idx_orc = 9; 
idx_mean = idx_orc-2; idx_med = idx_orc-1; idx_pcr = idx_end-2; idx_dgem = 3; idx_ind=6; idx_best=1;
x = m_list;
plot(x,a(:,idx_mean),'ko-','markerfacecolor','k','color','k','markersize',msize+1);
plot(x,a(:,idx_med),'k>-','markersize',msize,'markerfacecolor','k','color','k');
plot(x,a(:,idx_dgem),'d-','markersize',msize,'linewidth',2,'color',[.9 0 0],'markeredgecolor',[.9 0 0]);
plot(x,a(:,idx_ind),'v-','markersize',msize,'linewidth',2,'color',[0 .7 0],'markeredgecolor',[0 .7 0]);
plot(x,a(:,idx_pcr),'ms-','markersize',msize+1,'markerfacecolor','m','color','m');
%plot(x,a(:,idx_best));
legend('MEAN','MED','D-GEM','IND','U-PCR','Location','NorthEast'); 
%plot(x,a(:,idx_orc),'b--');
axis tight; %([3 50 .55 .9]); %
%ylim([.55 .85])

xlabel('number of trees, m'); ylabel('Median MSE/Var(Y)');
set(fig,'PaperPositionMode','auto');
set(fig,'Position',[574 656 1045 378]);
set(gca,'Position', [.08 .22 .91 .75]);
saveas(fig,['tmp/rf50/' files(file_idx).name '.fig'],'fig'); saveas(fig,['tmp/rf50/' files(file_idx).name '.jpg'],'jpg');

%%
% cols = {results{:,1}};
% out=cell2table(results_summary, 'VariableNames', cols) % 'RowNames', cellstr(num2str([m_min:m_max]')), 
% 
% results_table = cell2mat(results_summary);
% 
% %% PLOTTING ONLY FOR THE FIRST 1000 SAMPLES (ONLY LOOP OVER M)
% figure('Name',Description); hold on;
% plt=plot(m_min:m_max,results_table(1:n_max/n:end,1:num_predictors),'x-'); 
% mrk={'o','s','x','d','v','^','*','+','>'}.'; % <- note: transpose...
% col=num2cell(jet(num_predictors),2);
% set(plt,{'marker'},mrk,{'markerfacecolor'},col,{'Color'},cellfun(@(x){x*.5},col));
% 
% for m=m_min:m_max; rnd_frst_mse(m-m_min+1) = mean((mean(Z(1:m,:)) - y_true).^2) / var_y; end;
% hold on; plot(m_min:m_max,rnd_frst_mse,'k--');
% 
% grid minor; axis auto; legend([cols(1:num_predictors), 'Random Forest (without bias correction)']);
% %title('\bf Random Forest: Comparing Unsupervised Ensemble Methods');
% xlabel('Number of Decision Trees in the ensemble (m)');
% ylabel('MSE / Var(Y)'); ylim([0 1]);
% hold off;
% 
% T = get(gca,'tightinset'); set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]);
% 
% %% BOX PLOTS - M PREDICTORS, 10 ITERATIONS PER METHOD
% figure; hold on;
% 
% colors = num2cell(jet(num_predictors),2);
% for i=1:num_predictors
%     boxplot(results_table(:,i), results_table(:,find(strcmp(results(:,1),'n'))), 'positions', results_table(:,find(strcmp(results(:,1),'n'))) -.25 + i*.05, 'widths', 0.05, 'colors', .5*colors{i},'outliersize',4,'symbol','x');
% end;
% title('\bf Random Forest: Comparing Unsupervised Ensemble Methods');
% xlabel('Number of Decision Trees in the ensemble (m)');
% ylabel('MSE / Var(Y)');
% ylim([0 1]); grid minor;
% hLegend = legend(findall(gca,'Tag','Box'), cols(1:num_predictors));
%     % Among the children of the gca, find all boxes
%     box_vars = findall(gca,'Tag','Box');
% 
%     % Set the horizontal lines to the right colors
%     for i=1:num_predictors
%         idx = find(strcmp(cols, box_vars(i).DisplayName));
%         set(box_vars(idx),'Color',colors{i}*.5)
%     end;

toc