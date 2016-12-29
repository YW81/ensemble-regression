clear all; close all; close all hidden;
addpath Ensemble_Regressors
addpath Ensemble_Regressors/Missing_Values;
addpath HelperFunctions;
rng(0)
tic
%ROOT = './Datasets/RealWorld/CCLE/mydata/EC50/';
ROOT = './Datasets/final/misc/';
%ROOT = './Datasets/final/rf/';
%ROOT = './Datasets/final/mlp/';
%ROOT = './Datasets/final/mlp_different/';
files = dir([ROOT '*.mat']);
%files = struct(struct('name','ratings_of_sweets.mat'));

% Threasholds for m and n (min number of regressors and samples)
Tm = 5;
Tn = 5;

results_summary = {};
% for file_idx=1:length(files)
%     load([ROOT files(file_idx).name]);
%     fprintf('FILE: %s\n', files(file_idx).name);
filename = 'blog_feedback', delta = 0.3792, number_of_samples = 1000, % ONLY USE 1000 RANDOM SAMPLES FROM THE DATASET
%filename = 'online_videos', delta = 0.1034, number_of_samples = 1000, % ONLY USE 1000 RANDOM SAMPLES FROM THE DATASET
%filename = 'flights_AUS', delta = 0.3299, number_of_samples = 1000, % ONLY USE 1000 RANDOM SAMPLES FROM THE DATASET
load([ROOT filename '.mat']); idxs = randperm(size(Z,2),size(Z,2)-number_of_samples); Z(:,idxs) = []; y(idxs) = []; % leave only 500 samples
y_true = y;
clear y;
y_true = double(y_true) - mean(y_true);
%Z=Z(1:10,:);%Z = Z(1:2:end,:); %Z = Z(1:5,:); % Random Forest
Zbiased = Z;
Z = bsxfun(@minus, Z, mean(Z,2));
[m n] = size(Z);
Ey = mean(y_true);
Ey2 = mean(y_true.^2);
var_y = Ey2 - Ey.^2;
mse = @(x) nanmean((y_true' - x).^2 / var_y);    

if exist('y_RandomForest','var')
    y_RandomForest = y_RandomForest';
end;

Zfull = Z; Zbiasedorig = Zbiased;
sparsity_list = 0:.05:.8;
for sparsity = sparsity_list
    fprintf('sparsity: %0.2f\n',sparsity);
%%
    Z = Zfull;
    Z(randperm(numel(Z),round(numel(Z)*sparsity))) = nan;
    %% Estimators
    %[y_oracle2, w_oracle2] = MV_Oracle_2_Unbiased(y_true, Z, Tm, Tn); 
    %[y_oracle2, w_oracle2] = MV_Oracle_Rho(y_true, Z, Tm, Tn); 
    %y_biasedmean = nanmean(Zbiased)';
    y_MEAN = MV_MeanWithBiasCorrection(Z, Ey, Tm);
    y_MED = MV_MedianWithBiasCorrection(Z, Ey, Tm);
    y_DGEM = MV_UnsupervisedDiagonalGEM(Z, Ey, Tm);
    %y_gem = MV_UnsupervisedGEM(Z, Ey, Tm, Tn);
    y_UPCR = MV_UnsupervisedPCRstar(Z, Ey, Ey2, Tm, Tn, delta); 
    %y_pcrgivend = MV_UnsupervisedPCRgivendelta(y_true, Z, Ey, Ey2, Tm, Tn);
    figure; [y_UPCR_RE,y_UPCR_WRE] = MV_UnsupervisedPCRdeltaWRE(Z, Ey, Ey2,Tm,Tn);    
    y_SIE = MV_IndependentMisfits(Z,Ey, Ey2,Tm,Tn);
    %[y_rank1,w_rank1,Cstar_rank1_offdiag,rho_rank1] = ER_Rank1Misfit(Z,Ey, Ey2, Tm, Tn);

    %% Print results
    %y_best = min(nanmean((Z - repmat(y_true,m,1)).^2,2))/var_y;
    results = {};
    for alg=who('y_*')'
        if ~strcmp(alg{1}, 'y_true')
            results = [results; {filename, sparsity, alg{1}, mse(eval(alg{1})), sum(~isnan(eval(alg{1})))}];
        end;
    end;
    results_summary = [results_summary; results];
end;
writetable(table(results_summary), 'results/results.csv')

%% Best ensemble regression algorithm
% with oracle regressors (which requires oracle knowledge)
t =pivottable(results_summary,3,2,4,@sum);
best = min(cell2mat(t(2:end,2:end)));
a =cell2mat(t(2:end,2:end));
fprintf('\nWith oracle\n');
for i=1:length(sparsity_list); fprintf('%30s\t%s\n',filename, t{find(a(:,i) == best(i))+1,1}); end;

p=pivottable(results_summary,3,2,4,@sum)
a=cell2mat(p(2:end,2:end))
p(:,1)

%% for paper
fig=figure('Name',[filename ' with Missing Values']); 
% plot(sparsity_list, a,'s-'); 
% labels = p(2:end,1); for i=1:length(labels); labels{i} = strrep(labels{i},'y_',''); end;
% legend(labels,'interpreter','none'); 
subplot(3,1,1); % plot the number of predictions
    preds = pivottable(results_summary,2,3,5,@mean);
    preds = cell2mat(preds(2:end,:));
    plot(preds(:,1),mean(preds(:,2:end),2),'.-k'); 
    grid on; grid minor;     set(gca,'fontsize',16); 
    ylabel({'Number of','predictions'});
subplot(3,1,2:3);
msize = 8; set(gca,'fontsize',16); 
hold on; grid on; grid minor;
plot(sparsity_list,a(2,:),'-ko','markerfacecolor','k'); %mean
plot(sparsity_list,a(3,:),'-k>','markersize',msize,'markerfacecolor','k');   %median
plot(sparsity_list,a(1,:),'-d','markersize',msize,'linewidth',2,'color',[.9 0 0],'markeredgecolor',[.9 0 0]);   %D-GEM
plot(sparsity_list,a(5,:),'-ms','markersize',msize+1,'markerfacecolor','m');   %PCR given delta star*
%plot(sparsity_list,a(end-1,:),'-rx','markersize',msize);   %PCR delta=MRE*
plot(sparsity_list,a(end,:),'-^','markersize',msize+1,'markerfacecolor','r','color','m','markeredgecolor','m');   %PCR delta=WMRE*
plot(sparsity_list,a(4,:),'-v','markersize',msize,'linewidth',2,'color',[0 .7 0],'markeredgecolor',[0 .7 0]);  %INDEPENDENT ERRORS
%plot(sparsity_list,a(end-4,:),'-p','markersize',msize);  %RANK-1

legend({'MEAN','MED','DGEM','U-PCR($$\delta^*$$)','U-PCR($$\hat \delta$$)','SIE'},'Location','NorthWest','Interpreter','latex'); 
plot(sparsity_list,mse(ER_Oracle_2_Unbiased(y_true,Zfull))*ones(size(sparsity_list)),'b--'); 
xlabel('Sparsity'); ylabel('MSE/Var(Y)');

axis([0 max(sparsity_list) 0 2]);
set(fig,'PaperPositionMode','auto');
%set(fig,'Position',[574 656 1045 378]);
set(fig,'Position',[574 656 1045 578]);
set(gca,'Position', [.13 .11 .775 .5154]);
%saveas(fig,'plots/rf50_results.fig','fig'); saveas(fig,'plots/rf50_results.eps','psc2');


toc