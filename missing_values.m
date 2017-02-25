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

% for file_idx=1:length(files)
%     load([ROOT files(file_idx).name]);
%     fprintf('FILE: %s\n', files(file_idx).name);
%filename = 'blog_feedback', delta = 0.3792, number_of_samples = 1000, % ONLY USE 1000 RANDOM SAMPLES FROM THE DATASET
%filename = 'online_videos', delta = 0.1034, number_of_samples = 1000, % ONLY USE 1000 RANDOM SAMPLES FROM THE DATASET
filename = 'flights_AUS', delta = 0.3299, number_of_samples = 1000, % ONLY USE 1000 RANDOM SAMPLES FROM THE DATASET
load([ROOT filename '.mat']);
y_true = y;
clear y;
y_true = double(y_true) - mean(y_true);

Zpopulation = Z; y_true_population = y_true;
total_results = {}; 
num_of_sims = 1;
for sim_idx = 1:num_of_sims
    Z = Zpopulation; y_true = y_true_population;
    idxs = randperm(size(Z,2),size(Z,2)-number_of_samples); Z(:,idxs) = []; y_true(idxs) = []; % leave only 500 samples
    results_summary = {};


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
    fprintf('%% missing values: %d%%\n',round(sparsity*100));
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
    figure(1); [y_UPCR_RE,y_UPCR_WRE] = MV_UnsupervisedPCRdeltaWRE(Z, Ey, Ey2,Tm,Tn); %drawnow;
    y_IND = MV_IndependentMisfits(Z,Ey, Ey2,Tm,Tn);
    %[y_rank1,w_rank1,Cstar_rank1_offdiag,rho_rank1] = ER_Rank1Misfit(Z,Ey, Ey2, Tm, Tn);

    %% Print results
    %y_best = min(nanmean((Z - repmat(y_true,m,1)).^2,2))/var_y;
    results = {};
    algs=who('-regexp','^y_(?!true).'); % match all y_* that's not y_true*
    for i=1:length(algs)
        results = [results; {filename, sparsity, algs{i}, mse(eval(algs{i})), sum(~isnan(eval(algs{i})))}];
    end;
    results_summary = [results_summary; results];
end;
%writetable(table(results_summary), 'results/results.csv')

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
    plot(100*preds(:,1),mean(preds(:,2:end),2),'.-k'); 
    grid on; grid minor;     set(gca,'fontsize',16); 
    ylabel({'Number of','predictions'});
subplot(3,1,2:3);
msize = 8; set(gca,'fontsize',16); 
hold on; grid on; grid minor;
plot(100*sparsity_list,a(3,:),'-ko','markerfacecolor','k'); %mean
plot(100*sparsity_list,a(4,:),'-k>','markersize',msize,'markerfacecolor','k');   %median
plot(100*sparsity_list,a(1,:),'-d','markersize',msize,'linewidth',2,'color',[.9 0 0],'markeredgecolor',[.9 0 0]);   %D-GEM
plot(100*sparsity_list,a(2,:),'-v','markersize',msize,'linewidth',2,'color',[0 .7 0],'markeredgecolor',[0 .7 0]);  %INDEPENDENT ERRORS
plot(100*sparsity_list,a(5,:),'-ms','markersize',msize+1,'markerfacecolor','m');   %PCR given delta star*
%plot(100*sparsity_list,a(end-1,:),'-rx','markersize',msize);   %PCR delta=MRE*
%plot(100*sparsity_list,a(end,:),'-^','markersize',msize+1,'markerfacecolor','r','color','m','markeredgecolor','m');   %PCR delta=WMRE*
%plot(100*sparsity_list,a(end-4,:),'-p','markersize',msize);  %RANK-1

%legend({'MEAN','MED','DGEM','IND','U-PCR($$\delta^*$$)','U-PCR($$\hat \delta$$)'},'Location','NorthWest','Interpreter','latex'); 
legend({'MEAN','MED','DGEM','IND','U-PCR'},'Location','NorthWest','Interpreter','latex'); 
plot(100*sparsity_list,mse(ER_Oracle_2_Unbiased(y_true,Zfull))*ones(size(sparsity_list)),'b--'); 
xlabel('% Missing Values'); ylabel('MSE/Var(Y)');

axis([0 100*max(sparsity_list) 0 2]);
set(fig,'PaperPositionMode','auto');
%set(fig,'Position',[574 656 1045 378]);
set(fig,'Position',[574 656 1045 578]);
set(gca,'Position', [.13 .11 .775 .5154]);
%saveas(fig,'plots/rf50_results.fig','fig'); saveas(fig,'plots/rf50_results.eps','psc2');

%% end repeating simulations
    total_results = [total_results ; results_summary];
end;
pivottable(total_results,3,2,4,@sum)

%% PLOT WITH ERROR BARS
%a=pivottable(total_results, 2,3,[4 4 4],{@mean,@(x) (prctile(x,25)), @(x)(prctile(x,75))});
Y = pivottable(total_results, 2,3,4,@mean);
MED = pivottable(total_results, 2,3,4,@median);
L = pivottable(total_results, 2,3,4,@(x)(mean(x) - prctile(x,25)));
U = pivottable(total_results, 2,3,4,@(x)(prctile(x,75) - mean(x)));
X = 100*repmat(sparsity_list',1,size(Y,2)-1); spacing = linspace(-.005,.005,size(Y,2)-1);
X = bsxfun(@plus,X,spacing);
labels = Y(1,2:end);

Y = cell2mat(Y(2:end,2:end)); MED = cell2mat(MED(2:end,2:end)); 
L = cell2mat(L(2:end,2:end)); U = cell2mat(U(2:end,2:end)); 

%errorbar(X,Y,L,U,'.-'); hold on; plot(X,MED,'+'); hold off;
%legend(labels,'Location','NorthWest','Interpreter','None');
%grid on; grid minor; axis([-.1 1.1 0 1]);

fig=figure('Name',[filename ' with Missing Values']); 
% plot(sparsity_list, a,'s-'); 
% labels = p(2:end,1); for i=1:length(labels); labels{i} = strrep(labels{i},'y_',''); end;
% legend(labels,'interpreter','none'); 
subplot(3,1,1); % plot the number of predictions
    preds = pivottable(results_summary,2,3,5,@mean);
    preds = cell2mat(preds(2:end,:));
    plot(100*preds(:,1),mean(preds(:,2:end),2),'.-k'); 
    grid on; grid minor;     set(gca,'fontsize',16); 
    ylabel({'Number of','predictions'}); xlim([0 105]);
subplot(3,1,2:3);
msize = 8; set(gca,'fontsize',16); 
hold on; grid on; grid minor;
Y=Y';L=L';U=U';X=X';
errorbar(X(3,:),Y(3,:),L(3,:),U(3,:),'-ko','markerfacecolor','k'); %mean
errorbar(X(4,:),Y(4,:),L(4,:),U(4,:),'-k>','markersize',msize,'markerfacecolor','k');   %median
errorbar(X(1,:),Y(1,:),L(1,:),U(1,:),'-d','markersize',msize,'linewidth',2,'color',[.9 0 0],'markeredgecolor',[.9 0 0]);   %D-GEM
errorbar(X(2,:),Y(2,:),L(4,:),U(2,:),'-v','markersize',msize,'linewidth',2,'color',[0 .7 0],'markeredgecolor',[0 .7 0]);  %INDEPENDENT ERRORS
errorbar(X(5,:),Y(5,:),L(5,:),U(5,:),'-ms','markersize',msize+1,'markerfacecolor','m');   %PCR given delta star*
%errorbar(X(end-1,:),Y(end-1,:),L(end-1,:),U(end-1,:),'-rx','markersize',msize);   %PCR delta=MRE*
%errorbar(X(end,:),Y(end,:),L(end,:),U(end,:),'-^','markersize',msize+1,'markerfacecolor','r','color','m','markeredgecolor','m');   %PCR delta=WMRE*
%errorbar(X(end-4,:),Y(end-4,:),L(end-4,:),U(end-4,:),'-p','markersize',msize);  %RANK-1

%legend({'MEAN','MED','DGEM','U-PCR($$\delta^*$$)','U-PCR($$\hat \delta$$)','IND'},'Location','East','Interpreter','latex'); 
legend({'MEAN','MED','DGEM','IND','U-PCR'},'Location','East','Interpreter','latex'); 
plot(sparsity_list,mse(ER_Oracle_2_Unbiased(y_true,Zfull))*ones(size(sparsity_list)),'b--'); 
xlabel('% Missing Values'); ylabel('MSE/Var(Y)');

axis([0 105 0 max(Y(3,:)+U(3,:))]);
set(fig,'PaperPositionMode','auto');
%set(fig,'Position',[574 656 1045 378]);
set(fig,'Position',[574 656 1045 578]);
set(gca,'Position', [.13 .11 .775 .5154]);



%%
elapased_time = toc;
fprintf(2,'SIMULATION TOOK %02d:%02d\n',floor(elapased_time/60),floor(mod(elapased_time,60)));