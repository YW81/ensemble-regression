clear all; close all; close all hidden;
addpath Ensemble_Regressors;
addpath HelperFunctions;
rng(0)
ROOT = './Datasets/RealWorld/CCLE/mydata/EC50/';
%ROOT = './Datasets/final/misc/';
%ROOT = './Datasets/final/rf/';
%ROOT = './Datasets/final/mlp/';
%ROOT = './Datasets/final/mlp_different/';
files = dir([ROOT '*.mat']);
%files = struct(struct('name','ratings_of_sweets.mat'));

RHO_EST=zeros(3,length(files));
results_summary = {};
for file_idx=1:length(files)
    load([ROOT files(file_idx).name]);
    fprintf('FILE: %s\n', files(file_idx).name);
%%
    y_true = y;
    clear y;
    y_true = double(y_true) - mean(y_true);
    %Z=Z(1:10,:);%Z = Z(1:2:end,:); %Z = Z(1:5,:); % Random Forest
    Z = bsxfun(@minus, Z, mean(Z,2));
    [m n] = size(Z);
    Ey = mean(y_true);
    Ey2 = mean(y_true.^2);
    var_y = Ey2 - Ey.^2;
    mse = @(x) mean((y_true' - x).^2 / var_y);    
    
     if ~exist('Wstar','var')
        Wstar = zeros(m,length(files));
        Wrstar = zeros(m,length(files));
        Wnnstar = zeros(m,length(files));
        Wnnsum1star = zeros(m,length(files));
        Windepmisfit = zeros(m,length(files));
        Windepmisfitnn = zeros(m,length(files));
        Wrank1 = zeros(m,length(files));
        Wrank1nn = zeros(m,length(files));
        MSE_orig = zeros(m,length(files));
        VAR_orig = zeros(m,length(files));
        MN = zeros(2,length(files));
        VAR_Y = zeros(1,length(files));
        COS_DIST_V1_RHO = zeros(1,length(files));
        
        Rtrue = zeros(m,length(files));
        Rindepmisfit = zeros(m,length(files));
        Rindepmisfitnn = zeros(m,length(files));
        Rrank1 = zeros(m,length(files));
        Rrank1nn = zeros(m,length(files));        
    end;
    if exist('y_RandomForest','var')
        y_RandomForest = y_RandomForest';
    end;

    MN(:,file_idx) = size(Z)';
    VAR_Y(1,file_idx) = var_y;
    for i=1:m
        MSE_orig(i,file_idx) = mse(Z(i,:)');
        VAR_orig(i,file_idx) = var(Z(i,:)) / var_y;
    end;
    
    f=figure('Name',files(file_idx).name); suptitle({files(file_idx).name,''});
    
    %% Estimators
    [y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z); Wstar(:,file_idx) = w_oracle2;
    [y_oracle_rho, w_oracle_rho] = ER_Oracle_Rho(y_true,Z); Wrstar(:,file_idx) = w_oracle_rho;
    [y_oracle_nonneg, w_oracle_nonneg] = ER_Oracle_2_NonNegWeights(y_true,Z); Wnnstar(:,file_idx) = w_oracle_nonneg;
    [y_oracle_nonnegsum1, w_oracle_nonnegsum1] = ER_Oracle_2_NonNegSum1Weights(y_true,Z); Wnnsum1star(:,file_idx) = w_oracle_nonnegsum1;
    [y_mean,w_mean] = ER_MeanWithBiasCorrection(Z, Ey);
    y_median = ER_MedianWithBiasCorrection(Z, Ey);
    [y_dgem,w_dgem] = ER_UnsupervisedDiagonalGEM(Z, Ey);
    [y_gem,w_gem] = ER_UnsupervisedGEM(Z, Ey,Ey2);
    %[y_gem_with_rho_estimation,w_gem_with_rho_estimation] = ER_UnsupervisedGEM_with_rho_estimation(Z, Ey);
    [y_gem_with_rho_estimation,w_gem_with_rho_estimation] = ER_UnsupervisedGEM_with_rho_estimation_from_DGEM(Z, Ey);
    [y_spectral,w_spectral] = ER_SpectralApproach(Z, Ey, Ey2); subplot(222);
    [y_spectralgivend,w_spectralgivend] = ER_SpectralApproachGivenDeltaStar(Z, Ey, Ey2,mse(y_oracle2));
    [y_spectralminRE,w_spectralminRE] = ER_SpectralApproachDeltaMinMRE(Z, Ey, Ey2,mse(y_oracle2));    
    [y_spectralminWRE,w_spectralminWRE] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2,mse(y_oracle2));    
    [y_indepmisfit,w_indepmisfit,rho_indepmisfit] = ER_IndependentMisfits(Z,Ey, Ey2); Windepmisfit(:,file_idx) = w_indepmisfit; subplot(223);
    [y_indepmisfitnn,w_indepmisfitnn, rho_indepmisfitnn] = ER_IndependentMisfitsNonNegResiduals(Z,Ey, Ey2); Windepmisfitnn(:,file_idx) = w_indepmisfitnn; subplot(224);
%    [y_indepmisfitl1,w_indepmisfitl1] = ER_IndependentMisfits(Z,Ey, Ey2,'l1');
%    [y_indepmisfithuber,w_indepmisfithuber] = ER_IndependentMisfits(Z,Ey, Ey2,'huber');
    %[y_lrm,~,y_oracle_rho] = ER_LowRankMisfitCVX(Z, Ey, Ey2, y_true);
    %[y_rank1,w_rank1,Cstar_rank1_offdiag,rho_rank1] = ER_Rank1Misfit(Z,Ey, Ey2); Wrank1(:,file_idx) = w_rank1; 
    [y_rank1,w_rank1,Cstar_rank1_offdiag,rho_rank1] = ER_Rank1MisfitConstrainedRho(Z,Ey, Ey2, mse(y_oracle2)); Wrank1(:,file_idx) = w_rank1; 
    %[y_rank1nn,w_rank1nn,Cstar_rank1nn_offdiag,rho_rank1nn] = ER_Rank1MisfitNonNegResiduals(Z,Ey, Ey2); Wrank1nn(:,file_idx) = w_rank1nn; 
    %[y_rank2,w_rank2,Cstar_rank2_offdiag] = ER_Rank2Misfit(Z,Ey, Ey2);

    %% Print results
    results = {files(file_idx).name, 'best',min(mean((Z - repmat(y_true,m,1)).^2,2))/var_y}; % best individual regressor
    for alg=who('y_*')'
        if ~strcmp(alg{1}, 'y_true')
            results = [results; {files(file_idx).name, alg{1}, mse(eval(alg{1}))}];
        end;
    end;
    results_summary = [results_summary; results];

    %% Plot principal components
%     figure('Name',files(i).name);
%     W = [w_oracle2 w_oracle_rho w_mean(2:end), w_dgem, w_gem(2:end), w_gem_with_rho_estimation, w_spectral];
%     [pc,score,latent,tsquare] = princomp(W);
%     biplot(pc(:,1:2),'Scores',score(:,1:2),'VarLabels',{'oracle2','oracle rho','mean', 'dgem', 'gem', 'gem with rho estimation', 'spectral'}, 'MarkerSize',10);

    %% Plot misfit covariance matrix
%     labels = cellstr(num2str((1:m)'))';
%     Cstar = cov((Z - repmat(y_true,m,1))'); 
% %     Cstar = Cstar - Cstar_rank1_offdiag;
%     Cstar_norm = zeros(m); for i=1:m; for j=1:m; Cstar_norm(i,j) = Cstar(i,j) ./ sqrt(Cstar(i,i) * Cstar(j,j)); end; end;
%     a=HeatMap(Cstar_norm,'Colormap','redbluecmap','LabelsWithMarkers','true','DisplayRange',1, ...
%               'Symmetric','true','RowLabels',labels,'ColumnLabels',labels);
% %     a=HeatMap(corr(Z'),'Colormap','redbluecmap','LabelsWithMarkers','true','DisplayRange',1, ...
% %               'Symmetric','true','RowLabels',labels,'ColumnLabels',labels);
%     set(a,'Annotate','true'); 
%     %addTitle(a,['Regressor Correlation Matrix - ' files(file_idx).name],'interpreter','none');
%     addTitle(a,['Misfit Covariance C*_ij/sqrt(C*_ii C*_jj) - ' files(file_idx).name],'interpreter','none');
%     %addTitle(a,['Misfit Covariance After Rank2 reduction - ' files(file_idx).name],'interpreter','none');

    %% Heat Map of the residual to check whether Independent Misfit method can be applicable
    rho_true = mean(Z .* repmat(y_true,m,1),2); C = cov(Z');
    r=zeros(m);for i=1:m; for j=1:i-1; r(i,j) = (C(i,j)+Ey2-rho_true(i)-rho_true(j)) ./ Ey2; end;end;
    r=r+r';
%     labels = cellstr(num2str((1:m)'))';
% %     a=HeatMap(flip(r),'Colormap','redbluecmap','LabelsWithMarkers','true','DisplayRange',1, ...
% %               'Symmetric','true','RowLabels',flip(labels),'ColumnLabels',labels);
% %     close hidden;
% %     set(a,'Annotate','true'); set(a,'AnnotColor','k')
% %     addTitle(a,['Indep Misfit residual ' files(file_idx).name],'interpreter','none');
% %     h=plot(a);
%     
%     subplot(221); imagesc(r); colormap(hot); colorbar; title('Indep Misfit True Residual'); 
%     
%     %Cstar = cov((Z - repmat(y_true,m,1))')./Ey2; for i=1:m; Cstar(i,i)=0; end;
%     %norm(Cstar-r),
%     if any(r(:) < 0) 
%         fprintf(2,'FOUND ONE');
%     end;
%    
%     drawnow;
%     saveas(f,['plots/tmp/' files(file_idx).name '.png'],'png');        
    %%
    Rtrue(:,file_idx) = rho_true; Rindepmisfit(:,file_idx)=rho_indepmisfit; 
    Rindepmisfitnn(:,file_idx)=rho_indepmisfitnn; Rrank1(:,file_idx)=rho_rank1; %Rrank1nn(:,file_idx)=rho_rank1nn;
    RHO_EST(2,file_idx) = norm(rho_indepmisfit-rho_true);    
    RHO_EST(3,file_idx) = norm(rho_indepmisfitnn-rho_true);
    RHO_EST(4,file_idx) = norm(rho_rank1-rho_true);
    %RHO_EST(5,file_idx) = norm(rho_rank1nn-rho_true);
    fprintf('norm(residuals): indepmisfit: %.2f, indepmisfit-nn: %.2f, rank-1: %.2f, rank-1nn: %.2f', RHO_EST(:,file_idx))
    [v,d]=eig(C); COS_DIST_V1_RHO(1,file_idx) = 1-dot(v(:,end),rho_true./norm(rho_true));
end;
writetable(table(results_summary), 'results/results.csv')

%% Best ensemble regression algorithm
% with oracle regressors (which requires oracle knowledge)
t =pivottable(results_summary,2,1,3,@sum);
best = min(cell2mat(t(2:end,2:end)));
a =cell2mat(t(2:end,2:end));
fprintf('\nWith oracle\n');
for i=1:length(files); fprintf('%30s\t%s\n',files(i).name, t{find(a(:,i) == best(i))+1,1}); end;

% without oracle regressors (which requires oracle knowledge)
fprintf('\n\nWithout oracles\n');
t =pivottable(results_summary,2,1,3,@sum);
t(find(strcmp(t(:,1),'best')),:) = []; t(find(strcmp(t(:,1),'y_oracle2')),:) = []; 
t(find(strcmp(t(:,1),'y_oracle_rho')),:) = []; t(find(strcmp(t(:,1),'y_oracle_nonnegsum1')),:) = [];
t(find(strcmp(t(:,1),'y_oracle_nonneg')),:) = []; 
best = min(cell2mat(t(2:end,2:end)));
a =cell2mat(t(2:end,2:end));
for i=1:length(files); fprintf('%30s\t%s\n',files(i).name, t{find(a(:,i) == best(i))+1,1}); end;
fprintf('\n');

p=pivottable(results_summary,2,1,3,@sum)
a=cell2mat(p(2:end,2:end))
p(:,1)


%%
idx_orc = 12; idx_mean = idx_orc-5; idx_med = idx_orc-4;

figure(1); clf;  msize = 8; 
set(gca,'fontsize',20); 
hold on; grid on; 
plot(a(idx_orc,:),a(idx_mean,:),'k>'); %mean
plot(a(idx_orc,:),a(idx_med,:),'b.','markersize',msize);   %median
plot(a(idx_orc,:),a(2,:),'rs','markersize',msize);   %D-GEM
%plot(a(idx_orc,:),a(3,:),'md','markersize',msize);   %GEM direction 1
%plot(a(idx_orc,:),a(4,:),'gp','markersize',msize);   %GEM with rho estimation
%plot(a(idx_orc,:),a(end-1,:),'md','markersize',msize);   %PCR*
plot(a(idx_orc,:),a(end-2,:),'md','markersize',msize);   %PCR given delta star*
plot(a(idx_orc,:),a(end-1,:),'rx','markersize',msize);   %PCR delta=MRE*
plot(a(idx_orc,:),a(end,:),'ro','markersize',msize);   %PCR delta=WMRE*
plot(a(idx_orc,:),a(5,:),'g^','markersize',msize);  %INDEPENDENT ERRORS
plot(a(idx_orc,:),a(end-4,:),'p','markersize',msize);  %RANK-1
%plot(a(idx_orc,:),a(5:7,:),'p','markersize',msize);

%legend('MEAN','MED','DGEM','PCR','PCR given \delta^*','Location','NorthWest'); 
%legend('MEAN','MED','DGEM','GEM \propto 1','GEM with estimated \rho','Location','NorthWest'); 
legend('MEAN','MED','DGEM','PCR given \delta^*','PCR \delta=MRE','PCR \delta=WMRE','Indep Errors','Rank-1','Location','NorthWest'); 
plot(a(idx_orc,:),a(idx_orc,:),'b-'); 
plot(a(idx_orc,:),a(idx_orc-3,:),'bo','markersize',msize+2); 

%axis([0 0.5 0 0.8]); 

% remove ratings of sweets (since its ordinal, and not continuous)
a(:,end-1) = []; %a(2,:) = [];
%% for paper
idx_orc = 9; idx_mean = idx_orc-2; idx_med = idx_orc-1;
fig = 1;
figure(fig); clf;  msize = 8; 
set(gca,'fontsize',20); 
hold on; grid on; 
plot(a(idx_orc,:),a(idx_mean,:),'ko','markerfacecolor','k'); %mean
plot(a(idx_orc,:),a(idx_med,:),'k>','markersize',msize,'markerfacecolor','k');   %median
plot(a(idx_orc,:),a(2,:),'d','markersize',msize,'linewidth',2,'markeredgecolor',[.9 0 0]);   %D-GEM
plot(a(idx_orc,:),a(end-2,:),'ms','markersize',msize+1,'markerfacecolor','m');   %PCR given delta star*
%plot(a(idx_orc,:),a(end-1,:),'rx','markersize',msize);   %PCR delta=MRE*
%plot(a(idx_orc,:),a(end,:),'ro','markersize',msize);   %PCR delta=WMRE*
plot(a(idx_orc,:),a(5,:),'v','markersize',msize,'linewidth',2,'markeredgecolor',[0 .7 0]);  %INDEPENDENT ERRORS
%plot(a(idx_orc,:),a(end-4,:),'p','markersize',msize);  %RANK-1

% legend('MEAN','MED','DGEM','U-PCR','PCR \delta=MRE','PCR \delta=WMRE','Indep Errors','Rank-1','Location','NorthWest'); 
legend('MEAN','MED','DGEM','U-PCR','SIE','Location','SouthEast'); 
plot(a(idx_orc,:),a(idx_orc,:),'b-'); 
xlabel('\delta^*=MSE*/Var(Y)'); ylabel('MSE/Var(Y)');

axis([0 1 0 1]); 
set(fig,'PaperPositionMode','auto');
set(fig,'Position',[574 656 1045 378]);
% axis([0.2 .95 0.2 1]); set(fig,'Position',[574 656 1045 578]); % for drug response data
set(gca,'Position', [.08 .22 .91 .75]);
%saveas(fig,'plots/rf50_results.fig','fig'); saveas(fig,'plots/rf50_results.eps','psc2');
