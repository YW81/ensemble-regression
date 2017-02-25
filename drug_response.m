need to align with main.m
clear all; close all; close all hidden;
addpath Ensemble_Regressors;
addpath HelperFunctions;

%ROOT = './Datasets/RealWorld/CCLE/mydata/EC50/';
ROOT = './Datasets/RealWorld/CCLE/mydata/ActAreaWithVal/';
files = dir([ROOT '*.mat']);

m=7;
Wstar = zeros(m,length(files));
Wrstar = zeros(m,length(files));
Wnnstar = zeros(m,length(files));
Wnnsum1star = zeros(m,length(files));
MSE_orig = zeros(m,length(files));

results_summary = {};
for file_idx=1:length(files)
%for file_idx = [1]
    load([ROOT files(file_idx).name]);
    fprintf('DRUG FILE: %s\n', files(file_idx).name);
%%
    y_true = y;
    clear y;
    y_true = y_true - mean(y_true);
    Z = bsxfun(@minus, Z, mean(Z,2));
    [m n] = size(Z);
    Ey = mean(y_true);
    Ey2 = mean(y_true.^2);
    var_y = Ey2 - Ey.^2;
    mse = @(x) mean((y_true' - x).^2 / var_y);    

    for i=1:m
        MSE_orig(i,file_idx) = mse(Z(i,:)');
    end;
    
    %% Estimators
    [y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z); Wstar(:,file_idx) = w_oracle2;
    %[y_oracle_rho, w_oracle_rho] = ER_Oracle_Rho(y_true,Z); Wrstar(:,file_idx) = w_oracle_rho;
    %[y_oracle_nonneg, w_oracle_nonneg] = ER_Oracle_2_NonNegWeights(y_true,Z); Wnnstar(:,file_idx) = w_oracle_nonneg;
    %[y_oracle_nonnegsum1, w_oracle_nonnegsum1] = ER_Oracle_2_NonNegSum1Weights(y_true,Z); Wnnsum1star(:,file_idx) = w_oracle_nonnegsum1;
    [y_MEAN,w_mean] = ER_MeanWithBiasCorrection(Z, Ey);
    y_MED = ER_MedianWithBiasCorrection(Z, Ey);
    [y_DGEM,w_dgem] = ER_UnsupervisedDiagonalGEM(Z, Ey);
    %[y_gem,w_gem] = ER_UnsupervisedGEM(Z, Ey,Ey2);
    %[y_gem_with_rho_estimation,w_gem_with_rho_estimation] = ER_UnsupervisedGEM_with_rho_estimation(Z, Ey);
    [y_UPCR,w_upcr] = ER_SpectralApproachGivenDeltaStar(Z, Ey, Ey2, mse(y_oracle2));
    [y_UPCR_d0,w_upcr_d0] = ER_SpectralApproach(Z, Ey, Ey2);
    [y_UPCR_MRE,w_upcr_MRE] = ER_SpectralApproachDeltaMinMRE(Z, Ey, Ey2, mse(y_oracle2));
    [y_UPCR_WMRE,w_upcr_WMRE] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2, mse(y_oracle2));
    [y_UPCR_t1,w_upcr_t1] = ER_SpectralApproachWeightsSum1(Z, Ey, Ey2);
    %figure('Name',[files(file_idx).name ' Residuals']); 
%     [y_IND,w_ind] = ER_IndependentMisfits(Z,Ey, Ey2);
% 
%     figure('Name',files(file_idx).name); imagesc(cov(Z') ./ var_y); title(['Covariance \delta^*=' num2str(mse(y_oracle2))]); colorbar;
%     
%     g2_list = linspace(0,Ey2,100);
%     a_vec = zeros(m,length(g2_list));
%     rhoINDB = zeros(m,length(g2_list));
%     yINDB = zeros(n,length(g2_list));
%     wINDB = zeros(m,length(g2_list));
%     MSE = zeros(length(g2_list),1);
%     SCORE = zeros(length(g2_list),1);
%     for i=1:length(g2_list);
%         [yINDB(:,i), wINDB(:,i),rhoINDB(:,i),a_vec(:,i)] = ER_IndependentMisfitsBayes(y_true, Z, Ey, g2_list(i));
%         MSE(i) = mse(yINDB(:,i));
%         SCORE(i) = mean(mean((abs(Z - repmat(yINDB(:,i)',m,1)))));
%     end;
%     %figure(223); plot(g2_list/Ey2,log(sum(wINDB.^2))); grid on; 
%     figure(223); plot(g2_list/Ey2,log(SCORE)); grid on; 
%     [val g2_opt_indx] = min(SCORE); %min(sum(wINDB.^2)); 
%     figure(222); clf; set(gca,'fontsize',24); 
%     plot(g2_list/Ey2,MSE,'.-', ...
%         [0 1], [mse(y_oracle2),mse(y_oracle2)],'k--', ...
%         [0 1], [mse(y_UPCR),mse(y_UPCR)],'m--', ...
%         [0 1], [mse(y_IND),mse(y_IND)],'g--'); 
%     hold on; 
%     plot(g2_list(g2_opt_indx)/Ey2,MSE(g2_opt_indx),'rs','markersize',9); 
%     title('INDB MSE as a function of g_2'); xlabel('g_2'); ylabel('MSE / Var(Y)'); grid on;
%     axis([ 0 0.3 0 var_y]);     
%     pause;
%     
%     %[y_lrm,~,y_oracle_rho] = ER_LowRankMisfitCVX(Z, Ey, Ey2, y_true);
%     %[y_rank1,w_rank1,Cstar_rank1_offdiag] = ER_Rank1Misfit(Z,Ey, Ey2);
%     %[y_rank2,w_rank2,Cstar_rank2_offdiag] = ER_Rank2Misfit(Z,Ey, Ey2);

    %% Bayes Optimal Methods
    rho_true = mean(Z .* repmat(y_true,m,1),2);
    
    figure(300); clf; hold on; ylabel('ALL'); 
    [y_IND,w_IND,rho_IND] = ER_IndependentMisfits(Z,Ey, Ey2); 
    [y_INDB, w_INDB,rho_INDB, MSE_hat_INDB] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2,'l2',1);
    [inlier_idx,outlier_idx, MSE_ss] = subset_selection(y_true,Z,Ey,Ey2,'rho');
    [y_MEAN_ss, w_MEAN_ss] = ER_MeanWithBiasCorrection(Z(inlier_idx,:), Ey);
    [y_UPCRrhoINDB, w_UPCRrhoINDB] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_INDB);
    [y_UPCRrhoOracle, w_UPCRrhoOracle] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_true);
    figure(301); hold on; ylabel('SUBSET SELECTION'); 
    [y_INDB_ss, w_INDB_ss,rho_INDB_ss, ~] = ER_IndependentMisfitsBayes(y_true, Z(inlier_idx,:), Ey, Ey2,'l2',1);
    [y_UPCRrhoINDB_ss, w_UPCRrhoINDB_ss] = ER_UPCRgivenRho(Z(inlier_idx,:),Ey,Ey2,rho_INDB_ss);    

    figure(130); clf; set(gca,'fontsize',24); 
    plot(rho_true/var_y,rho_IND/var_y,'rs',rho_true/var_y,rho_true/var_y,'k-'); grid on; xlabel('RHO TRUE'); ylabel('RHO EST'); 
    hold on; 
    plot(rho_true/var_y,rho_INDB/var_y,'bo');
    plot(rho_true(outlier_idx)/var_y, rho_INDB(outlier_idx)/var_y,'k>','markersize',20);
    plot(rho_true(inlier_idx)/var_y, rho_INDB_ss/var_y,'cd');

    figure(400); plot(sort(eig(cov(Z')),'descend') / trace(cov(Z')), 'ko-');    

    %% Print results
    C = cov(Z'); 
    results = {files(file_idx).name, 'best',min(mean((Z - repmat(y_true,m,1)).^2,2)),median(sum(C)) / min(sum(C))}; % best individual regressor
    for alg=who('y_*')'
        if ~strcmp(alg{1}, 'y_true')
            results = [results; {files(file_idx).name, alg{1}, mse(eval(alg{1})), median(sum(C)) / min(sum(C))}];
        end;
    end;
    results_summary = [results_summary; results];
    
    results,
    fprintf('PAUSE\n'); pause;
    
    %% Plot principal components
%     figure('Name',files(i).name);
%     W = [w_oracle2 w_oracle_rho w_mean(2:end), w_dgem, w_gem(2:end), w_gem_with_rho_estimation, w_spectral];
%     [pc,score,latent,tsquare] = princomp(W);
%     biplot(pc(:,1:2),'Scores',score(:,1:2),'VarLabels',{'oracle2','oracle rho','mean', 'dgem', 'gem', 'gem with rho estimation', 'spectral'}, 'MarkerSize',10);

    %% Plot misfit covariance matrix
     Cstar = cov((Z - repmat(y_true,m,1))'); labels = {'1','2','3','4','5','6','7'};
     %figure('Name',[files(file_idx).name ' Cstar']); imagesc(Cstar); colorbar; title('Cstar');
%     Cstar = Cstar - Cstar_rank1_offdiag;
%     Cstar_norm = zeros(m); for i=1:m; for j=1:m; Cstar_norm(i,j) = Cstar(i,j) ./ sqrt(Cstar(i,i) * Cstar(j,j)); end; end;
%     a=HeatMap(Cstar_norm,'Colormap','redbluecmap','LabelsWithMarkers','true','DisplayRange',1, ...
%               'Symmetric','true','RowLabels',labels,'ColumnLabels',labels);
%     set(a,'Annotate','true'); addTitle(a,['Misfit Covariance C*_ij/sqrt(C*_ii C*_jj) - ' files(file_idx).name],'interpreter','none');
%     %addTitle(a,['Misfit Covariance After Rank2 reduction - ' files(file_idx).name],'interpreter','none');
end;
writetable(table(results_summary), 'results/drug_response.csv')

%% Best ensemble regression algorithm
% with oracle regressors (which requires oracle knowledge)
t =pivottable(results_summary,2,1,3,@sum);
best = min(cell2mat(t(2:end,2:end)));
a =cell2mat(t(2:end,2:end));
fprintf('\nWith oracle\n');
for i=1:length(files); fprintf('%s\n',t{find(a(:,i) == best(i))+1,1}); end;

% without oracle regressors (which requires oracle knowledge)
fprintf('\n\nWithout oracles\n');
t =pivottable(results_summary,2,1,3,@sum);
t(find(strcmp(t(:,1),'best')),:) = []; t(find(strcmp(t(:,1),'y_oracle2')),:) = []; 
t(find(strcmp(t(:,1),'y_oracle_rho')),:) = []; t(find(strcmp(t(:,1),'y_oracle_nonneg')),:) = [];
best = min(cell2mat(t(2:end,2:end)));
a =cell2mat(t(2:end,2:end));
for i=1:length(files); fprintf('%s\n',t{find(a(:,i) == best(i))+1,1}); end;
fprintf('\n');

p=pivottable(results_summary,2,1,3,@sum)
a=cell2mat(p(2:end,2:end))
p(:,1)


%%
% idx_orc = size(a,1);
% 
% figure(1); clf;  msize = 8; 
% set(gca,'fontsize',20); 
% hold on; grid on; 
% plot(a(idx_orc,:),a(4,:),'k>'); %mean
% plot(a(idx_orc,:),a(5,:),'b.','markersize',msize);   %median
% plot(a(idx_orc,:),a(2,:),'rs','markersize',msize);   %D-GEM
% plot(a(idx_orc,:),a(6,:),'md','markersize',msize);
% %plot(a(11,:),a(5,:),'gp','markersize',msize);  %INDEPENDENT ERRORS
% 
% legend('MEAN','MED','DGEM','U-PCR','Location','NorthWest'); 
% plot(a(idx_orc,:),a(idx_orc,:),'b-'); 
% plot(a(idx_orc,:),a(idx_orc-3,:),'bo','markersize',msize+2); 
% 
% axis([0 0.7 0 1]); 


%%
idx_orc = size(a,1); idx_mean = 4; idx_med = 5;
fig = 111;
figure(fig); clf;  msize = 8; 
set(gca,'fontsize',18); 
hold on; grid on; 
plot(a(idx_orc,:),a(idx_mean,:)-a(idx_orc,:),'ko','markerfacecolor','k'); %mean
plot(a(idx_orc,:),a(idx_med,:)-a(idx_orc,:),'k>','markersize',msize,'markerfacecolor','k');   %median
plot(a(idx_orc,:),a(2,:)-a(idx_orc,:),'d','markersize',msize,'linewidth',2,'markeredgecolor',[.9 0 0]);   %D-GEM
plot(a(idx_orc,:),a(3,:)-a(idx_orc,:),'v','markersize',msize,'linewidth',2,'markeredgecolor',[0 .7 0]);  %INDEPENDENT ERRORS
plot(a(idx_orc,:),a(6,:)-a(idx_orc,:),'ms','markersize',msize+1,'markerfacecolor','m');   %PCR given delta star*
plot(a(idx_orc,:),a(idx_orc-1,:)-a(idx_orc,:),'bp','markersize',msize+1,'markerfacecolor','b');   %PCR given sum(abs(w))=1
%plot(a(idx_orc,:),a(end-1,:),'bx','markersize',msize+1);   %PCR delta=MRE*
%plot(a(idx_orc,:),a(end,:),'bo','markersize',msize+1);   %PCR delta=WMRE*
%plot(a(idx_orc,:),a(end-3,:),'b*','markersize',msize+1);   %PCR delta=0
%plot(a(idx_orc,:),a(end-4,:),'p','markersize',msize);  %RANK-1

%legend('MEAN','MED','DGEM','IND','U-PCR','PCR \delta=MRE','PCR \delta=WMRE','PCR \delta=0','Location','NorthEast'); 
legend('MEAN','MED','DGEM','IND','U-PCR','U-PCR sum(|w|)=1','Location','NorthEast'); 
%plot(a(idx_orc,:),a(idx_orc,:),'b-'); 
xlabel('\delta_{OR}=MSE(oracle)/Var(Y)'); ylabel('MSE/Var(Y) - \delta_{or}');
%set(gca,'yscale','log'); grid minor; set(gca,'ytick',[.01 .1])

axis tight; xlim([0 1]);%axis([0 1 0 .5]); 
set(fig,'PaperPositionMode','auto');
set(fig,'Position',[574 656 1045 378]);
% axis([0.2 .95 0.2 1]); set(fig,'Position',[574 656 1045 578]); % for drug response data
set(gca,'Position', [.1 .22 .89 .70]);
%saveas(fig,'plots/rf50_results.fig','fig'); saveas(fig,'plots/rf50_results.eps','psc2');
