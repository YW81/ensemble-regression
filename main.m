clear all; close all; close all hidden;
addpath Ensemble_Regressors;
addpath HelperFunctions;
rng(0)
filename_contains_index = false;
%ROOT = './Datasets/RealWorld/CCLE/mydata/EC50/';
ROOT = './Datasets/final/misc/';
%ROOT = './Datasets/final/rf/';
%ROOT = './Datasets/final/mlp/';
%ROOT = './Datasets/final/mlp_different/';
%ROOT = './Datasets/final/repeat/misc/'; filename_contains_index = true;
files = dir([ROOT '*.mat']);
files = struct(struct('name','ccpp.mat'));

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
    C = cov(Z');
    [v_1 lambda_1] = eigs(C,1,'lm'); 
    
    mse = @(x) mean((y_true' - x).^2 / var_y);    
    rho_true = mean(Z .* repmat(y_true,m,1),2);
    mse_true = zeros(m,1); 
    for i=1:m 
        mse_true(i) = mse(Z(i,:)');
    end

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
        VAR_Y = zeros(length(files),1);
        COS_DIST_V1_RHO = zeros(length(files),1);
        G2_EST = zeros(length(files),1);
        EIGVALS = zeros(m,length(files));
        
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
    VAR_Y(file_idx) = var_y;
    for i=1:m
        MSE_orig(i,file_idx) = mse(Z(i,:)');
        VAR_orig(i,file_idx) = var(Z(i,:)) / var_y;
    end;
    
    %f=figure('Name',files(file_idx).name); suptitle({files(file_idx).name,''});
    
    %% Estimators
    [y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z); Wstar(:,file_idx) = w_oracle2;
%    [y_oracle_rho, w_oracle_rho] = ER_Oracle_Rho(y_true,Z); Wrstar(:,file_idx) = w_oracle_rho;
%    [y_oracle_nonneg, w_oracle_nonneg] = ER_Oracle_2_NonNegWeights(y_true,Z); Wnnstar(:,file_idx) = w_oracle_nonneg;
%    [y_oracle_nonnegsum1, w_oracle_nonnegsum1] = ER_Oracle_2_NonNegSum1Weights(y_true,Z); Wnnsum1star(:,file_idx) = w_oracle_nonnegsum1;
    [y_MEAN,w_MEAN] = ER_MeanWithBiasCorrection(Z, Ey);
    y_MED = ER_MedianWithBiasCorrection(Z, Ey);
%    [y_DGEM,w_DGEM] = ER_UnsupervisedDiagonalGEM(Z, Ey);
%    [y_gem,w_gem] = ER_UnsupervisedGEM(Z, Ey,Ey2);
    %[y_gem_with_rho_estimation,w_gem_with_rho_estimation] = ER_UnsupervisedGEM_with_rho_estimation(Z, Ey);
    %[y_gem_with_rho_estimation,w_gem_with_rho_estimation] = ER_UnsupervisedGEM_with_rho_estimation_from_DGEM(Z, Ey);
%    [y_UPCRd0,w_UPCRd0] = ER_SpectralApproach(Z, Ey, Ey2);% subplot(222);
%    [y_UPCRt1,w_UPCRt1] = ER_SpectralApproachWeightsSum1(Z, Ey, Ey2);% subplot(222);
%    [y_UPCR,w_UPCR,~,rho_UPCR] = ER_SpectralApproachGivenDeltaStar(Z, Ey, Ey2,mse(y_oracle2));
%    [y_UPCRminRE,w_UPCRminRE] = ER_SpectralApproachDeltaMinMRE(Z, Ey, Ey2,mse(y_oracle2));    
%    [y_UPCRest_g2,w_UPCRest_g2] = ER_SpectralApproachDeltaMinMRE(Z, Ey, Ey2,(Ey2-estimate_g2(Z,Ey,Ey2))/var_y);    
%    [y_UPCRminWRE,w_UPCRminWRE] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2,mse(y_oracle2));    
%    [y_pcrnew,w_pcrnew] = ER_PCRtest_k_eigs(Z, Ey, Ey2, mse(y_oracle2));% subplot(222);
%    [y_IND,w_IND,rho_IND] = ER_IndependentMisfits(Z,Ey, Ey2); Windepmisfit(:,file_idx) = w_IND; %subplot(223);
%    [yIND_L1,wIND_L1,rhoIND_L1] = ER_IndependentMisfits(Z,Ey, Ey2,'l1');

    [~, ~,~,~, G2_EST(file_idx)] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2,'l2',0, true); % use initialization to get G2_EST
    is_bad_dataset = 1+double(G2_EST(file_idx)/var_y < .2);
    fprintf(is_bad_dataset,'G2_EST = %.2f\n',G2_EST(file_idx)/var_y);
    figure(300); clf; hold on; ylabel('ALL'); 
    [y_INDB, w_INDB,rho_INDB, MSE_hat_INDB] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2,'l2',1);
    
%    [y_indepmisfitnn,w_indepmisfitnn, rho_indepmisfitnn] = ER_IndependentMisfitsNonNegResiduals(Z,Ey, Ey2); Windepmisfitnn(:,file_idx) = w_indepmisfitnn; subplot(224);
%    [y_indepmisfitl1,w_indepmisfitl1] = ER_IndependentMisfits(Z,Ey, Ey2,'l1');
%    [y_indepmisfithuber,w_indepmisfithuber] = ER_IndependentMisfits(Z,Ey, Ey2,'huber');
    %[y_lrm,~,y_oracle_rho] = ER_LowRankMisfitCVX(Z, Ey, Ey2, y_true);
    %[y_rank1,w_rank1,Cstar_rank1_offdiag,rho_rank1] = ER_Rank1Misfit(Z,Ey, Ey2); Wrank1(:,file_idx) = w_rank1; 
%    [y_rank1,w_rank1,Cstar_rank1_offdiag,rho_rank1] = ER_Rank1MisfitConstrainedRho(Z,Ey, Ey2, mse(y_oracle2)); Wrank1(:,file_idx) = w_rank1; 
    %[y_rank1nn,w_rank1nn,Cstar_rank1nn_offdiag,rho_rank1nn] = ER_Rank1MisfitNonNegResiduals(Z,Ey, Ey2); Wrank1nn(:,file_idx) = w_rank1nn; 
    %[y_rank2,w_rank2,Cstar_rank2_offdiag] = ER_Rank2Misfit(Z,Ey, Ey2);
    %[y_UPCRrhoIND, w_UPCRrhoIND] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_IND);
    [y_UPCRrhoINDB, w_UPCRrhoINDB] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_INDB);
    [y_UPCRrhoOracle, w_UPCRrhoOracle] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_true);
    [y_UPCRrhoINDB2c, w_UPCRrhoINDB2c] = ER_UPCRgivenRho2Components(Z,Ey,Ey2,rho_INDB);

    [inlier_idx,outlier_idx, MSE_ss] = subset_selection(y_true,Z,Ey,Ey2,'rho');
    %fprintf('MAIN OUTLIER IDX = '); find(outlier_idx)'
    %fprintf('MAIN INLIER IDX = '); find(inlier_idx)'
    
    if sum(inlier_idx) > 4
        figure(301); hold on; ylabel('SUBSET SELECTION'); 
        [y_INDB_ss, w_INDB_ss,rho_INDB_ss, ~] = ER_IndependentMisfitsBayes(y_true, Z(inlier_idx,:), Ey, Ey2,'l2',1);
        [y_UPCRrhoINDB_ss, w_UPCRrhoINDB_ss] = ER_UPCRgivenRho(Z(inlier_idx,:),Ey,Ey2,rho_INDB_ss);
        [y_UPCRrhoINDB2c_ss, w_UPCRrhoINDB2c_ss] = ER_UPCRgivenRho2Components(Z(inlier_idx,:),Ey,Ey2,rho_INDB_ss);
    else
        % If after subset selection we have 4 or less inliers, take their average (that's not enough
        % to rubstly estimate rho using least squares)
        [y_INDB_ss, w_INDB_ss] = ER_MeanWithBiasCorrection(Z(inlier_idx,:), Ey);
        [y_UPCRrhoINDB_ss, w_UPCRrhoINDB_ss] = ER_MeanWithBiasCorrection(Z(inlier_idx,:), Ey);
        [y_UPCRrhoINDB2c_ss, w_UPCRrhoINDB2c_ss] = ER_MeanWithBiasCorrection(Z(inlier_idx,:), Ey);
    end;

    %% TRYING TO REMOVE OUTLIERS
    fprintf(['OUTLIER IDX = ' num2str(find(outlier_idx)') '\n']);

    MSE_IND = zeros(m,1); MSE_INDB = zeros(m,1); MSE_UPCR = zeros(m,1);MSE_IND_L1 = zeros(m,1);
    for i=1:m
        %MSE_IND(i) = (var_y - 2*rho_IND(i) + C(i,i)) / var_y;
        %MSE_IND_L1(i) = (var_y - 2*rhoIND_L1(i) + C(i,i)) / var_y;
        MSE_INDB(i) = (var_y - 2*rho_INDB(i) + C(i,i)) / var_y;
        %MSE_UPCR(i) = (var_y - 2*rho_UPCR(i) + C(i,i)) / var_y;
    end;
   
    L = eig(C)';
    EIGVALS(:,file_idx) = L;
    figure(400); plot(sort(L,'descend')/sum(L),'kp-'); title('EIGENVALUES OF C'); 
    
    %%    
%     figure(112); clf; set(gca,'fontsize',24); 
%     plot(MSE_IND,MSE_orig(:,file_idx),  'rs', MSE_IND, MSE_IND, 'b-'); xlabel('PREDICTED MSE'); ylabel('ACTUAL MSE'); 
%     hold on; plot(MSE_IND(outlier_idx),MSE_orig(outlier_idx,file_idx),'k>','markersize',10);
%     hold on; plot(MSE_IND,mse(y_MEAN)*ones(m,1),'g--'); 
%     plot(MSE_IND,mse(y_MEAN_ss)*ones(m,1),'m--'); 
%     [val,loc] = sort(MSE_ss);
%     plot(MSE_IND,mse(ER_MeanWithBiasCorrection(Z(loc(1:3),:),Ey))*ones(m,1),'c.-','linewidth',1.5); % Plot mean of best 3
%     plot(MSE_IND,mse(y_oracle2)*ones(m,1),'k.-'); 
%     %axis([0 1 0 1]); 
%     title('IND'); grid on;
%     figure(113); clf; set(gca,'fontsize',24); 
%     plot(MSE_INDB,MSE_orig(:,file_idx),  'rs', MSE_INDB,MSE_INDB , 'b-'); hold on; 
%     plot(MSE_INDB,mse(y_oracle2)*ones(m,1),'k.-');         
%     %axis([0 1 0 1]); 
%     title('INDB'); grid on;
%     figure(114); 
%     plot(MSE_IND_L1,MSE_orig(:,file_idx),  'rs', MSE_IND_L1,MSE_IND_L1, 'b-');
%     title('IND-L1'); grid on;
%     figure(115); 
%     plot(w_UPCR, MSE_orig(:,file_idx), 'rs'); xlabel('VECTOR w'); ylabel('ACTUAL MSE');  % MSE_orig(:,file_idx), MSE_orig(:,file_idx), 'b-');
%     %axis([0 1 0 1]); 
%     title('UPCR'); grid on;

    figure(130); clf; set(gca,'fontsize',24); 
    plot(rho_true/var_y,rho_true/var_y,'k-'); grid on; xlabel('RHO TRUE'); ylabel('RHO EST'); 
    hold on;     
    %plot(rho_true/var_y,rho_IND/var_y,'rs');
    plot(rho_true/var_y,rho_INDB/var_y,'bo','markersize',8);
    if sum(inlier_idx) > 4
        plot(rho_true(inlier_idx)/var_y, rho_INDB_ss/var_y,'cd','markersize',8);
        plot(rho_true/var_y,sign(sum(v_1))*v_1,'md','markersize',8); 
        plot(rho_true(outlier_idx)/var_y, rho_INDB(outlier_idx)/var_y,'k>','markersize',20);
        legend('RHO TRUE','RHO-INDB','RHO-INDB-ss','V1');
    else
        plot(rho_true/var_y,sign(sum(v_1))*v_1,'md','markersize',8); 
        legend('RHO TRUE','RHO-INDB','RHO-INDB-ss','V1');
    end;
    
    %% FOR PAPER
    fig=figure(131); clf; set(gca,'fontsize',20); msize=8;
    plot(rho_true/var_y,rho_INDB/var_y,'cd','markerfacecolor','c','markeredgecolor','b','markersize',msize,'linewidth',1);hold on; 
    %plot(rho_true(outlier_idx)/var_y, rho_INDB(outlier_idx)/var_y,'k>','markersize',msize+5);
    plot(rho_true/var_y,rho_true/var_y,'k-');     
    grid on; xlabel('RHO TRUE / Var(Y)'); ylabel('RHO EST / Var(Y)');
    set(fig,'PaperPositionMode','auto','Position',[500 700 300 200]); set(gca,'Position', [.17 .2 .80 .70]);        

    %%

    mse_true_ss = mse_true(inlier_idx); 
    [mse_val, mse_loc] = min(MSE_INDB(inlier_idx)); 
    EXCESS_MSE(file_idx) = mse_true_ss(mse_loc)-min(mse_true);
    %[v1_val, v1_loc] = max(sign(sum(v_1))*v_1(inlier_idx)); 
    %EXCESS_MSE_v1(file_idx)  = mse_true_ss(v1_loc)-min(mse_true);
    [rho_val, rho_loc] = max(rho_INDB(inlier_idx)); 
    EXCESS_rho(file_idx)  = mse_true_ss(rho_loc)-min(mse_true);

    tmp = Z(inlier_idx,:);
    y_BEST_MSEhat = tmp(mse_loc,:)';
    y_BEST_RHOhat = tmp(rho_loc,:)';    
    
    figure(140); clf; set(gca,'fontsize',24); 
    rank_rho = corr(mse_true,MSE_INDB,'type','Spearman'); 
    rank_v1  = corr(mse_true,-sign(sum(v_1))*v_1,'type','Spearman'); 
    
    %plot(mse_true,MSE_INDB,'rs',mse_true,mse_true,'b-',mse_true,sign(sum(v_1))*v_1,'md'); 
    plot(mse_true,MSE_INDB,'rs',mse_true,mse_true,'b-',mse_true,rho_INDB/var_y,'md'); 
    legend('MSE INDB','MSE TRUE','RHO EST','Location','North');
    hold on;
    plot(mse_true(outlier_idx), MSE_INDB(outlier_idx),'k>','markersize',10);
    %plot(mse_true(outlier_idx), sign(sum(v_1))*v_1(outlier_idx),'k>','markersize',10);
    plot(mse_true(outlier_idx), rho_INDB(outlier_idx)/var_y,'k>','markersize',10);
    plot(mse_true_ss(mse_loc), mse_val,'*k');
    plot(mse_true_ss(rho_loc), rho_val/var_y,'*k'); %plot(mse_true_ss(v1_loc), v1_val,'*k');
    grid on; xlabel('MSE TRUE'); 
    %title(['EXCESS RHO ' num2str(EXCESS_MSE(file_idx)) ' EXCESS v1 ' num2str(EXCESS_MSE_v1(file_idx)) ]); 
    title(['EXCESS MSE ' num2str(EXCESS_MSE(file_idx)) ' EXCESS RHO ' num2str(EXCESS_rho(file_idx)) ]); 
    
    %% FOR PAPER
    fig=figure(141); set(fig,'Name','Outlier Detection (before outlier removal)'); clf; set(gca,'fontsize',24);
    msize=7;
    plot(mse_true(outlier_idx), MSE_INDB(outlier_idx),'ko','markersize',msize+5); hold on; 
    %plot(mse_true(outlier_idx), rho_INDB(outlier_idx)/var_y,'k>','markersize',10);
    plot(mse_true_ss(mse_loc), mse_val,'sr','markersize',msize+5);
    %plot(mse_true_ss(rho_loc), rho_val/var_y,'*k'); %plot(mse_true_ss(v1_loc), v1_val,'*k');
    grid on; xlabel('TRUE MSE'); ylabel('ESTIMATED MSE');
    if any(outlier_idx)
        legend('OUTLIERS','EST BEST','Location','SouthEast');
    else
        legend('EST BEST','Location','SouthEast');
    end;
    plot(mse_true,MSE_INDB,'cd','markerfacecolor','c','markersize',msize,'markeredgecolor','b'); %,mse_true,rho_INDB/var_y,'md'); 
    plot(mse_true,mse_true,'k-');    
    axis([.4 1 .4 1.25]);
    set(fig,'PaperPositionMode','auto'); set(fig,'Position',[300 500 400 300]); set(gca,'Position', [.17 .2 .8 .78]);        

    fig=figure(142); set(fig,'Name','Outlier Detection (after)'); clf; set(gca,'fontsize',24);
    MSE_INDB_SS = ones(sum(inlier_idx),1); loc = find(inlier_idx);
    for i=1:sum(inlier_idx)
        % NOTICE that rho_INDB_ss has only sum(inlier_idx) entries, while C is m x m
        MSE_INDB_SS(i) = (var_y - 2*rho_INDB_ss(i) + C(loc(i),loc(i))) / var_y;
    end
    plot(mse_true_ss(mse_loc), MSE_INDB_SS(MSE_INDB_SS == min(MSE_INDB_SS)),'sr','markersize',msize+5); hold on;
    grid on; xlabel('TRUE MSE'); ylabel('ESTIMATED MSE');
    if any(outlier_idx)
        legend('EST BEST','Location','SouthEast');
    else
        legend('EST BEST','Location','SouthEast');
    end;
    plot(mse_true(inlier_idx),MSE_INDB_SS,'cd','markerfacecolor','c','markersize',msize,'markeredgecolor','b');
    plot(mse_true,mse_true,'k-');
    axis([.4 1 .4 1.25]);
    box on;
    set(fig,'PaperPositionMode','auto'); set(fig,'Position',[500 700 400 300]); set(gca,'Position', [.17 .2 .8 .78]);        
    
    
    
    %% Print results
    dataset_name = files(file_idx).name;
    if filename_contains_index
        seps = strfind(files(file_idx).name,'_');
        last_separator = seps(end);
        dataset_name = dataset_name(1:last_separator-1);
    end;
    
    results = {files(file_idx).name, 'best',min(mean((Z - repmat(y_true,m,1)).^2,2))/var_y, dataset_name}; % best individual regressor
    for alg=who('y_*')'
        if ~strcmp(alg{1}, 'y_true')
            results = [results; {files(file_idx).name, alg{1}, mse(eval(alg{1})), dataset_name}];
        end;
    end;
    results_summary = [results_summary; results];
    
    results,
    %fprintf('PAUSED\n'); pause;    

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
    %Rtrue(:,file_idx) = rho_true; Rindepmisfit(:,file_idx)=rho_indepmisfit; 
    %Rindepmisfitnn(:,file_idx)=rho_indepmisfitnn; Rrank1(:,file_idx)=rho_rank1; %Rrank1nn(:,file_idx)=rho_rank1nn;
    %RHO_EST(2,file_idx) = norm(rho_indepmisfit-rho_true);    
    %RHO_EST(3,file_idx) = norm(rho_indepmisfitnn-rho_true);
    %RHO_EST(4,file_idx) = norm(rho_rank1-rho_true);
    %RHO_EST(5,file_idx) = norm(rho_rank1nn-rho_true);
    %fprintf('norm(residuals): indepmisfit: %.2f, indepmisfit-nn: %.2f, rank-1: %.2f, rank-1nn: %.2f', RHO_EST(:,file_idx))
    %[v,d]=eig(C); COS_DIST_V1_RHO(1,file_idx) = 1-dot(v(:,end),rho_true./norm(rho_true));
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

%p=pivottable(results_summary,2,1,3,@sum)
p=pivottable(results_summary,2,4,3,@mean)
a=cell2mat(p(2:end,2:end))
p(:,1)

%% SAVE ENVIRONMENT
save('main.mat');
%load('icml.mat');

%%
is_RF = 0;
idx_orc = 8+is_RF; idx_mean = idx_orc-2; idx_med = idx_orc-1;

figure(1); clf;  msize = 8; 
set(gca,'fontsize',20); 
hold on; grid on; 
plot(a(idx_orc,:),a(idx_mean,:),'ko'); %mean
plot(a(idx_orc,:),a(idx_med,:),'k>','markersize',msize);   %median
plot(a(idx_orc,:),a(2,:),'rd','markersize',msize);   %D-GEM
%plot(a(idx_orc,:),a(3,:),'md','markersize',msize);   %GEM direction 1
%plot(a(idx_orc,:),a(4,:),'gp','markersize',msize);   %GEM with rho estimation
%plot(a(idx_orc,:),a(end-1,:),'md','markersize',msize);   %PCR*
plot(a(idx_orc,:),a(end-2,:),'ms','markersize',msize);   %PCR given delta star*
plot(a(idx_orc,:),a(end-1,:),'rx','markersize',msize);   %PCR delta=MRE*
plot(a(idx_orc,:),a(end,:),'ro','markersize',msize);   %PCR delta=WMRE*
plot(a(idx_orc,:),a(end-4,:),'mp','markersize',msize);   %PCR new
plot(a(idx_orc,:),a(5,:),'g^','markersize',msize);  %INDEPENDENT ERRORS
%plot(a(idx_orc,:),a(end-4,:),'p','markersize',msize);  %RANK-1
%plot(a(idx_orc,:),a(5:7,:),'p','markersize',msize);

%legend('MEAN','MED','DGEM','PCR','PCR given \delta^*','Location','NorthWest'); 
%legend('MEAN','MED','DGEM','GEM \propto 1','GEM with estimated \rho','Location','NorthWest'); 
legend('MEAN','MED','DGEM','PCR given \delta^*','PCR \delta=MRE','PCR \delta=WMRE','PCRnew','IND','Location','NorthWest'); 
plot(a(idx_orc,:),a(idx_orc,:),'b-'); 
plot(a(idx_orc,:),a(idx_orc-3,:),'bo','markersize',msize+2); 

%axis([0 0.5 0 0.8]); 

% remove ratings of sweets (since its ordinal, and not continuous)
%a(:,end-1) = []; %a(2,:) = [];
%% for paper
is_RF = 0;
%idx_orc = 13+is_RF; idx_mean = 5; idx_med = idx_mean+1;
idx_orc      = find(strcmp(p(2:end,1),'y_oracle2'));
idx_mean     = find(strcmp(p(2:end,1),'y_MEAN'));
idx_med      = find(strcmp(p(2:end,1),'y_MED'));
idx_upcr     = find(strcmp(p(2:end,1),'y_UPCRrhoINDB'));
idx_upcr_ss  = find(strcmp(p(2:end,1),'y_UPCRrhoINDB_ss'));
idx_upcr2c  = find(strcmp(p(2:end,1),'y_UPCRrhoINDB2c'));
idx_upcr2c_ss  = find(strcmp(p(2:end,1),'y_UPCRrhoINDB2c_ss'));

fig = 1;
figure(fig); clf;  msize = 8; 
set(gca,'fontsize',18); 
hold on; grid on; 
plot(a(idx_orc,:),a(idx_mean,:)-a(idx_orc,:),'bo','markerfacecolor','k'); %mean
plot(a(idx_orc,:),a(idx_med,:)-a(idx_orc,:),'k>','markersize',msize,'markerfacecolor','k');   %median
%plot(a(idx_orc,:),a(2+is_RF,:)-a(idx_orc,:),'d','markersize',msize,'linewidth',2,'markeredgecolor',[.9 0 0]);   %D-GEM
%plot(a(idx_orc,:),a(3+is_RF,:)-a(idx_orc,:),'v','markersize',msize,'linewidth',2,'markeredgecolor',[0 .7 0]);  %INDEPENDENT ERRORS
%plot(a(idx_orc,:),a(3+is_RF,:)-a(idx_orc,:),'v','markersize',msize,'linewidth',2,'markeredgecolor',[0 .7 0]);  %INDB
%plot(a(idx_orc,:),a(8,:)-a(idx_orc,:),'ms','markersize',msize+1,'markerfacecolor','m');   %PCR given delta star*
%plot(a(idx_orc,:),a(4,:)-a(idx_orc,:),'gp','markersize',msize+1,'markerfacecolor',[0 .7 0]);   %INDB
%plot(a(idx_orc,:),a(10,:)-a(idx_orc,:),'mo','markersize',msize+1);   %PCR delta=0
%plot(a(idx_orc,:),a(13,:)-a(idx_orc,:),'mp','markersize',msize+1);   %PCR rhoIND
%plot(a(idx_orc,:),a(14,:)-a(idx_orc,:),'b>','markersize',msize+1);   %PCR rhoINDB
%plot(a(idx_orc,:),a(17,:)-a(idx_orc,:),'cv','markersize',msize+1);   %PCR delta=sum(t)=1
%plot(a(idx_orc,:),a(idx_mean+1,:)-a(idx_orc,:),'kp','markersize',msize+1);   %Mean over subset selection
%plot(a(idx_orc,:),a(end-3,:)-a(idx_orc,:),'b*','markersize',msize+1);   %PCR delta=0
%plot(a(idx_orc,:),a(end-4,:)-a(idx_orc,:),'p','markersize',msize);  %RANK-1

legend('MEAN','MED','DGEM','IND','U-PCR','PCR \delta=0','PCR \rho IND','PCR \rho INDB','U-PCR sum(w)=1','MEAN(ss)','Location','NorthEast'); 
%legend('MEAN','MED','DGEM','IND','U-PCR','INDB','U-PCR minMSE','MEAN(ss)','Location','NorthEast'); 
%plot(a(idx_orc,:),a(idx_orc,:),'b-'); 
xlabel('\delta_{OR}=MSE(oracle)/Var(Y)'); ylabel('MSE/Var(Y) - \delta_{or}');
%set(gca,'yscale','log'); grid minor; set(gca,'ytick',[.01 .1])

axis([0 1 0 .5]); 
set(fig,'PaperPositionMode','auto');
set(fig,'Position',[574 656 1045 378]);
% axis([0.2 .95 0.2 1]); set(fig,'Position',[574 656 1045 578]); % for drug response data
set(gca,'Position', [.1 .22 .89 .70]);
%saveas(fig,'plots/rf50_results.fig','fig'); saveas(fig,'plots/rf50_results.eps','psc2');

%% Print Results Table
is_RF = 0;
idx_orc = 8+is_RF; idx_mean = idx_orc-2; idx_med = idx_orc-1;
p = pivottable(results_summary,4,2,3,@mean);
e = pivottable(results_summary,4,2,3,@std);
fprintf('Dataset \t\t& n\t& n_{train}\t& d\t& avgRE (+/-sd)\t& minRE (+/-sd)\t& deltastar (+/-sd)\n');
for i=2:size(p,1)
    dataset_name = p{i,1};
    dataset_name=strrep(dataset_name,'_',' '); dataset_name(1) = upper(dataset_name(1));
    
    fprintf('%20s &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) \\\\ \\hline\n',...
    dataset_name,p{i,idx_orc+1},e{i,idx_orc+1},p{i,idx_mean+1},e{i,idx_mean+1}, ...
                 p{i,idx_med+1},e{i,idx_med+1},p{i,1+2+is_RF},e{i,1+2+is_RF}, ...
                 p{i,1+5+is_RF},e{i,1+5+is_RF},p{i,end-2},e{i,end-2});
end;