need to add the case for m<=4 after subset selection
need to do subset selection accoring to MSE
%% Read data from files
clear all; close all;
addpath Ensemble_Regressors/
addpath HelperFunctions/

%load ./Datasets/RealWorld/Erhan/tox_challenge/tox_gs.mat; load ./Datasets/RealWorld/Erhan/tox_challenge/tox_pred.mat
% y_true = gs(:)'; 
% for i=1:size(pred,1); 
%     Z(i,:) = pred(i,:); 
% end; 
% Z(sum(isnan(Z')) > 0,:) = [];

load ./Datasets/RealWorld/Erhan/tox.mat

% for file_idx=1:size(gs,2)
%     y_true = gs(:,file_idx)';
%     Z = pred(:,:,file_idx);
%     Z(sum(isnan(Z')) > 0,:) = [];

%% Estimators
y_true = double(y_true) - mean(y_true);
Z = bsxfun(@minus, Z, mean(Z,2));
[m,n] = size(Z);
Ey = mean(y_true);
Ey2 = mean(y_true.^2);
var_y = Ey2 - Ey.^2;
C = cov(Z');
[v_1 lambda_1] = eigs(C,1,'lm'); 
rho_true = mean(Z .* repmat(y_true,m,1),2);
mse = @(x) mean((y_true' - x).^2 / var_y);    
mse_true = zeros(m,1); 
for i=1:m 
    mse_true(i) = mse(Z(i,:)');
end


MSE_orig = zeros(m,1);
for i=1:m
    MSE_orig(i) = mse(Z(i,:)');
end;

[y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z);
[y_best,w_best] = ER_BestRegressor(y_true,Z);
[y_MEAN,beta_MEAN] = ER_MeanWithBiasCorrection(Z, Ey);
%y_biasedmean = mean(Z)'; % most teams are unbiased, so this should be equivalent to the mean
y_MED = ER_MedianWithBiasCorrection(Z, Ey);
[y_DGEM,w_DGEM] = ER_UnsupervisedDiagonalGEM(Z, Ey);
%y_gem  = ER_UnsupervisedGEM(Z, Ey,Ey2);
y_UPCR_delta0 = ER_SpectralApproach(Z, Ey, Ey2);
[y_UPCR,w_UPCR] = ER_SpectralApproachGivenDeltaStar(Z, Ey, Ey2,mse(y_oracle2));
[y_UPCR_MRE,w_UPCR_MRE] = ER_SpectralApproachDeltaMinMRE(Z, Ey, Ey2,mse(y_oracle2));
[y_UPCR_WMRE,w_UPCR_WMRE] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2,mse(y_oracle2));
[y_IND,w_IND] = ER_IndependentMisfits(Z,Ey, Ey2); 
[y_UPCRt1,w_UPCRt1] = ER_SpectralApproachWeightsSum1(Z, Ey, Ey2);

%% Bayes Optimal Methods
figure(300); clf; hold on; ylabel('ALL'); 
[y_IND,w_IND,rho_IND] = ER_IndependentMisfits(Z,Ey, Ey2); 
[y_INDB, w_INDB,rho_INDB, MSE_hat_INDB] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2,'l2',1);
[inlier_idx,outlier_idx, MSE_ss] = subset_selection(y_true,Z,Ey,Ey2,'rho');
[y_MEAN_ss, w_MEAN_ss] = ER_MeanWithBiasCorrection(Z(inlier_idx,:), Ey);
[y_UPCRrhoINDB, w_UPCRrhoINDB] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_INDB);
[y_UPCRrhoOracle, w_UPCRrhoOracle] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_true);
[y_UPCRrhoINDB2c, w_UPCRrhoINDB2c] = ER_UPCRgivenRho2Components(Z,Ey,Ey2,rho_INDB);
figure(301); hold on; ylabel('SUBSET SELECTION'); 
[y_INDB_ss, w_INDB_ss,rho_INDB_ss, ~] = ER_IndependentMisfitsBayes(y_true, Z(inlier_idx,:), Ey, Ey2,'l2',1);
[y_UPCRrhoINDB_ss, w_UPCRrhoINDB_ss] = ER_UPCRgivenRho(Z(inlier_idx,:),Ey,Ey2,rho_INDB_ss);    
[y_UPCRrhoINDB2c_ss, w_UPCRrhoINDB2c_ss] = ER_UPCRgivenRho2Components(Z(inlier_idx,:),Ey,Ey2,rho_INDB_ss);

figure(130); clf; set(gca,'fontsize',24); 
plot(rho_true/var_y,rho_IND/var_y,'rs',rho_true/var_y,rho_true/var_y,'k-'); grid on; xlabel('RHO TRUE'); ylabel('RHO EST'); 
hold on; 
plot(rho_true/var_y,rho_INDB/var_y,'bo');
plot(rho_true(outlier_idx)/var_y, rho_INDB(outlier_idx)/var_y,'k>','markersize',20);
plot(rho_true(inlier_idx)/var_y, rho_INDB_ss/var_y,'cd');

figure(400); plot(sort(eig(cov(Z')),'descend') / trace(cov(Z')), 'ko-');

% y_E_ls = ER_Erhan(Z','ls');
% y_E_lad = ER_Erhan(Z','lad');
% y_E_sd = ER_Erhan(Z','sd');
% y_E_wls = ER_Erhan(Z','wls');
% y_E_wlad = ER_Erhan(Z','wlad');
% y_E_als = ER_Erhan(Z','als');

%% MSE Results
fprintf('\n\n');
mse = @(x) mean((y_true' - x).^2) / var_y;
for alg=who('y_*')'
    if ((~strcmp(alg{1}, 'y_true')) && (~strcmp(alg{1}, 'y_true_orig')))
        alg_name = alg{1}; alg_name = alg_name(3:end);%upper(alg_name(3:end));
        %fprintf('%s\t%.3f\n',alg_name,sqrt(mse(eval(alg{1}))));        
        fprintf('%s\t%.3f\n',alg_name,(mse(eval(alg{1}))));        
    end;
end;
%fprintf('Best individual RMSE: %g\n', sqrt(min(mean((Z - repmat(y_true,m,1)).^2,2))/var_y));
fprintf('Best individual MSE: %g\n', (min(mean((Z - repmat(y_true,m,1)).^2,2))/var_y));

% fprintf('\n\n\nAlgorithm\t& $\\text{MAD}$\t& Concordance\n');
% for alg=who('y_*')'
%     if ((~strcmp(alg{1}, 'y_true')) && (~strcmp(alg{1}, 'y_true_orig')))
%         algstr = alg{1}; algstr = algstr(3:end); algstr(1) = upper(algstr(1));
%         fprintf('\\multicolumn{1}{|l|}{%s} \t& %.2f & %.3f \\\\ \\hline\n',algstr,mad(eval(alg{1})),concordance_index(y_true,eval(alg{1})));
%     end;
% end;

%% Some plots

% Original predictions after de-biasing
figure(1);
plot(y_true',Z','.')
grid minor; grid on;
hold; plot(y_true',y_true','b-');
title('Unbiased Competitor Predictions vs. True Response');
axis([-5 5 -5 5]);
xlabel('Y');
ylabel('$$\hat{y}(X)$$','interpreter','latex')

% Ensemble Method Predictions
figure(2);
plot(y_true',y_true','b-'); hold;
grid minor; grid on;
leg = {'true response'};
for alg=who('y_*')'
    if ((strcmp(alg{1}, 'y_DGEM')) || (strcmp(alg{1}, 'y_MEAN')) || (strcmp(alg{1}, 'y_MED')) ...
         || (strcmp(alg{1}, 'y_UPCR')) || (strcmp(alg{1}, 'y_IND')))
        plot(y_true',eval(alg{1}),'.');
        leg = [leg, alg{1}];
    end;
end;
legend(leg,'interpreter','none');
axis([-5 5 -5 5]);
title('Unsupervised Ensemble Methods vs. True Response');
xlabel('Y');
ylabel('$$\hat{y}(X)$$','interpreter','latex')

%% Rank Correlation
MSE_IND = zeros(m,1); MSE_INDB = zeros(m,1); MSE_UPCR = zeros(m,1);MSE_IND_L1 = zeros(m,1);
for i=1:m
    MSE_IND(i) = (var_y - 2*rho_IND(i) + C(i,i)) / var_y;
    %MSE_IND_L1(i) = (var_y - 2*rhoIND_L1(i) + C(i,i)) / var_y;
    MSE_INDB(i) = (var_y - 2*rho_INDB(i) + C(i,i)) / var_y;
    %MSE_UPCR(i) = (var_y - 2*rho_UPCR(i) + C(i,i)) / var_y;
end;
[val loc] = min(MSE_INDB); 
EXCESS_MSE_rho = mse_true(loc)-min(mse_true); 
mse_best = loc;
[val loc] = max(sum(sign(v_1))*v_1); 
EXCESS_MSE_v1  = mse_true(loc)-min(mse_true); 
v1_best = loc;

figure(140); clf; set(gca,'fontsize',24); 
rank_rho = corr(mse_true,MSE_INDB,'type','Spearman'); 
rank_v1  = corr(mse_true,-sign(sum(v_1))*v_1,'type','Spearman'); 

plot(mse_true,MSE_INDB,'rs',mse_true,mse_true,'b-',mse_true,sign(sum(v_1))*v_1,'md'); 
hold on;
plot(mse_true(outlier_idx), MSE_INDB(outlier_idx),'k>','markersize',10);
plot(mse_true(outlier_idx), sign(sum(v_1))*v_1(outlier_idx),'k>','markersize',10);
plot(mse_true(mse_best), MSE_INDB(mse_best),'*k');
plot(mse_true(v1_best), sign(sum(v_1))*v_1(v1_best),'*k');
grid on; xlabel('MSE TRUE'); 
%title(['rho ' num2str(rank_rho) ' v1 ' num2str(rank_v1) ]); 
title(['EXCESS RHO ' num2str(EXCESS_MSE_rho) ' EXCESS v1 ' num2str(EXCESS_MSE_v1) ]); 
legend('MSE INDB','MSE TRUE','V_1');

%     fprintf('PAUSE\n'); pause;
% end;