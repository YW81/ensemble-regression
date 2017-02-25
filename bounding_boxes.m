clear all; close all;
addpath Ensemble_Regressors/
addpath HelperFunctions/
load box_regression_coco4

%% RUN ENSEMBLE METHODS FOR EVERY CLASS INDIVIDUALLY
classes = unique(class_id); % classes = 4; %
response_vars = {'x1','y1','x2','y2'}; % response_vars = {'x1'}; %

fprintf(['\tClass\t # Instances (n)\n']); disp([unique(class_id),histc(class_id,unique(class_id))]);

results_summary = {};
tests_size = [length(response_vars),length(classes)];
for var_idx=1:length(response_vars)
    for cls_idx=1:length(classes)
        fprintf(['RESPONSE ' response_vars{var_idx} ', CLASS ' num2str(classes(cls_idx)) '\n==============================\n']);
        
        clear Z y_*
        Z      = eval(['Z_' response_vars{var_idx}]); Z = Z(:,class_id == classes(cls_idx));
        y_true = eval(['yy_true_' response_vars{var_idx}]); y_true = y_true(class_id == classes(cls_idx));
        file_idx = sub2ind(tests_size,var_idx,cls_idx); % the index of the current experiment
        
        %% START HERE
        Z = Z([4,6,8,9,11,12],:); % high correlation between the 8 best regressors, this filters all of them [val loc]=sort(mean(MSE_orig,2)); imagesc(corr(Z(loc,loc)'));
        %Z([9,11],:) = [];
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

if ~exist('MSE_orig','var')
    MSE_orig = zeros(m,prod(tests_size)); %to access use: sub2ind(tests_size,var_idx,cls_idx)
    G2_EST = zeros(prod(tests_size),1);
    VAR_Y = zeros(prod(tests_size),1);
end

VAR_Y(file_idx) = var_y;
for i=1:m
    MSE_orig(i, file_idx) = mse(Z(i,:)');
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
[~, ~,~,~, G2_EST(file_idx)] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2,'l2',0, true); % use initialization to get G2_EST
is_bad_dataset = 1+double(G2_EST(file_idx)/var_y < .2);
fprintf(is_bad_dataset,'G2_EST = %.2f\n',G2_EST(file_idx)/var_y);
figure(300); clf; hold on; ylabel('ALL'); 
[y_INDB, w_INDB,rho_INDB, MSE_hat_INDB] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2,'l2',1);

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
plot(y_true',Z','.');
grid minor; grid on;
hold on; plot(y_true',y_true','b-');
title('Unbiased Competitor Predictions vs. True Response');
%axis([-5 5 -5 5]);
xlabel('Y');
ylabel('$$\hat{y}(X)$$','interpreter','latex')

% Ensemble Method Predictions
figure(2);
plot(y_true',y_true','b-'); hold on;
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
%axis([-5 5 -5 5]);
title('Unsupervised Ensemble Methods vs. True Response');
xlabel('Y');
ylabel('$$\hat{y}(X)$$','interpreter','latex')

%% Rank Correlation
MSE_IND = zeros(m,1); MSE_INDB = zeros(m,1); MSE_UPCR = zeros(m,1);MSE_IND_L1 = zeros(m,1);
for i=1:m
    %MSE_IND(i) = (var_y - 2*rho_IND(i) + C(i,i)) / var_y;
    %MSE_IND_L1(i) = (var_y - 2*rhoIND_L1(i) + C(i,i)) / var_y;
    MSE_INDB(i) = (var_y - 2*rho_INDB(i) + C(i,i)) / var_y;
    %MSE_UPCR(i) = (var_y - 2*rho_UPCR(i) + C(i,i)) / var_y;
end;

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
    plot(mse_true,MSE_INDB,'cd','markerfacecolor','c','markersize',msize,'markeredgecolor','b'); %,mse_true,rho_INDB/var_y,'md'); 
    hold on;
    plot(mse_true,mse_true,'k-');    
    h1=plot(mse_true(outlier_idx), MSE_INDB(outlier_idx),'ko','markersize',msize+5,'linewidth',1); 
    %plot(mse_true(outlier_idx), rho_INDB(outlier_idx)/var_y,'k>','markersize',10);
    h2=plot(mse_true_ss(mse_loc), mse_val,'sr','markersize',msize+5,'linewidth',2);
    %plot(mse_true_ss(rho_loc), rho_val/var_y,'*k'); %plot(mse_true_ss(v1_loc), v1_val,'*k');
    grid on; xlabel('TRUE MSE'); ylabel('ESTIMATED MSE');
    if any(outlier_idx)
        legend([h1 h2],'OUTLIERS','EST BEST','Location','SouthEast');
    else
        legend(h2,'EST BEST','Location','SouthEast');
    end;
    %axis([.4 1 .4 1.25]);
    set(fig,'PaperPositionMode','auto'); set(fig,'Position',[300 500 400 300]); set(gca,'Position', [.17 .2 .8 .78]);        
    if classes(cls_idx) == 4 && strcmp(response_vars{var_idx},'x1')
        print -depsc 'plots/icml/bounding_box_MSE_estimation.eps';
        fprintf(2,'SAVED 141 TO plots/icml/bounding_box_MSE_estimation.eps');
    end;
    
    fig=figure(142); set(fig,'Name','Outlier Detection (after)'); clf; set(gca,'fontsize',24);
    MSE_INDB_SS = ones(sum(inlier_idx),1); loc = find(inlier_idx);
    for i=1:sum(inlier_idx)
        % NOTICE that rho_INDB_ss has only sum(inlier_idx) entries, while C is m x m
        MSE_INDB_SS(i) = (var_y - 2*rho_INDB_ss(i) + C(loc(i),loc(i))) / var_y;
    end
    plot(mse_true(inlier_idx),MSE_INDB_SS,'cd','markerfacecolor','c','markersize',msize,'markeredgecolor','b');
    hold on;
    plot(mse_true,mse_true,'k-');
    h=plot(mse_true_ss(mse_loc), MSE_INDB_SS(MSE_INDB_SS == min(MSE_INDB_SS)),'sr','markersize',msize+5,'linewidth',2);
    grid on; xlabel('TRUE MSE'); ylabel('ESTIMATED MSE');
    if any(outlier_idx)
        legend(h,'EST BEST','Location','SouthEast');
    else
        legend(h,'EST BEST','Location','SouthEast');
    end;
    %axis([.4 1 .4 1.25]);
    
    set(fig,'PaperPositionMode','auto'); set(fig,'Position',[500 700 400 300]); set(gca,'Position', [.17 .2 .8 .78]);        
%%

results = {['VAR ' response_vars{var_idx} ', CLS ' num2str(classes(cls_idx))], ...
           'best',min(mean((Z - repmat(y_true,m,1)).^2,2))/var_y, ...
           ['VAR ' response_vars{var_idx} ', CLS ' num2str(classes(cls_idx))]}; % best individual regressor
for alg=who('y_*')'
    if ~strcmp(alg{1}, 'y_true')
        results = [results; {['VAR ' response_vars{var_idx} ', CLS ' num2str(classes(cls_idx))], ...
                             alg{1}, mse(eval(alg{1})), ...
                             ['VAR ' response_vars{var_idx} ', CLS ' num2str(classes(cls_idx))]}];
    end;
end;
results_summary = [results_summary; results];

%[w_oracle2, w_best],

fprintf('PAUSE\n'); pause;

    end; % FOR EACH CLASS
end; % FOR EACH RESPONSE VARIABLE (X1,Y1,X2,Y2)


p=pivottable(results_summary,2,1,3,@mean)
a=cell2mat(p(2:end,2:end))
p(:,1)

%save bounding_boxes_results.mat