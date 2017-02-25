clear all; close all;
addpath Ensemble_Regressors/
addpath HelperFunctions/
load box_regression_coco4_alldata
IMG_ROOT = './Datasets/RealWorld/Moshik/coco_reg4_images/';

%% RUN ENSEMBLE METHODS FOR EVERY CLASS INDIVIDUALLY
classes = unique(class_id); % classes = 4; %
response_vars = {'x1','y1','x2','y2'}; % response_vars = {'x1'}; %

fprintf(['\tClass\t # Instances (n)\n']); disp([unique(class_id),histc(class_id,unique(class_id))]);

results_summary = {};
tests_size = [length(response_vars),length(classes)];
for cls_idx=1:length(classes)
    clear PRED_UPCR PRED_ORIG TRUE_RESP MEAN_TRUE MEAN_PRED MEAN_X MEAN_Y X_* Y_* H_* W_*
        
    for var_idx=1:length(response_vars)
%         fprintf(['RESPONSE ' response_vars{var_idx} ', CLASS ' num2str(classes(cls_idx)) '\n==============================\n']);
        
        clear Z y_*
        Z      = eval(['Z_' response_vars{var_idx}]); Z = Z(:,class_id == classes(cls_idx));
        y_true = eval(['yy_true_' response_vars{var_idx}]); y_true = y_true(class_id == classes(cls_idx));
        file_idx = sub2ind(tests_size,var_idx,cls_idx); % the index of the current experiment
        
        %% START HERE
        Z = Z([4,6,8,9,11,12],:); % high correlation between the 8 best regressors, this filters all of them [val loc]=sort(mean(MSE_orig,2)); imagesc(corr(Z(loc,loc)'));
        %Z([9,11],:) = [];
    orig_mean = mean(y_true); orig_mean_pred = mean(Z,2);
    
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
        
        PRED_UPCR = zeros(n,length(response_vars));
        PRED_ORIG = zeros(n,length(response_vars),m);
        TRUE_RESP = zeros(n,length(response_vars));
    
        MEAN_TRUE = zeros(1,length(response_vars));
        MEAN_PRED = zeros(m,length(response_vars));
    end
    
    VAR_Y(file_idx) = var_y;
    for i=1:m
        MSE_orig(i, file_idx) = mse(Z(i,:)');
    end;
    
    MEAN_TRUE(1,var_idx) = orig_mean;
    MEAN_PRED(:,var_idx) = orig_mean_pred;

    [y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z);
    [y_best,w_best] = ER_BestRegressor(y_true,Z);
    [y_MEAN,beta_MEAN] = ER_MeanWithBiasCorrection(Z, Ey);
    %y_biasedmean = mean(Z)'; % most teams are unbiased, so this should be equivalent to the mean
    y_MED = ER_MedianWithBiasCorrection(Z, Ey);
%     [y_DGEM,w_DGEM] = ER_UnsupervisedDiagonalGEM(Z, Ey);
%     %y_gem  = ER_UnsupervisedGEM(Z, Ey,Ey2);
%     y_UPCR_delta0 = ER_SpectralApproach(Z, Ey, Ey2);
%     [y_UPCR,w_UPCR] = ER_SpectralApproachGivenDeltaStar(Z, Ey, Ey2,mse(y_oracle2));
%     [y_UPCR_MRE,w_UPCR_MRE] = ER_SpectralApproachDeltaMinMRE(Z, Ey, Ey2,mse(y_oracle2));
%     [y_UPCR_WMRE,w_UPCR_WMRE] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2,mse(y_oracle2));
%     [y_IND,w_IND] = ER_IndependentMisfits(Z,Ey, Ey2); 
%     [y_UPCRt1,w_UPCRt1] = ER_SpectralApproachWeightsSum1(Z, Ey, Ey2);

    %% Bayes Optimal Methods
    [~, ~,~,~, G2_EST(file_idx)] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2,'l2',0, true); % use initialization to get G2_EST
    is_bad_dataset = 1+double(G2_EST(file_idx)/var_y < .2);
%     fprintf(is_bad_dataset,'G2_EST = %.2f\n',G2_EST(file_idx)/var_y);

    [y_INDB, w_INDB,rho_INDB, MSE_hat_INDB] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2,'l2',0);
    [y_UPCRrhoINDB, w_UPCRrhoINDB] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_INDB);
    [y_UPCRrhoOracle, w_UPCRrhoOracle] = ER_UPCRgivenRho(Z,Ey,Ey2,rho_true);
    [y_UPCRrhoINDB2c, w_UPCRrhoINDB2c] = ER_UPCRgivenRho2Components(Z,Ey,Ey2,rho_INDB);

    [inlier_idx,outlier_idx, MSE_ss] = subset_selection(y_true,Z,Ey,Ey2,'rho');

    if sum(inlier_idx) > 4
        [y_INDB_ss, w_INDB_ss,rho_INDB_ss, ~] = ER_IndependentMisfitsBayes(y_true, Z(inlier_idx,:), Ey, Ey2,'l2',0);
        [y_UPCRrhoINDB_ss, w_UPCRrhoINDB_ss] = ER_UPCRgivenRho(Z(inlier_idx,:),Ey,Ey2,rho_INDB_ss);
        [y_UPCRrhoINDB2c_ss, w_UPCRrhoINDB2c_ss] = ER_UPCRgivenRho2Components(Z(inlier_idx,:),Ey,Ey2,rho_INDB_ss);
    else
        % If after subset selection we have 4 or less inliers, take their average (that's not enough
        % to rubstly estimate rho using least squares)
        [y_INDB_ss, w_INDB_ss] = ER_MeanWithBiasCorrection(Z(inlier_idx,:), Ey);
        [y_UPCRrhoINDB_ss, w_UPCRrhoINDB_ss] = ER_MeanWithBiasCorrection(Z(inlier_idx,:), Ey);
        [y_UPCRrhoINDB2c_ss, w_UPCRrhoINDB2c_ss] = ER_MeanWithBiasCorrection(Z(inlier_idx,:), Ey);
    end;


%     %% MSE Results
%     fprintf('\n\n');
%     mse = @(x) mean((y_true' - x).^2) / var_y;
%     for alg=who('y_*')'
%         if ((~strcmp(alg{1}, 'y_true')) && (~strcmp(alg{1}, 'y_true_orig')))
%             alg_name = alg{1}; alg_name = alg_name(3:end);%upper(alg_name(3:end));
%             fprintf('%s\t%.3f\n',alg_name,(mse(eval(alg{1}))));        
%         end;
%     end;
%     fprintf('Best individual MSE: %g\n', (min(mean((Z - repmat(y_true,m,1)).^2,2))/var_y));


    %% Rank Correlation
    MSE_INDB = zeros(m,1);
    for i=1:m
        MSE_INDB(i) = (var_y - 2*rho_INDB(i) + C(i,i)) / var_y;
    end;

    mse_true_ss = mse_true(inlier_idx); 
    [mse_val, mse_loc] = min(MSE_INDB(inlier_idx)); 
    EXCESS_MSE(file_idx) = mse_true_ss(mse_loc)-min(mse_true);
    [rho_val, rho_loc] = max(rho_INDB(inlier_idx)); 
    EXCESS_rho(file_idx)  = mse_true_ss(rho_loc)-min(mse_true);

    tmp = Z(inlier_idx,:);
    y_BEST_MSEhat = tmp(mse_loc,:)';
    y_BEST_RHOhat = tmp(rho_loc,:)';
    
    PRED_UPCR(:,var_idx) = y_UPCRrhoINDB2c_ss + orig_mean;
    for i=1:m
        PRED_ORIG(:,var_idx,i) = Z(i,:)' + orig_mean_pred(i);
    end;
    TRUE_RESP(:,var_idx) = y_true' + orig_mean;

    end; % FOR EACH RESPONSE VARIABLE (X1,Y1,X2,Y2)

    %% CALCULATE SmoothL1Loss ON CENTER, WIDTH AND HEIGHT
    X_TRUE = (TRUE_RESP(:,1)+TRUE_RESP(:,3))/2;
    Y_TRUE = (TRUE_RESP(:,2)+TRUE_RESP(:,4))/2;
    W_TRUE = TRUE_RESP(:,3)-TRUE_RESP(:,1);
    H_TRUE = TRUE_RESP(:,4)-TRUE_RESP(:,2);

    LOC_TRUE = [TRUE_RESP(:,1), TRUE_RESP(:,2), W_TRUE, H_TRUE];
    %LOC_TRUE = [X_TRUE, Y_TRUE, W_TRUE, H_TRUE];

    X_UPCR = (PRED_UPCR(:,1)+PRED_UPCR(:,3))/2;
    Y_UPCR = (PRED_UPCR(:,2)+PRED_UPCR(:,4))/2;
    W_UPCR = PRED_UPCR(:,3)-PRED_UPCR(:,1);
    H_UPCR = PRED_UPCR(:,4)-PRED_UPCR(:,2);
    LOC_UPCR = [PRED_UPCR(:,1), PRED_UPCR(:,2), W_UPCR, H_UPCR];
    %LOC_UPCR = [X_UPCR, Y_UPCR, W_UPCR, H_UPCR];
    
    X_ORIG = zeros(n,m); Y_ORIG = zeros(n,m); W_ORIG = zeros(n,m); H_ORIG = zeros(n,m);
    LOC_ORIG = zeros(n,4,m);
    for i=1:m
        X_ORIG(:,i) = (PRED_ORIG(:,1,i)+PRED_ORIG(:,3,i))/2;
        Y_ORIG(:,i) = (PRED_ORIG(:,2,i)+PRED_ORIG(:,4,i))/2;
        W_ORIG(:,i) = PRED_ORIG(:,3,i)-PRED_ORIG(:,1,i);
        H_ORIG(:,i) = PRED_ORIG(:,4,i)-PRED_ORIG(:,2,i);
        LOC_ORIG(:,:,i) = [PRED_ORIG(:,1,i), PRED_ORIG(:,2,i), W_ORIG(:,i), H_ORIG(:,i)];
    end;
    
    %%

    %SmoothL1Loss(x) =
    %  //   0.5 * (x) ** 2    -- if x < 1.0
    %  //   |x| - 0.5-- otherwise
    SmoothL1Loss = @(x) ((.5 * (x).^2) .* (x<1)) + ((abs(x) - .5).*(x>=1));
    TotalLoss = @(x,y,w,h) (SmoothL1Loss(x-X_TRUE).*SmoothL1Loss(y-Y_TRUE).*SmoothL1Loss(w-W_TRUE).*SmoothL1Loss(h-H_TRUE));
    
    loss_upcr = TotalLoss(X_UPCR,Y_UPCR,W_UPCR,H_UPCR);
    loss_orig = zeros(n,m);
    for i=1:m
        loss_orig(:,i) = TotalLoss(X_ORIG(:,i),Y_ORIG(:,i),W_ORIG(:,i),H_ORIG(:,i));
    end;
    loss_mean = mean(loss_orig,2);
    loss_prod = nthroot(prod(loss_orig,2),m);
    
    %% look for high loss_mean/prod and low loss_upcr
    [val1 loc1] = sort(loss_mean - loss_upcr,'descend');
    [val2 loc2] = sort(loss_prod - loss_upcr,'descend');
    
%     fprintf(['CLASS ' num2str(classes(cls_idx)) ':\n']);
    d=squeeze(ALLDATA(1,:,7:10));
    
    CLASS_DICT = {'person','chair','table','dog','cat','sofa'}; %(categories_id_dict={'person': 0, 'chair': 1, 'table': 2,'dog': 3,'cat': 4,'sofa': 5}
    fprintf(['CLASS: ' CLASS_DICT{classes(cls_idx)+1} '\n']);    
    for i=1:5
        %%
        correction = [1 1 0 0];
        
        image_id = ALLDATA(1,all([d(:,1) d(:,2) d(:,3) d(:,4)] == repmat(TRUE_RESP(loc1(i),:),size(d,1),1),2),1);
        fprintf(['FIG1 IMAGE ID: ' num2str(image_id) ' IMAGE LOC: ' num2str(loc1(i)) '\n']);
        
        im = imread([IMG_ROOT num2str(image_id) '.jpg']);
        fig1=figure(1); clf; imshow(im); hold on;
        for j=1:m
            if all(LOC_ORIG(loc1(i),:,j) >=0)
                rectangle('Position', LOC_ORIG(loc1(i),:,j) + correction,'EdgeColor','r', 'LineWidth', 2);
                fprintf(['LOC_ORIG ' num2str(j) ': ']); LOC_ORIG(loc1(i),:,j), 
            end;
        end;
        rectangle('Position', LOC_UPCR(loc1(i),:) + correction,'EdgeColor','g', 'LineWidth', 3,'LineStyle','--');
        fprintf('LOC_UPCR: '); LOC_UPCR(loc1(i),:),
        rectangle('Position', LOC_TRUE(loc1(i),:) + correction,'EdgeColor','b', 'LineWidth', 3);
        fprintf('LOC_TRUE: '); LOC_TRUE(loc1(i),:),
        %rectangle('Position', mean(LOC_ORIG(loc1(i),:,:),3),'EdgeColor','g', 'LineWidth', 2);
        print('-depsc', ['plots/icml/bbox_images/' CLASS_DICT{classes(cls_idx)+1} num2str(image_id) '.eps']);
        
        image_id = ALLDATA(1,all([d(:,1) d(:,2) d(:,3) d(:,4)] == repmat(TRUE_RESP(loc2(i),:),size(d,1),1),2),1);
        fprintf(['FIG2 IMAGE ID: ' num2str(image_id) ' IMAGE LOC: ' num2str(loc1(i)) '\n']);
        im = imread([IMG_ROOT num2str(image_id) '.jpg']);
        fig2=figure(2); clf; imshow(im); hold on;
        for j=1:m
            if all(LOC_ORIG(loc2(i),:,j) >=0)
                rectangle('Position', LOC_ORIG(loc2(i),:,j) + correction,'EdgeColor','r', 'LineWidth', 2);
                fprintf(['LOC_ORIG ' num2str(j) ': ']); LOC_ORIG(loc2(i),:,j), 
            end;
        end
        rectangle('Position', LOC_UPCR(loc2(i),:) + correction,'EdgeColor','g', 'LineWidth', 3,'LineStyle','--');
        fprintf('LOC_UPCR: '); LOC_UPCR(loc2(i),:),
        rectangle('Position', LOC_TRUE(loc2(i),:) + correction,'EdgeColor','b', 'LineWidth', 3);
        fprintf('LOC_TRUE: '); LOC_TRUE(loc2(i),:),
        %rectangle('Position', mean(LOC_ORIG(loc2(i),:,:),3),'EdgeColor','g', 'LineWidth', 2);
        print('-depsc', ['plots/icml/bbox_images/' CLASS_DICT{classes(cls_idx)+1} num2str(image_id) '.eps']);
        
        %%
        %fprintf('PAUSE\n'); pause;            
    end
end; % FOR EACH CLASS

