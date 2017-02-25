% function [inlier_idx,outlier_idx,MSE_hat, rho_hat] = subset_selection(y_true,Z,Ey,Ey2,method,loss)
% Method can be 'MSE' or 'rho'. Defaults to 'MSE'.
% Loss can be 'l2','l1','huber', and is passed to INDB as a parameter for the minimization loss
% function (INDB minimizes |Aa - Cshifted| where rho_i = g_2 + a_i, and Cshifted = C - g_2).
% values returns the values of the MSEs or rho's that were used for the subset selection
function [inlier_idx,outlier_idx,MSE_hat, rho_hat] = subset_selection(y_true,Z,Ey,Ey2,method,loss)
    if ~exist('method','var')
        method = 'MSE'
    end
    if ~exist('loss','var')  % loss can be 'l2','l1','huber'
        loss = 'l2';
    end;
    var_y = Ey2 - Ey^2;
    [m,n] = size(Z);
    C = cov(Z');
    
    inlier_idx = 1:m; outlier_idx = [];

    %% Estimate Rho
    [~,~,rho_IND] = ER_IndependentMisfits(Z,Ey, Ey2);
    [~,~,rhoIND_L1] = ER_IndependentMisfits(Z,Ey, Ey2,'l1');    
    [~,~,rho_INDB,~] = ER_IndependentMisfitsBayes(y_true,Z,Ey,Ey2,loss,0,true);
    
    %% MSE_i Estimation
    MSE_IND = zeros(m,1); MSE_INDB = zeros(m,1); MSE_IND_L1 = zeros(m,1);
    for i=1:m
        MSE_IND(i) = (var_y - 2*rho_IND(i) + C(i,i)) / var_y;
        MSE_IND_L1(i) = (var_y - 2*rhoIND_L1(i) + C(i,i)) / var_y;
        MSE_INDB(i) = (var_y - 2*rho_INDB(i) + C(i,i)) / var_y;
    end;

    %% TRYING TO REMOVE OUTLIERS
    if strcmp(method,'MSE')
        % remove large MSEs
        values = MSE_INDB;
        a1 = median(values); a2 = median(abs(values-a1)); 
        outlier_idx = ((values-a1)/a2 > 3); 
        inlier_idx = ((values-a1)/a2 <= 3); 
    else
        % remove small rhos
        values = rho_INDB; 
        %a1 = median(values); a2 = median(abs(values-a1)); 
        %outlier_idx = ((values-a1)/a2 < -3) | (values/var_y < 0.05); 
        
        a1 = max(values); 
        outlier_idx = ((values/a1)<(1/3)) | ((values/var_y) < 0.05); 
        inlier_idx = logical(1-outlier_idx); %((values-a1)/a2 >= -3); 
    end;
    
    if all(outlier_idx)
        outlier_idx = false(size(outlier_idx))
        inlier_idx = true(size(inlier_idx));
    end;
    
    [val, loc] = sort(values); 
    %((values-a1)/a2)'
    fprintf('RHO ENTRIES/MAX:');
    (values/a1)'
    %fprintf('OUTLIER IDX = '); find(outlier_idx)'

%     % Do a second round of outlier removal (Fix: this was only written for method='MSE')
%     if false %~isempty(find(outlier_idx))
%         [y_IND_new,w_IND_new,rho_IND_new] = ER_IndependentMisfits(Z(inlier_idx,:),Ey, Ey2);
%         MSE_IND_new = ones(m,1);
%         idxs = find(inlier_idx);
%         for i=1:length(idxs)
%             MSE_IND_new(idxs(i)) = (var_y - 2*rho_IND_new(i) + C(idxs(i),idxs(i))) / var_y;
%         end;
%         MSEhat = MSE_IND_new;
%         a1 = median(MSEhat(inlier_idx)); a2 = median(abs(MSEhat(inlier_idx)-a1)); 
%         fprintf('NEW z-score(MSEhat) :'); ((MSEhat(inlier_idx)-a1)/a2)'
%         outlier_idx = (outlier_idx) | ((MSEhat-a1)/a2 > 3); 
%         inlier_idx = (inlier_idx) & ((MSEhat-a1)/a2 <= 3);
%     end;
    
    %% INDB to get a good new estimate of values (MSE or rho)
    [~, ~,rho_hat, ~] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2, loss, 0, true);

    MSE_hat = zeros(m,1);
    for i=1:m
        MSE_hat(i) = (var_y - 2*rho_hat(i) + C(i,i)) / var_y;
    end;
end