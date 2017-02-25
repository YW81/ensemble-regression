% function [y_pred, w] = ER_UnsupervisedGEM_with_rho_estimation(Z,Ey)
% Unsupervised GEM, estimator of the form y = Ey + sum_i ( w_i (f_i-mu_i) ) where the sum of the weights equals 1
% and rho is estimated using the mean predicted response.
%
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan. If two predictors have less than threshold_n elements in common, we assume Cij cannot be
% calculated accurately and ignore that one of them completely. We ignore the one with a smaller
% total number of predictions.
% The function returns reg_idxs which is a vector containing the indexes of the regressors that were
% used to make the predictions (had enough common elements with the other predictors in the ensemble
% to accurately estimate Cij).
function [ y_pred, idxs_of_regressors_used ] = MV_UnsupervisedGEM_with_rho_estimation(Z, Ey, threshold_m, threshold_n)
    [C, Z, idxs_of_regressors_used ] = calc_C_with_missing_values( Z, threshold_n );
    if length(idxs_of_regressors_used) < 2
        throw(MException('MV_UnsupervisedGEMwithRhoEst:tooSparse', ...
                         'Not enough common predictions to calculate the covariance terms. Try smaller threshold_n.'))
    end;

    n = size(Z,2);
    y_pred = nan*ones(n,1);
    Z0 = Z - nanmean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i

    rho = nanmean(bsxfun(@times,bsxfun(@minus, Z, nanmean(Z,2)),nanmean(Z)),2);
    
    for i=1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Z(:,i)));   % indices of experts that provided prediction on stock i
        [curC,subidxs] = get_largest_dense_submatrix(C(idxs,idxs)); % remove NaNs, and update idxs
        idxs = idxs(subidxs); % update idxs, remove unchosen indexes
        m = numel(idxs);

        %% Pruning - Not so good for highly sparse data (might get NaNs in the covariance matrix)
%         idxs_removed = []; idxs_selected = idxs;
%         while cond(C(idxs_selected,idxs_selected)) > 1e6
%             best_cond = Inf;
%             for i=idxs_selected
%                 cur_idxs = idxs_selected;
%                 cur_idxs(cur_idxs == i) = [];
%                 if cond(C(cur_idxs,cur_idxs)) < best_cond
%                     best_cond = cond(C(cur_idxs,cur_idxs));
%                     best_idx = i;
%                 end;
%             end;
%             idxs_selected(idxs_selected == best_idx) = [];
%             idxs_removed = [idxs_removed best_idx];
%         end;
% 
%         C_pruned = C(idxs_selected,idxs_selected);
%         new_m = length(idxs_selected);

        %% Use diagonal loading instead of pruning, because if the data is sparse, the
        %  covariance matrix might already be small, and pruning will make it even smaller.
        C_pruned = C(idxs,idxs);
        Cinv = pinv(C_pruned,.01);
        idxs_selected = idxs;        
        new_m = m; % this is the new guy, same as the old.


        if new_m < threshold_m
            continue; 
        end
        
        %w = C\(rho - ones(m,1)*(ones(1,m)*(C\rho)-1)/sum(sum(pinv(C))));
        w = Cinv*(rho(idxs_selected) - ones(new_m,1)*(ones(1,new_m)*(Cinv*rho(idxs_selected))-1)/sum(sum(Cinv)));
        y_pred(i) = Ey + Z0(idxs_selected,i)'*w;
    end;
end