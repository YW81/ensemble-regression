% function [y_pred, idxs_of_regressors_used] = MV_UnsupervisedPCRstar(Z, Ey, Ey2, threshold_m, threshold_n)
%
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan. If two predictors have less than threshold_n elements in common, we assume Cij cannot be
% calculated accurately and ignore that one of them completely. We ignore the one with a smaller
% total number of predictions.
% The function returns reg_idxs which is a vector containing the indexes of the regressors that were
% used to make the predictions (had enough common elements with the other predictors in the ensemble
% to accurately estimate Cij).
function [y_pred, idxs_of_regressors_used] = MV_UnsupervisedPCRstar(Z, Ey, Ey2, threshold_m, threshold_n, delta)
    if ~exist('delta','var')
        delta = 0;
    end;

    % basic variables
    [C, Z, idxs_of_regressors_used ] = calc_C_with_missing_values( Z, threshold_n );
    if length(idxs_of_regressors_used) < 2
        throw(MException('MV_UnsupervisedPCRstar:tooSparse', ...
                         'Not enough common predictions to calculate the covariance terms. Try smaller threshold_n.'))
    end;

    n = size(Z,2);
    y_pred = nan*ones(n,1);
    Z0 = Z - nanmean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    var_y = Ey2 - Ey^2;

    for i=1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Z(:,i)));   % indices of experts that provided prediction on stock i
        [curC,subidxs] = get_largest_dense_submatrix(C(idxs,idxs)); % remove NaNs
        idxs = idxs(subidxs); % update idxs, remove unchosen indexes
        
        m = numel(idxs);

        if m < threshold_m
            continue; 
        end
    
        % Get leading eigenvector and eigenvalue of the largest dense submatrix of C(idxs,idxs)
        [v_1,lambda_1] = eigs(curC,1,'lm');
        t_sign = sign(sum(v_1));
        t = t_sign * sqrt((1-delta)*var_y / lambda_1);

        % Calculate predictions
        w = t * v_1;
        y_pred(i) = Ey + Z0(idxs,i)' * w;
    end;
end