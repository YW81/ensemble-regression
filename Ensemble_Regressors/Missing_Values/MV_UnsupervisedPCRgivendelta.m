% function [y_predictions, idxs_of_regressors_used] = MV_UnsupervisedPCRstar(Z, Ey, Ey2, threshold_m, threshold_n)
%
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan. If two predictors have less than threshold_n elements in common, we assume Cij cannot be
% calculated accurately and ignore that one of them completely. We ignore the one with a smaller
% total number of predictions.
% The function returns reg_idxs which is a vector containing the indexes of the regressors that were
% used to make the predictions (had enough common elements with the other predictors in the ensemble
% to accurately estimate Cij).
function [y_pred, deltastar, idxs_of_regressors_used] = MV_UnsupervisedPCRgivendelta(y_true,Z, Ey, Ey2, threshold_m, threshold_n)

    % basic variables
    [C, Z, idxs_of_regressors_used ] = calc_C_with_missing_values( Z, threshold_n );
    if length(idxs_of_regressors_used) < 2
        throw(MException('MV_UnsupervisedPCRstar:tooSparse', ...
                         'Not enough common predictions to calculate the covariance terms. Try smaller threshold_n.'))
    end;

    n = size(Z,2);
    Z0 = Z - nanmean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    var_y = Ey2 - Ey^2;
    mse = @(x) ((nanmean((y_true'-x).^2))/var_y);
    y_centered = y_true - nanmean(y_true);
    
    acc_delta = 0;
    y_pred = nan*ones(n,1);
    for i=1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Z(:,i)));   % indices of experts that provided prediction on stock i
        [curC,subidxs] = get_largest_dense_submatrix(C(idxs,idxs)); % remove NaNs
        idxs = idxs(subidxs); % update idxs, remove unchosen indexes

        m = numel(idxs);

        if m < threshold_m
            continue; 
        end;

        samples_idxs = find(all(~isnan(Z(idxs,:))));
        data = Z0(idxs,samples_idxs);
        oracle_pred = nan*zeros(n,1);
        oracle_pred(samples_idxs) = Ey + Z0(idxs,samples_idxs)' * (data' \ y_centered(samples_idxs)');
        delta = min(mse(oracle_pred),1); % occasionaly mse_oracle > 1. It happens when there are outliers.
        acc_delta = acc_delta+delta;
        % Get leading eigenvector and eigenvalue of the largest dense submatrix of C(idxs,idxs)
        [v_1,lambda_1] = eigs(curC,1,'lm');
        t_sign = sign(sum(v_1));
        t = t_sign * sqrt((1-delta)*var_y / lambda_1);

        % Calculate predictions
        w = t * v_1;
        y_pred(i) = Ey + Z0(idxs,i)' * w;
    end; % sample i
    fprintf('average delta = %g\n',acc_delta/n);
end