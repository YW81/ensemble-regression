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
function [y_pred_RE, y_pred_WRE, avg_delta_RE, avg_delta_WRE, idxs_of_regressors_used] = MV_UnsupervisedPCRdeltaWRE(Z, Ey, Ey2, threshold_m, threshold_n)

    % basic variables
    [C, Z, idxs_of_regressors_used ] = calc_C_with_missing_values( Z, threshold_n );
    if length(idxs_of_regressors_used) < 2
        throw(MException('MV_UnsupervisedPCRstar:tooSparse', ...
                         'Not enough common predictions to calculate the covariance terms. Try smaller threshold_n.'))
    end;

    n = size(Z,2);
    Z0 = Z - nanmean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    var_y = Ey2 - Ey^2;
    mse = @(x) (nanmean((y_true-x).^2))/var_y;    

    y_pred_RE = nan*ones(n,1); y_pred_WRE = nan*ones(n,1); delta_RE = nan*ones(n,1); delta_WRE = nan*ones(n,1); 
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

        % accumulate deltas
        delta_RE(i) = max(0,1 - lambda_1*sum(v_1)^2 / (var_y*m^2));
        delta_WRE(i) = max(0,1 - lambda_1 / (var_y*sum(v_1)^2));
        t_sign = sign(sum(v_1));
        t_RE = t_sign * sqrt((1-delta_RE(i))*var_y / lambda_1);
        t_WRE = t_sign * sqrt((1-delta_WRE(i))*var_y / lambda_1);

        % Calculate predictions
        w_RE = t_RE * v_1;
        w_WRE = t_WRE * v_1;
        y_pred_RE(i) = Ey + Z0(idxs,i)' * w_RE;
        y_pred_WRE(i) = Ey + Z0(idxs,i)' * w_WRE;
    end; % sample i
    fprintf('avg delta_RE=%g,avg delta_WRE=%g\n',nanmean(delta_RE), nanmean(delta_WRE));
    plot(delta_RE,delta_WRE,'b.'); 
    y = delta_WRE(~isnan(delta_WRE)); x = delta_RE(~isnan(delta_RE));
    lsline; title(sprintf('WRE = %.2f + %.2f RE',[ones(length(x),1) x] \ y));
    grid on; grid minor; axis tight; xlabel('\delta_{RE}'); ylabel('\delta_{WRE}');
end