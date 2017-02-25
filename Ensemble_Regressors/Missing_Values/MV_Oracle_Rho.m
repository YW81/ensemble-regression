% function [y_pred, w] = ER_Oracle_Rho(y_true, Z)
% Oracle of the form y = Ey + sum_i ( w_i (f_i-mu_i) ) where the sum of the weights equals 1
function [y_pred, reg_idxs] = MV_Oracle_Rho(y_true, Z, threshold_m, threshold_n)
    [C, Z, reg_idxs ] = calc_C_with_missing_values( Z, threshold_n );
    if length(reg_idxs) < 2
        throw(MException('MV_UnsupervisedPCRstar:tooSparse', ...
                         'Not enough common predictions to calculate the covariance terms. Try smaller threshold_n.'))
    end;

    n = size(Z,2);
    y_pred = nan*ones(n,1);
    Z0 = Z - nanmean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    Ey = mean(y_true);
    rho = nanmean(bsxfun(@times,bsxfun(@minus, Z, nanmean(Z,2)),y_true),2);
    
    for i = 1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Z(:,i)));   % indices of experts that provided prediction on stock i
        m = numel(idxs);

        if m < threshold_m
            continue; 
        end
        %w = C\(rho - ones(m,1)*(ones(1,m)*(C\rho)-1)/sum(sum(pinv(C))));
        Cinv = pinv(C(idxs,idxs),1e-5);
        w = Cinv*(rho(idxs) - ones(m,1)*(ones(1,m)*(Cinv*rho(idxs))-1)/sum(sum(Cinv)));
        y_pred(i) = Ey + Z0(idxs,i)'*w;
    end;
end