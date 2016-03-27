function [y_pred,w_new,C] = ER_unsupervisedGEM(Z, b_hat, prev_w)
    [m n] = size(Z);

    % update prediction
    y_pred = (Z - repmat(b_hat,1,n))' * prev_w;

    % update weights
    misfit = repmat(y_pred',m,1) - (Z - repmat(b_hat,1,n));
    
    if any(any(isnan(misfit)))
        w_new = prev_w; % keep the original (mean) weighting
        C = 0;
        
    else % no NaNs in data
        
        C = misfit * misfit' / n;
        Cinv = pinv(C);
        w_new = sum(Cinv,1)' ./ sum(sum(Cinv)); % w_i = sum_j(Cinv_ij) / sum(sum(Cinv))
    end;
end