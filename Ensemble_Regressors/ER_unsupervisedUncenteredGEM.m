function [y_pred,w_new] = ER_unsupervisedUncenteredGEM(Z, b_hat, R_hat, prev_w)
    % find y_hat
    %y_uncentered_gem{i} = Z' * w_uncentered_gem{i} - repmat(w_uncentered_gem{i}',n,1)*b_hat;
    n = size(Z,2);
    y_pred = (Z - repmat(b_hat,1,n))' * prev_w;
    
    % update w
    w_new = inv(R_hat)*Z*y_pred / n;
    w_new = w_new / sum(w_new); % keep sum(w) = 1. TODO: Is this needed?
end