function [cost] = cvx_opt_for_w(beta, m, n, Ey, var_y, C, Z1c, K1, K2, lambda_mean, lambda_var)
%function [cost] = cvx_opt_for_w(beta, m, n, Ey, var_y, C, Z1c, K1, K2, lambda_mean, lambda_var)


% find beta_prime - the updated weights
    w_prime = var_y*(C\ones(m,1)) + (C\(K1*beta))/n - (C\(ones(m,1)*beta'*K2*beta))/n;
    beta_prime = [Ey*(1-sum(w_prime)); w_prime ];
    
    % calculate the cost function
    cost = (Z1c'*beta - Z1c'*beta_prime)'*(Z1c'*beta - Z1c'*beta_prime)   ...
        + lambda_mean * (mean(Z1c'*beta_prime) - Ey)^2                    ...
        + lambda_var * (var(Z1c'*beta_prime) - var_y)^2;
end

