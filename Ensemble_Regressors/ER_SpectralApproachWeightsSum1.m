% function [y_pred, w, t] = ER_SpectralApproach(Z, Ey)
function [y_pred, w, t] = ER_SpectralApproachWeightsSum1(Z, Ey, Ey2)

    % basic variables
    [m,n] = size(Z);    
    var_y = Ey2 - Ey^2;
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z'); % cov(Z') == cov(Zc') == cov(Z0')...

    % Get leading eigenvector and eigenvalue
    [v_1,lambda_1] = eigs(C,1,'lm');
    t_sign = sign(sum(v_1));
    %t = t_sign * sqrt(var_y / lambda_1);
    t = t_sign*1/sum(abs(v_1));
    
    % Calculate predictions
    w = t * v_1;
    w = w * abs(lambda_1 / sum(eig(C)));    
    
    %w = w.*(w>1/sqrt(n)); w = w/sum(w);  % removes both negative & small entries
    
    %w = max(0,w);     w = w/sum(w);   % removes negative entries
    
    
    y_pred = Ey + Z0' * w;
end