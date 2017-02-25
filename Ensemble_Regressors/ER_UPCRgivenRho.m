% function [y_pred, w, t, residual] = ER_SpectralApproach(Z, Ey)
function [y_pred, w, t, residual] = ER_UPCRgivenRho(Z, Ey, Ey2, rho_est)

    % basic variables
    [m,n] = size(Z);    
    var_y = Ey2 - Ey^2;
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z'); % cov(Z') == cov(Zc') == cov(Z0')...

    % Get leading eigenvector and eigenvalue
    [v_1,lambda_1] = eigs(C,1,'lm');
    %t_sign = sign(sum(v_1));
    t = v_1' * rho_est / lambda_1;
    %t = t_sign * sqrt((1-deltastar)*var_y / lambda_1);
    
    % Calculate predictions
    w = t * v_1;
    y_pred = Ey + Z0' * w;
    
    residual = norm(w*lambda_1 - rho_est); 
    
end