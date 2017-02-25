% function [y_pred, w, t, residual] = ER_SpectralApproach(Z, Ey)
function [y_pred, w, t, residual] = ER_UPCRgivenRho2Components(Z, Ey, Ey2, rho_est)

    % basic variables
    [m,n] = size(Z);    
    var_y = Ey2 - Ey^2;
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z'); % cov(Z') == cov(Zc') == cov(Z0')...

    % Get leading eigenvector and eigenvalue
    %[V,D] = eigs(C,2,'lm');
    [V,D] = eig(C);
    v_1 = V(:,1); v_2 = V(:,2);
    lambda_1 = D(1,1); lambda_2 = D(2,2);
    %t_sign = sign(sum(v_1));
    t_1 = v_1' * rho_est / lambda_1;
    t_2 = v_2' * rho_est / lambda_2;
    %t = t_sign * sqrt((1-deltastar)*var_y / lambda_1);
    
    D = diag(D);
    idxs = D/sum(D) > .1; % Thresholding
    V = V(:,idxs); D = D(idxs);
    t = V' * rho_est ./ D;
    w = V * t;
    
    % Calculate predictions
    %w = t_1 * v_1 + t_2 * v_2;
    y_pred = Ey + Z0' * w;
    
    %residual = norm((v_1'*rho_est)*v_1 + (v_2'*rho_est)*v_2 - rho_est); 
    V = [V zeros(m,m-size(V,2))]; % zero all eigenvectors with eigenvalues <= .1
    residual = norm(rho_est - V*(V'*rho_est)); % The residual what's left over after subtracting the projection of rho_est on V
    
end