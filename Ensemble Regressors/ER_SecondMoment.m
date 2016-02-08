% Input:
%   Z = data matrix
%   Ey = E[y] (y being the response variable)
%   Ey2 = E[y^2]
%   
%   mu = mean(Z,2);    % Mean estimate of each regressor mu(i) = mean(f(i))
%   b_hat = mean(Z,2) - Ey; % Estimate of the bias of each regressor
%   R_hat = Sigma + mu*mu'; % Estimated uncentered covariance
%
function [y_pred,beta,rho_hat] = ER_SecondMoment(Z, Ey, Ey2, mu, b_hat, R_hat)
    rho_hat = Ey2 + b_hat .* Ey; % + Ey^2 (was in the old code, bug?)
    rho_tilde = rho_hat - b_hat .* Ey; % = Ey2 * ones(m,1)
    % w = inv(R_hat - b_hat*mu' - mu*b_hat' + b_hat*b_hat') *(rho_hat - Ey * b_hat);
    % R_tilde = R_hat - b_hat*mu' - mu*b_hat' + b_hat * b_hat';
    % Ef_tilde = mu - b_hat; % (mu = Ef)
    m = size(Z,1);
    I_m = eye(m);
    
    R_tilde = R_hat - Ey * b_hat * ones(1,m) - mu * b_hat';
    %w = inv(I_m + inv(R_tilde) * Ey * ones(m)) * inv(R_tilde) * (rho_hat - Ey * mu);
    var_y = Ey2 - Ey.^2;
    w = inv(cov(Z'))*ones(m,1)*var_y;
    w_0 = Ey * (1 - sum(w));
    
    y_pred = w_0 + (Z - repmat(b_hat,1,size(Z,2)))'*w;
    beta = [w_0;w];
end