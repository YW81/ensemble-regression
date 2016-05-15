% function [y_pred,beta,rho_hat] = ER_SecondMoment(Z, Ey, Ey2, mu, b_hat, R_hat)
% Calculate the second moment estimation of y
% Input:
%   Z = data matrix
%   Ey = E[y] (y being the response variable)
%   Ey2 = E[y^2]
%   
%   mu = mean(Z,2);    % Mean estimate of each regressor mu(i) = mean(f(i))
%   b_hat = mean(Z,2) - Ey; % Estimate of the bias of each regressor
%   R_hat = Sigma + mu*mu'; % Estimated uncentered covariance
%
function [y_pred,beta] = ER_SecondMomentOracle(Z, Ey, Ey2, y)
    [m,n] = size(Z);

    b_hat = mean(Z,2)-Ey; 
    var_y = Ey2 - Ey.^2;

    C = cov(Z');
    e = Z - b_hat * ones(1,n) - ones(m,1) * y; % note this is using estimated bias, and not real bias
    g = mean(e .* repmat(y,m,1),2);
%    % Test with noisy g, see how it affects estimation (answer - dramatic errors)    % BOAZ STEP 1
%     noise = 1+ (.01 * rand(m,1) - .005);
%     g = g .* noise;

    % Test with C_tilde instead of C
    Ci_tilde = pinv(C,.01);
    w = Ci_tilde * (ones(m,1) * var_y + g); % BOAZ STEP 2
    %w = (C\(ones(m,1)*var_y + g));
    w_0 = Ey * (1 - sum(w));
    
    y_pred = w_0 + (Z - repmat(b_hat,1,size(Z,2)))'*w;
    beta = [w_0;w];
end