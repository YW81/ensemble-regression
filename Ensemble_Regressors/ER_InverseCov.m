% Unsupervised estimator with weights that are inversly proportional to the variance
function [y_pred, beta] = ER_InverseCov(Z, Ey)
    [m,n] = size(Z);
    b_hat = mean(Z,2) - Ey;
    %Zc = Z - b_hat * ones(1,n);
    %C = cov(Zc');

    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z0');    
    
    Cinv = pinv(C,.01);
    rho = zeros(m,1); % assume no correlation between regressor error and response
    w = Cinv * (rho - ones(m,1) * ( (ones(1,m) * Cinv * rho - 1) / sum(sum(Cinv))));
    beta = [0;w];
    y_pred = Ey + Z0'*w;
%     var_i = var(Zc,[],2);
%     w = var_i / sum(var_i);
%     mu = mean(Z,2);
%     
%   
%     %% Calculate varw predictions
%     y_pred = Zc'*w;
%     beta = [0;w];
%     
%     
%     %% Notes:
%     Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
%     C = cov(Z0');    
%     r_true = Z0*y_true' / n;
%     lambda_true = ( (ones(1,m)*(C\r_true) ) - 1) / (ones(1,m)*(C\ones(m,1)))
%     w_true = C \ (r_true - lambda_true * ones(m,1))
%     y = Ey + w_true'*Z0;
%     mean((y_true - y).^2)
%     
%     [y_oracle, beta_oracle] = ER_linear_regression_oracle(y_true, Z);
%     mean((y_true' - y_oracle).^2)
%     % corr_true = r_true / var_y;
%     
%     % r = var_y * ones(m,1) / 2; % assume rho_i = .5*var_y
%     % w = C \ (r - ones(m,1) * (1 - ones(1,m)*(C\r)) / sum(sum(inv(C))))
%     clear r w y i; %clc;
%     r(:,1) = r_true; %var_y * ones(m,1) / 2 % assume rho_i = .5*var_y % option b: assume y=mean(Zc) and calculate rho_i based on that assumption
%     i = 1;
%     % w(:,i) = C \ (r(:,i) - ones(m,1) * (1 - ones(1,m)*(C\r(:,i))) / sum(sum(inv(C))))
%     lambda = ((1-Ey)*(ones(1,m)*(C\r(:,i))) - 1) / (ones(1,m)*(C\ones(m,1)))
%     w(:,i) = (C\(  (1-Ey)*r(:,i) - lambda*ones(m,1)  ))
%     sum(w)
%     y = Ey + w(:,i)'*Z0; % notice it's a row vector instead of a column vector like normal people do
%     mean((y_true - y).^2)
%     r(:,i+1) = mean(repmat(y,m,1) .* Zc,2)
%     i=i+1
    
end