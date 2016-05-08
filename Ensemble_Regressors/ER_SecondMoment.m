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
function [y_pred,beta] = ER_SecondMoment(Z, Ey, Ey2)
%     rho_hat = Ey2 + b_hat .* Ey; % + Ey^2 (was in the old code, bug?)
%     rho_tilde = rho_hat - b_hat .* Ey; % = Ey2 * ones(m,1)
%     % w = inv(R_hat - b_hat*mu' - mu*b_hat' + b_hat*b_hat') *(rho_hat - Ey * b_hat);
%     % R_tilde = R_hat - b_hat*mu' - mu*b_hat' + b_hat * b_hat';
%     % Ef_tilde = mu - b_hat; % (mu = Ef)
    [m,n] = size(Z);
    I_m = eye(m);
    b_hat = mean(Z,2)-Ey; 
    
%     R_tilde = R_hat - Ey * b_hat * ones(1,m) - mu * b_hat';
%     %w = inv(I_m + inv(R_tilde) * Ey * ones(m)) * inv(R_tilde) * (rho_hat - Ey * mu);
    var_y = Ey2 - Ey.^2;

%% Using Cij = cov(f_i, f_j)
     C = cov(Z');
     %C = 1/n*(Z - repmat(b_hat,1,n))* (Z - repmat(b_hat,1,n))' - Ey2;
%      %Z_new = Z-b_hat * ones(1,n); 
%      %C = 1/n * Z_new * (Z_new'); 
% %     if rcond(C) < 1e-5 % diagonal loading
% %         C = C + eye(size(C)) * 1e-5;
% %     end;
    g = 0; %g = .5*(mean((Z - repmat(b_hat,1,n)).^2,2) - Ey2); % -\E[e.^2] which is unknown
    w = (C\(ones(m,1)*var_y + g));
%     %w = var_y * (C+1e-5*eye(size(C))) \ ones(m,1); 

%% Using Eij = cov(e_i, e_j)
%     Z_new = Z-b_hat * ones(1,n);
%     E = Z_new*Z_new'/n + Ey2;
%     w = (ones(m) + E/var_y) \ ones(m,1);

%% calculate predictions
    w_0 = Ey * (1 - sum(w));
    
    y_pred = w_0 + (Z - repmat(b_hat,1,size(Z,2)))'*w;
    beta = [w_0;w];
end