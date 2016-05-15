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
    [m,n] = size(Z);
    I_m = eye(m);
    b_hat = mean(Z,2)-Ey; 
    
    var_y = Ey2 - Ey.^2;

%% Using Cij = cov(f_i, f_j)
    C = cov(Z');
    g = 0; %g = .5*(mean((Z - repmat(b_hat,1,n)).^2,2) - Ey2); % -\E[e.^2] which is unknown

    Ci_tilde = pinv(C,.01);
    w = Ci_tilde * (ones(m,1) * var_y + g); % BOAZ STEP 2 is for the oracle, but why not do this for the non-oracle?
%    w = (C\(ones(m,1)*var_y + g));
    
%% calculate predictions
    w_0 = Ey * (1 - sum(w));
    
    y_pred = w_0 + (Z - repmat(b_hat,1,size(Z,2)))'*w;
    beta = [w_0;w];
end