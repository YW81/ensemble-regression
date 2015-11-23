% Input:
%   Z = data matrix
%   Ey = E[y] (y being the response variable)
%   Ey2 = E[y^2]
%   
%   mu = mean(Z,2);    % Mean estimate of each regressor mu(i) = mean(f(i))
%   b_hat = mean(Z,2) - Ey; % Estimate of the bias of each regressor
%   R_hat = Sigma + mu*mu'; % Estimated uncentered covariance
%
function [y_pred,w,rho_hat] = ER_SecondMoment(Z, Ey, Ey2, mu, b_hat, R_hat)
    rho_hat = (Ey2+Ey^2 + b_hat .* Ey);
    w = inv(R_hat - b_hat*mu' - mu*b_hat' + b_hat*b_hat') *(rho_hat - Ey * b_hat);
    w = w ./ sum(w);
    y_pred = (Z - repmat(b_hat,1,size(Z,2)))'*w;
end