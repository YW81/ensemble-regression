% function [y_pred, w] = ER_UnsupervisedDiagonalGEM(Z, Ey)
% Functions:
% 1. Estimates y by simple averaging. 
% 2. Estimate MSE of every regressor given y=simple averaging.
% 3. Set weights as inverse \hat MSE.
% 4. Make new predictions.
function [y_pred, w] = ER_UnsupervisedDiagonalGEM(Z, Ey)

    % basic variables
    [m,n] = size(Z);    
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i

    % Get mean prediction
    y_mean = ER_MeanWithBiasCorrection(Z, Ey);
    
    % Estimate MSE per regressor
    MSEs = mean((repmat(y_mean',m,1) - Z0).^2,2);
    
    % Calculate unsupervised equivalent of 1/MSE weighting
    w = 1 ./ MSEs;
    w = w ./ sum(w);
    y_pred = Ey + Z0' * w;
end