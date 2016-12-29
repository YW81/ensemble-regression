% function [y_pred, w] = ER_UnsupervisedDiagonalGEM(Z, Ey)
% Functions:
% 1. Estimates y by simple averaging. 
% 2. Estimate MSE of every regressor given y=simple averaging.
% 3. Set weights as inverse \hat MSE.
% 4. Make new predictions.
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan. If two predictors have less than threshold_n elements in common, we assume Cij cannot be
% calculated accurately and ignore that one of them completely. We ignore the one with a smaller
% total number of predictions.
function y_pred = MV_UnsupervisedDiagonalGEM(Z, Ey, threshold_m)
    %Ey = .5; Z = rand(10,100) + repmat(10*rand(10,1),1,100); Z(ceil(numel(Z)*rand(.5*numel(Z),1))) = nan;

    % basic variables
    [m,n] = size(Z);    
    y_pred = nan*ones(n,1);    
    Z0 = Z - nanmean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i

    % Get mean prediction
    y_mean = MV_MeanWithBiasCorrection(Z, Ey, threshold_m);
    
    % Estimate MSE per regressor
    MSEs = nanmean((repmat(y_mean',size(Z,1),1) - Z).^2,2); % Estimate MSE per regressor
    
    % Calculate unsupervised equivalent of 1/MSE weighting
    for i=1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Z(:,i)));   % indices of experts that provided prediction on stock i
        m = numel(idxs);

        if m < threshold_m
            continue; 
        end

        w = 1 ./ MSEs(idxs);
        w = w ./ sum(w);
        y_pred(i) = Ey + Z0(idxs,i)' * w;
    end;
end