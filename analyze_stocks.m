clear all;
load '~/code/github/stocks/data/parsed_data_matrix.mat'

%% Init Data
% Fix Data Matrix
Z = Z';
Z(Z == 0) = NaN;
m = size(Z,1);
n = size(y,1);

%% Init Assumptions
Ey = mean(y);
var_y = var(y);
CO_OCCURRENCE_THRESHOLD = 20;
MIN_NUMBER_OF_REGRESSORS = 10;

%% Estimation

b_hat = nanmean(Z,2) - Ey;

% calculate covariance with missing values
Sigma = zeros(m);
for i=1:m
    f_i_tilde = Z(i,:) - b_hat(i);
    for j=1:m
        f_j_tilde = Z(j,:) - b_hat(i);
        
        counter = 0;
        for k=1:n
            if ~isnan(Z(i,k)) && ~isnan(Z(j,k))
                % first sum the elements
                Sigma(i,j) = Sigma(i,j) + f_i_tilde(k)*f_j_tilde(k);
                counter = counter + 1;
            end;
        end;
        if counter > CO_OCCURRENCE_THRESHOLD
            % now divide to get the mean
            Sigma(i,j) = Sigma(i,j) / counter;
        else
            Sigma(i,j) = 0;
        end;
    end;
end;

%% Prediction
y_pred = zeros(size(y));
for i=1:n
    non_zero_regressor_idxs = find(~isnan(Z(:,i)));
    non_zero_regressor_idxs = non_zero_regressor_idxs(any(Sigma(non_zero_regressor_idxs, non_zero_regressor_idxs)));
    if length(non_zero_regressor_idxs) < MIN_NUMBER_OF_REGRESSORS
        y_pred(i) = NaN;
        continue;
    end;
           
    Sigma_tilde = Sigma(non_zero_regressor_idxs, non_zero_regressor_idxs);
    inv_Sigma_tilde = pinv(Sigma_tilde);    
    
    % \w = \Sigma^{-1} [1 \cdots 1]^T\mathrm{Var}(y)
    w = inv_Sigma_tilde * ones(length(non_zero_regressor_idxs),1) * var_y;
    w_0 = Ey * (1 - sum(w));
        
    % $\hat y = w_0 + \sum_{i=1}^m w_i \big( f_i - \hat b_i \big)$
    y_pred(i) = w_0;
    for j = 1:length(non_zero_regressor_idxs)
        y_pred(i) = y_pred(i) + w(j) * (Z(non_zero_regressor_idxs(j), i) - b_hat(non_zero_regressor_idxs(j)));
    end;
end;

yd = y(~isnan(y_pred));
yd_pred = y_pred(~isnan(y_pred));

fprintf('Number of predictions: %d\n', length(yd_pred));
fprintf('Correlation between predictions and true response: %g\n', corr(yd,yd_pred));
fprintf('Average absolute error distance: %g\n', norm(yd - yd_pred,1) / length(yd_pred));
plot([min(yd) max(yd)], [min(yd) max(yd)], '--'); hold; plot(yd,yd_pred,'x');


%% Find best analyst
for i=1:m
    y_analyst = Z(i,:)';
    idxs = find(~isnan(y_analyst));
    analyst_avg_error(i) = mean(abs(y_analyst(idxs) - y(idxs)));
end;