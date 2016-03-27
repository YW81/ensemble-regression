clear all;
load 'parsed_data_matrix.mat'

%% Init Data
% Fix Data Matrix
Z = Z';
Z(Z == 0) = NaN;
%Z = Z(sum(~isnan(Z),2) > 20, :);
m = size(Z,1);
n = size(y,1);

%% Init Assumptions
Ey = mean(y);
var_y = var(y);
CO_OCCURRENCE_THRESHOLD = 50;
MIN_NUMBER_OF_REGRESSORS = 10;

%% Estimation

b_hat = nanmean(Z,2) - Ey;
% % Z matrix completion
% for i=1:m
%     for j=1:n
%         if isnan(Z(i,j))
%             Z(i,j) = b_hat(i);
%         end;
%     end;
% end


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
% calculate the prediction one stock at a time
for i=1:n
    % choose only regressors with some recommendation for the current stock.
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

fprintf('\n\n');
fprintf('Correlation between predictions and true response: %g\n', corr(yd,yd_pred));
fprintf('%50s\t L1 Error: %g,\t Std %6.03g\t Number of predictions: %d\n', ...
  'SECOND MOMENT META-REGRESSOR', mean(abs(yd - yd_pred)), std(abs(yd - yd_pred)), length(yd_pred));
plot([min(yd) max(yd)], [min(yd) max(yd)], '--'); hold on; plot(yd,yd_pred,'x'); hold off;


%% Find best analyst
for i=1:m
    y_analyst = Z(i,:)';
    idxs = find(~isnan(y_analyst));
    
    % only rank analysts with the minimal number of predictions
    if length(idxs) < CO_OCCURRENCE_THRESHOLD
        analyst_avg_error(i) = inf;
        analyst_error_stdev(i) = inf;
        continue;
    end;
    
    analyst_avg_error(i) = mean(abs(y_analyst(idxs) - y(idxs)));
    analyst_error_stdev(i) = std(abs(y_analyst(idxs) - y(idxs)));
end;
[~, top_analysts_idxs] = sort(analyst_avg_error);
fprintf('Top analysts: \n=============\n');
for i=1:5
    fprintf('%50s\t L1 Error: %g,\t Std %6.03g\t Number of predictions: %d\n', ...
        labels_analysts{top_analysts_idxs(i)}, ...
        analyst_avg_error(top_analysts_idxs(i)), analyst_error_stdev(top_analysts_idxs(i)), ...
        sum(~isnan(Z(top_analysts_idxs(i),:))));
end;