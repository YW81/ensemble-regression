% function [y_pred, beta] = ER_linear_regression_oracle(y_true, Z)
% Linear regression oracle.
% w_0 = beta(1), w_i = beta(i+1)
function [y_pred, beta] = ER_linear_regression_oracle(y_true, Z)
warnstate=warning('off','MATLAB:rankDeficientMatrix');
    beta = mvregress([ones(size(Z,2),1) Z'], y_true');
warning(warnstate);
    w_0 = beta(1);
    w = beta(2:end);
    
    %% Calculate oracle predictions
    y_pred = w_0 + Z'*w;
end