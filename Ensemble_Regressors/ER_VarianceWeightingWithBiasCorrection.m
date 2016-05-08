% Unsupervised estimator with weights that are inversly proportional to the variance
function [y_pred, beta] = ER_VarianceWeightingWithBiasCorrection(Z, Ey)
    [m,n] = size(Z);
    b_hat = mean(Z,2) - Ey;
    Zc = Z - b_hat * ones(1,n);
    var_i = var(Zc,[],2);
    w = var_i / sum(var_i);
    
    %% Calculate oracle predictions
    y_pred = Zc'*w;
    beta = [0;w];
end