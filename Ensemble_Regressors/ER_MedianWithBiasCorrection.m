% function [y_pred, beta] = ER_MedianWithBiasCorrection(Z, Ey)
% Unsupervised estimator with weights that are inversly proportional to the variance
function y_pred = ER_MedianWithBiasCorrection(Z, Ey)
    n = size(Z,2);
    b_hat = mean(Z,2) - Ey;
    Zc = Z - b_hat * ones(1,n);
    y_pred = zeros(n,1);
    for i=1:n
        y_pred(i) = median(Zc(:,i));
    end;
end