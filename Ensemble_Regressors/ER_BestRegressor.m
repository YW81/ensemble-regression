% function [y_best, w_best] = ER_BestRegressor(y_true, Z)
% Find the best predictor in the ensemble
function [y_best, w_best] = ER_BestRegressor(y_true, Z)
    mse = @(x) (mean((y_true-x).^2));
    [m,n] = size(Z);

    MSEs = zeros(m,1);
    for i=1:m
        MSEs(i) = mse(Z(i,:));
    end;
    w_best = (MSEs == min(MSEs));
    y_best = Z(MSEs == min(MSEs), :);
    y_best = y_best';
end