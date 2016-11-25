% function [y_best, w_best] = ER_BestRegressor(y_true, Z)
% Find the best predictor in the ensemble
function [y_best, w_best] = ER_BestRegressor(y_true, Z)
    mse = @(x) (mean((y_true-x).^2));
    [m,n] = size(Z);

    MSEs = zeros(m,1);
    for i=1:m
        MSEs(i) = mse(Z(i,:));
    end;
    best_idx = find(MSEs == min(MSEs),1);
    %w_best = (MSEs == min(MSEs));
    w_best = zeros(m,1); w_best(best_idx) = 1;
    %y_best = Z(MSEs == min(MSEs), :);
    y_best = Z(best_idx, :);
    y_best = y_best';
end