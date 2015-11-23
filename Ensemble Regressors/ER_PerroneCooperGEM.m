function [y_pred,w,C] = ER_PerroneCooperGEM(Ztrain, ytrain, Ztest)
    n_train = numel(ytrain);
    m = size(Ztrain, 1);
    misfit_sgem = Ztrain - repmat(ytrain,m,1);
    C = (misfit_sgem * misfit_sgem') / n_train;
    Ci = inv(C);
    w = zeros(m,1);
    for i=1:m
        w(i) = sum(Ci(i,:)) / sum(sum(Ci));
    end;
    y_pred = w' * Ztest;
end