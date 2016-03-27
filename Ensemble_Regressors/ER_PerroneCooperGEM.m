function [y_pred,w,C] = ER_PerroneCooperGEM(Ztrain, ytrain, Ztest)
    n_train = numel(ytrain);
    m = size(Ztrain, 1);
    
    % Calculate Misfit
    misfit_sgem = Ztrain - repmat(ytrain,m,1);
    
    % Calculate Covariance C
    C = (misfit_sgem * misfit_sgem') / n_train;
%     if rcond(C) < 1e-5 % diagonal loading
%         C = C + eye(size(C)) * 1e-5;
%     end;
    
    Ci = inv(C);
    
    % Calculate Weights w
    w = zeros(m,1);
    for i=1:m
        w(i) = sum(Ci(i,:)) / sum(sum(Ci));
    end;
    
    % Calculate Predicted Response y
    y_pred = w' * Ztest;
end