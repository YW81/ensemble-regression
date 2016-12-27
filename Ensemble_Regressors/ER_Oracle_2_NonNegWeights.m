% function [y_pred, w] = ER_Oracle_2_Unbiased(y_true, Z)
% Oracle of the form y = Ey + sum_i ( w_i (f_i-mu_i) ).
% w_0 = beta(1), w_i = beta(i+1)
function [y_pred, w] = ER_Oracle_2_NonNegWeights(y_true, Z)
    Ey = mean(y_true);
    Z0 = bsxfun(@minus,Z,mean(Z,2));
    y_centered = y_true - Ey;
    
    % w = Z0'\y_centered';
    % min_w norm(Z0'*w - y_centered) s.t. w_i >= 0
    w=lsqnonneg(Z0',y_centered');
    
    
    %% Calculate oracle predictions
    y_pred = Ey + Z0'*w;
end