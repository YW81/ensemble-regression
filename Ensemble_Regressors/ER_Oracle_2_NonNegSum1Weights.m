% function [y_pred, w] = ER_Oracle_2_Unbiased(y_true, Z)
% Oracle of the form y = Ey + sum_i ( w_i (f_i-mu_i) ).
% w_0 = beta(1), w_i = beta(i+1)
function [y_pred, w] = ER_Oracle_2_NonNegSum1Weights(y_true, Z)
    m = size(Z,1);
    Ey = mean(y_true);
    Z0 = bsxfun(@minus,Z,mean(Z,2));
    y_centered = y_true - Ey;
    
    % w = Z0'\y_centered';
    % min_w norm(Z0'*w - y_centered) s.t. w_i >= 0 AND sum w_i = 1
    A=-1*eye(m);b=zeros(m,1); % w_i >= 0
    Aeq = ones(1,m); beq = 1; % sum w_i = 1
    w = lsqlin(Z0',y_centered',A,b,Aeq,beq,[],[],[],optimoptions('lsqlin','Algorithm','active-set'));
    
    
    %% Calculate oracle predictions
    y_pred = Ey + Z0'*w;
end