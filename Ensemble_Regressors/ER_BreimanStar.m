function [y_pred,w,fval] = ER_BreimanStar(Ztrain, ytrain, Ztest)
% function [y_pred,beta,fval] = ER_BreimanStar(Ztrain, ytrain, Ztest)
% Minimize cost function as defined in "Stacked Regressions" / Breiman 1996 
% with the non-negativity constraint which Breiman found to be most accurate.
% I call this "BreimanStar" since in the original paper Breiman utilized Leave-One-Out Cross
% Validation on the data, and here we use all the data for the optimization problem.
%
% If we wanted to actually implement Breiman, we would need to construct the ensemble predictions Z
% using leave-one-out on the original X's. (Meaning - in the python code that generates the dataset,
% each regressor in the ensemble needs to be reconstructed for every sample, leaving the current
% sample out.

    % Init params
    [m,n_train] = size(Ztrain);
    [m_test, n] = size(Ztest);
    
    assert(m == m_test, 'Ztrain and Ztest must have the same number of rows')
    
    %b_hat = mean(Z,2) - Ey; % approximate bias
    %Zc = Ztrain - repmat(b_hat,1,n_train);
    
    % min sum_n (y_n - sum_k w_k Z_kn)^2 = min (y-Z'*w)'*(y-Z'*w) = min w'*Z'*Z*w - 2*(Z*y)'*w
    H = Ztrain*Ztrain';
    f = -Ztrain*ytrain';
    lb = zeros(m,1);  % constraining w_k>0
    
    options = optimoptions('quadprog', 'Display','None');
    [w, fval] = quadprog(H,f,[],[],[],[],lb,[],[],options);
    
    % Calculate Predicted Response y
    y_pred = Ztest' * w;
end