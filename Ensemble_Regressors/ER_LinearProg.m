% function [y_pred,beta,rho_hat] = ER_SecondMoment(Z, Ey, Ey2)
% Calculate the second moment estimation of y
% Input:
%   Z = data matrix
%   Ey = E[y] (y being the response variable)
%   Ey2 = E[y^2]
%   
function [y_pred,beta] = ER_LinearProg(Z, Ey, Ey2)


    [m n] = size(Z);
    var_y = Ey2 - Ey.^2;

    b_hat = mean(Z,2)-Ey; 
    Z_new = Z-b_hat * ones(1,n);
    C = cov(Z_new');
    
    % constraints
    lb = []; ub = [];
    Aeq = [1/Ey ones(1,m);[zeros(m,1), C/var_y]]; beq = ones(m+1,1);
    A = [zeros(m,1), -eye(m)]; b = zeros(m,1); % w_i >= 0 for i=1..m
    f = [];
    x0 = ones(m,1)/m; % start with even weights
    
    [beta,fval] = linprog(f,A,b,Aeq,beq,lb,ub,x0);
    w_0 = beta(1); w = beta(2:end);
    
    y_pred = w_0 + (Z_new)'*w;
end