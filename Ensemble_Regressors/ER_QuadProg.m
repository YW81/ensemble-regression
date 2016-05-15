% function [y_pred,beta,rho_hat] = ER_SecondMoment(Z, Ey, Ey2)
% Calculate the second moment estimation of y
% Input:
%   Z = data matrix
%   Ey = E[y] (y being the response variable)
%   Ey2 = E[y^2]
%   
function [y_pred,beta] = ER_QuadProg(Z, Ey, Ey2)


    [m n] = size(Z);
    var_y = Ey2 - Ey.^2;

    b_hat = mean(Z,2)-Ey; 
    Z_new = Z-b_hat * ones(1,n); 
    
    T = 1/n * Z_new * (Z_new'); 
    
    lb = []; ub = [];
    % constraints w_i <= 1, w_i >= 0
    A = [[zeros(m,1) eye(m)]; [zeros(m,1), -eye(m)]];
    b = [ones(m,1) ; zeros(m,1)];
    % constraints w_i >= 0
%     A = [zeros(m,1), -eye(m)];
%     b = [zeros(m,1)];
    % constrains sum(w_i) = 1
    %Aeq = [0 ones(1,m)]; beq = 1;
    Aeq = []; beq = [];
    f = -[Ey Ey2*ones(1,m)]';
    H = [[1 Ey*ones(1,m)] ; [Ey*ones(m,1) T]];
    H = (H+H')/2; % make sure the hessian is symmetric to numerical accuracy (eps).
    x0 = []; %[0;ones(m,1)/m];    
    
    options = optimset('Algorithm', 'interior-point-convex', 'Display','None');
    [beta, fval, exitflag] = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
    w_0 = beta(1);
    w = beta(2:end);
%     Q = T-Ey^2 * ones(size(T)); 
%     %w = var_y * Q \ ones(m,1); 
%     w = var_y *inv(Q) * ones(m,1); 
%     
%     w_0 = Ey * (1 - sum(w));
    
    y_pred = w_0 + (Z_new)'*w;
%     beta = [w_0;w];
    
%     fprintf('Eigs(Q): '); eig(Q)
end