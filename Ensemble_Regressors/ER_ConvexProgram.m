function [y_pred,beta,fval] = ER_ConvexProgram(Z, Ey, Ey2, lambda_mean, lambda_var, beta_0)
% function [y_pred,beta,fval] = ER_ConvexProgram(Z, Ey, Ey2, lambda_mean, lambda_var, beta_0)
% Convex program for optimizing the weights with penalties for deviation from mean and variance

    % Init params
    [m,n] = size(Z);
    var_y = Ey2 - Ey.^2;
    
    b_hat = mean(Z,2) - Ey; % approximate bias
    C = cov(Z');

    Zc = Z - repmat(b_hat,1,n);
    Z1c = [ones(1,n); Zc];
    K1 = Zc*Z1c';
    K2 = Z1c*Z1c';

    %% Convex Optimization for w/beta
    % constraints w_i >= 0
    A = [zeros(m,1), -eye(m)];
    b = zeros(m,1);
    
    options = optimoptions('fmincon', 'Display','None');%'iter-detailed');%,'MaxFunEvals',1e4);
    obj = @(beta) cvx_opt_for_w(beta,m,n,Ey,var_y, C, Z1c, K1, K2, lambda_mean, lambda_var);
    [beta,fval,exitflag,output,lambda,grad,hessian] = fmincon(obj, beta_0, A, b,[],[],[],[], [], options);

    y_pred = Z1c'*beta;
    
%     beta,
%     fprintf('======= NOW STARTING AT beta\n'); 
%     [beta_new,fval,exitflag,output,lambda,grad,hessian] = fmincon(obj, beta, A, b,[],[],[],[], [], options);
%     
%     fprintf('NEW:\n'); beta_new
end