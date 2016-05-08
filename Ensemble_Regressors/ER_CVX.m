function [y_pred,beta,fval] = ER_CVX(Z, Ey, Ey2, lambda_mean, lambda_var)
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

    % get initial estimate
    %[~, beta_0] = ER_VarianceWeightingWithBiasCorrection(Z,Ey);    
    [~, beta_0] = ER_MeanWithBiasCorrection(Z,Ey);    

    %% Convex Optimization for w/beta
    % constraints w_i >= 0
    Const1 = (Ey2/n)*(C\ones(m,1));
    
    %% CVX
    cvx_begin % quiet
        variable w(m) nonnegative
        variable w0
        variable s
        expressions beta(m+1) yi(n) w_prime(m) beta_prime(m+1) y_prime(n)
        
        beta = [w0;w];
        yi = Z1c'*beta;
        %w_prime = var_y*(C\ones(m,1)) + (C\(K1*beta))/n - (C\(ones(m,1)*quad_form(beta,K2)))/n;
        w_prime = var_y*(C\ones(m,1)) + (C\(K1*beta))/n - Const1;
        beta_prime = [Ey*(1-sum(w_prime)); w_prime ];
        y_prime = Z1c'*beta_prime;
        
        %minimize( norm(yi - y_prime) + lambda_var*norm(var(Z1c'*beta) - var_y) )
        minimize( norm(yi - y_prime) + lambda_var*norm(var_y - var(Z1c'*beta)) )
%         minimize( norm(yi - y_prime) + lambda_var*(norm(s)) );
%         subject to
%             var(Z1c'*beta) + s == var_y;
        
    cvx_end
    
    beta = [w0;w];
    
    %% End CVX
    y_pred = Z1c'*beta;
    
    beta,
    fprintf('======= NOW STARTING AT beta\n'); 
    [beta_new,fval,exitflag,output,lambda,grad,hessian] = fmincon(obj, beta, A, b,[],[],[],[], [], options);
    
    fprintf('NEW:\n'); beta_new
end