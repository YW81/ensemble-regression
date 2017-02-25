% function [y_pred, w, t] = ER_SpectralApproach(Z, Ey)
function [y_pred, w, t] = ER_PCRtest_k_eigs(Z, Ey, Ey2, deltastar)

    % basic variables
    [m,n] = size(Z);    
    var_y = Ey2 - Ey^2;
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z'); % cov(Z') == cov(Zc') == cov(Z0')...

    % Get leading eigenvector and eigenvalue
    [v_1,lambda_1] = eigs(C,1,'lm');
    
    t_sign = sign(sum(v_1));
    t_old = t_sign * sqrt(var_y / lambda_1);
    
    [V,D] = eig(C);
    V = fliplr(V); D = flip(diag(D)); % sort larger eigval first, make D into a vector
%    % Assume rho =1
    k = find(cumsum(D) ./ sum(D) > .95,1); % find first eigval which brings the cumulative sum of the eigvals over 95%
%     A = repmat(D(1:k)',m,1) .* V(:,1:k);
%     
%     t = zeros(m,1);
%     t(1:k) = A \ (var_y * ones(m,1));
% 
%     fprintf(2,'Took %d PCs, average rho_i %.2f\n', k, mean(A*t(1:k) ./ var_y));

    % min MSE ==> max_t t'(V'DV)t 
    % not precise, this assumes that t=t*, the true value of t for 
    [t, fval, exitflag] = quadprog(V'*diag(D)*V,zeros(m,1),[],[],C*V,var_y*ones(m,1)); % minimize MSE with the constraint rho=Cw*=1*var_y.
    if exitflag == -2
        t = zeros(m,1);
    end;
    
    % Calculate predictions
    w_old = t_old * v_1;
    w = V(:,1:k) * t(1:k);
    y_pred = Ey + Z0' * w;
end