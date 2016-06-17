% function [y_pred, w, s] = ER_SemiSupervised_S_SelectionSpectralApproach(Ztrain, ytrain, Ztest)
function [y_pred, w, s] = ER_SemiSupervised_S_SelectionSpectralApproach(Ztrain, ytrain, Ztest)
    [m,n_train] = size(Ztrain);
    [m_test, n] = size(Ztest);
    
    assert(m == m_test, 'Ztrain and Ztest must have the same number of rows')

    % basic variables
    Ey = mean(ytrain);
    var_y = var(ytrain);
    Z0 = Ztest - mean(Ztest,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    Z0train = Ztrain - mean(Ztrain,2)*ones(1,n_train);
    C = cov(Ztest'); % cov(Z') == cov(Zc') == cov(Z0')...

    % Get leading eigenvector and eigenvalue
    [V,D] = eigs(C,5,'lm'); % columns of V are possible w's. Take 5 largest magnitude eigvals
    eigvals = diag(D);
    lambda_1 = eigvals(1); v_1 = V(:,1); 
    
    % Find the best value for s on the training set
    s_max = var_y / lambda_1;
    s_scan = linspace(0,max(.75,2*s_max),200); 
    scan_mse = zeros(size(s_scan));
    y_scan = zeros(n,numel(s_scan)); 
    for i=1:length(s_scan); 
        y_scan(:,i) = Ey + Z0' * v_1 * s_scan(i); 
        scan_mse(i) = mean((ytrain' - (Ey + Z0train' * v_1 * s_scan(i))).^2);
    end;
    s_best_mse_ind = find(scan_mse == min(scan_mse));
    
    % return results
    y_pred = y_scan(:,s_best_mse_ind);
    s = s_scan(s_best_mse_ind);
    w = v_1 * s_scan(i);
end