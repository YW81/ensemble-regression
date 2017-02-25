% function [y_lrm, w_lrm, y_oracle_rho, w_oracle_rho, Cstar] = ER_LowRankMisfitCVX( Z, Ey, Ey2, y_true, alpha )
function [y_lrm, w_lrm, y_oracle_rho, w_oracle_rho, Cstar] = ER_LowRankMisfitCVX( Z, Ey, Ey2, y_true, alpha )
    addpath('./HelperFunctions/') % for getApproxRank
    addpath('../HelperFunctions/') % for getApproxRank
    [m,n] = size(Z);
    var_y = Ey2 - Ey^2;
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z');
    
%     %% Solve A\rho = \gamma, where A is m-choose-2 combinations of regressors (only off-diagonal elements are chosen), 
%     %  and \gamma is a vector with the corresponding element in C + Ey2
%     subs = nchoosek(1:m,2);
%     idxs = sub2ind(size(C),subs(:,1),subs(:,2));
%     gamma = C(idxs) + Ey2; %var_y;
%     A = zeros(size(subs,1), m);
%     for i=1:size(subs,1)
%         A(i,subs(i,1)) = 1;
%         A(i,subs(i,2)) = 1;
%     end;
%     % Solve for rho: A*rho = gamma
%     rho = A\gamma;

    if ~exist('alpha','var')
        % alpha = 0.1/var_y;
        alpha = 1/(sqrt(m*Ey2^3));
        alpha_was_given = false;
    else
        alpha_was_given = true;
    end;

    %% Solve minimization problem Min||A-B||_F subject to B positive semi-definite
    run_optimization = true;
    while run_optimization ~= false
        cvx_begin sdp
            variable Cstar(m,m) symmetric;
            variable rho(m)
            expression Gamma(m,m);
            Cstar == semidefinite(m);

            for i=1:m; for j=1:m; Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j); end; end;

            minimize(norm(Gamma-Cstar,'fro') + alpha*norm_nuc(Cstar))
            %subject to
            %    Cstar+Cstar' >= 0
        cvx_end
        
        % Finish optimizating is the optimization problem was solved, or if alpha was given as a
        % parameter to the function. Otherwise, if alpha was calculated here and the optimization
        % failed, keep reducing this constraint until the problem becomes feasible.
        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            run_optimization = false;
        elseif alpha_was_given || (alpha < eps) % alpha was given, or it's already very small
            run_optimization = false;
            fprintf(2,'Optimization Failed\n');
        else  % alpha wasn't given, but the chosen alpha created an infeasible problem.
            alpha = alpha / 10;
            fprintf(2,'Problem not feasible with alpha=%.2g, trying new alpha=%.2g\n', alpha*10, alpha);
        end;
    end;

    % Calculate Low-Rank Misfit Covariance Estimation Weights and Predictions
    lambda = ( ones(1,m)*(C\rho) - 1 ) / (ones(1,m)*(C\ones(m,1)));
    w_lrm = C\(rho - lambda*ones(m,1));
    y_lrm = Ey + Z0' * w_lrm;    
    
    %% Oracle rho
    % Calculate Oracle Rho Weights and Predictions. Note that for an ill-conditioned C, having an
    % oracle rho is still not great. Consider diagonally loading covariance C
    rho_true = Z0*y_true' / n;
warnmode = warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:singularMatrix');
    lambda_oracle_rho = ( (ones(1,m)*(C\rho_true) ) - 1) / (ones(1,m)*(C\ones(m,1)));
    w_oracle_rho = C \ (rho_true - lambda_oracle_rho * ones(m,1));
warning(warnmode);    
    y_oracle_rho = Ey + Z0' * w_oracle_rho;
    
    [rho rho_true]
    if (alpha > eps)
        fprintf('Approx. rank(C*) = %d\n',getApproxRank(Cstar));
    end;
end

