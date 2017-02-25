% function [y_pred, reg_idxs] = MV_Rank1Misfit( Z, Ey, Ey2, threshold_m, threshold_n )
% Assume rank-1 misfit errors (diagonal C*), recover rho, find optimal weights.
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan. If two predictors have less than threshold_n elements in common, we assume Cij cannot be
% calculated accurately and ignore that one of them completely. We ignore the one with a smaller
% total number of predictions.
% The function returns reg_idxs which is a vector containing the indexes of the regressors that were
% used to make the predictions (had enough common elements with the other predictors in the ensemble
% to accurately estimate Cij).
function [y_pred, reg_idxs] = MV_Rank1Misfit( Z, Ey, Ey2, threshold_m, threshold_n )
    [C, Z, reg_idxs] = calc_C_with_missing_values(Z, threshold_n); % note this overwrites Z
    if length(reg_idxs) < 5
        throw(MException('MV_Rank1Misfit:tooSparse', ...
                         'Not enough common predictions to calculate the covariance terms. Try smaller threshold_n.'))
    end;
    [full_m, n] = size(Z);
    y_pred = nan*ones(n,1);
    Z0 = Z - nanmean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    
    %% Solve linear equation  C_ij + Ey2 = rho_i + rho_j only once
    % using the matrix A which selects the off rho that correspond to the off-diagonal of C
    % such that A\rho = C + Ey2. A is m-choose-2 combinations of regressors (off-diagonal elements only)
    subs = nchoosek(1:full_m,2);
    idx_pairs = sub2ind(size(C),subs(:,1),subs(:,2));
    A = zeros(size(subs,1), full_m);
    for i=1:size(subs,1)
        A(i,subs(i,1)) = 1;
        A(i,subs(i,2)) = 1;
    end;

    Cshifted = C(idx_pairs) + Ey2; %var_y;        
    
    %% Solve for rho,v: v*v'+A*rho = C+Ey2
    
    % define cost function and initial guess
    function [fval] = offdiag_frobenius_norm (rho,v) 
        P=v*v'; % rank-1 perturbation
        fval = norm(P(idx_pairs) + A*rho - Cshifted);
    end
    cost = @(x) offdiag_frobenius_norm(x(1:full_m),x(full_m+1:end));
    %x0 = [A\Cshifted ; rand(full_m,1)]; % initial guess is independent misfit error
    x0 = [Ey2*ones(full_m,1); 1*rand(full_m,1)];
    %x0 = zeros(2*full_m,1); % initial guess is independent misfit error
    
    % run solver
    options = optimoptions('fsolve','MaxFunEvals',1e6,'MaxIter',1e6,'TolX',eps);%1e-18);
    [x,fval]=fsolve(cost, x0,options);
%     options2 = optimoptions('fminunc','MaxFunEvals',1e6,'MaxIter',1e6,'TolX',eps);
%     [x,fval] = fminunc(cost,x0,options2);
    rho = x(1:full_m); v = x(full_m+1:2*full_m); 
    P = v*v'; Cstar_offdiag = zeros(full_m); 
    Cstar_offdiag(idx_pairs) = P(idx_pairs); Cstar_offdiag = Cstar_offdiag + Cstar_offdiag';
    fval,

    for i=1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Z(:,i)));   % indices of experts that provided prediction on stock i
        m = numel(idxs);

        if m < threshold_m
            continue; 
        end
        
        % Calculate weights based on the assumption of independent misfit errors
        Cinv = pinv(C(idxs,idxs),1e-5);
        w = Cinv*(rho(idxs) - ones(m,1)*(ones(1,m)*(Cinv*rho(idxs))-1)/sum(sum(Cinv)));
        y_pred(i) = Ey + Z0(idxs,i)' * w;
    end;
end

