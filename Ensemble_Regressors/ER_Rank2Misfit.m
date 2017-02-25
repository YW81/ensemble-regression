% function [y_lrm, w_lrm, Cstar_offdiag, rho] = ER_Rank1Misfit( Z, Ey, Ey2 )
% Assume independent misfit errors (diagonal C*), recover rho, find optimal weights.
function [y_lrm, w_lrm, Cstar_offdiag, rho, rho_true] = ER_Rank2Misfit( Z, Ey, Ey2 )
    [m,n] = size(Z);
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z');
    
    %% Test (the following 2 lines generate C,diagonal C*, and rho that allow rho to be recovered):
    fprintf(2,'====== TESTING ======\n');
    v_true = rand(2*m,1); rho_true = rand(m,1); C =  diag(1:m) + v_true(1:m)*v_true(1:m)'+ v_true(m+1:2*m)*v_true(m+1:2*m)' + repmat(rho_true,1,m) + repmat(rho_true',m,1) - Ey2
    % clear e; for m=2:40; for i=1:30; [~,~,~,rho,rho_true]=ER_Rank2Misfit(rand(m,1000), 0,1); e(i,m) = pdist([rho' ; rho_true'],'cosine'); end; end; figure; boxplot(e); xlabel('m'); ylabel('cos dist (rho, rho true)');
    
    %% Solve linear equation  C_ij + Ey2 = rho_i + rho_j 
    % using the matrix A which selects the off rho that correspond to the off-diagonal of C
    % such that A\rho = C + Ey2. A is m-choose-2 combinations of regressors (off-diagonal elements only)
    subs = nchoosek(1:m,2);
    idxs = sub2ind(size(C),subs(:,1),subs(:,2));
    A = zeros(size(subs,1), m);
    for i=1:size(subs,1)
        A(i,subs(i,1)) = 1;
        A(i,subs(i,2)) = 1;
    end;

    Cshifted = C(idxs) + Ey2; %var_y;        
    
    %% Solve for rho,v: v*v'+A*rho = C+Ey2
    
    % define cost function and initial guess
    function [fval] = offdiag_frobenius_norm (rho,v1,v2) 
        P=v1*v1' + v2*v2'; % rank-1 perturbation
        fval = norm(P(idxs) + A*rho - Cshifted);
    end
    cost = @(x) offdiag_frobenius_norm(x(1:m),x(m+1:2*m),x(2*m+1:end));
    x0 = [A\Cshifted ; rand(2*m,1)]; % initial guess is independent misfit error
    
    % run solver
    options = optimoptions('fsolve','MaxFunEvals',1e6,'MaxIter',1e6,'TolX',eps);
    [x,fval]=fsolve(cost, x0,options);
%     options2 = optimoptions('fminunc','MaxFunEvals',1e6,'MaxIter',1e6,'TolX',eps);
%     [x,fval] = fminunc(cost,x0,options2);
    rho = x(1:m); v1 = x(m+1:2*m); v2 = x(2*m+1:3*m); 
    P = v1*v1' + v2*v2'; Cstar_offdiag = zeros(m); 
    Cstar_offdiag(idxs) = P(idxs); Cstar_offdiag = Cstar_offdiag + Cstar_offdiag';
    fval,

    % Calculate weights based on the assumption of independent misfit errors
    Cinv = pinv(C,1e-5);
    w_lrm = Cinv*(rho - ones(m,1)*(ones(1,m)*(Cinv*rho)-1)/sum(sum(Cinv)));
    y_lrm = Ey + Z0' * w_lrm;    
end

