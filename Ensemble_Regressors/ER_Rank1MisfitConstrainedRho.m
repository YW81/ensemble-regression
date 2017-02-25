% function [y_lrm, w_lrm, Cstar_offdiag, rho] = ER_Rank1Misfit( Z, Ey, Ey2 )
% Assume independent misfit errors (diagonal C*), recover rho, find optimal weights.
function [y_lrm, w_lrm, Cstar_offdiag, rho, v, rho_true, v_true, fval] = ER_Rank1MisfitConstrainedRho( Z, Ey, Ey2, deltastar )
    [m,n] = size(Z);
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z');
    
    var_y = Ey2 - Ey^2;
    [v_1,lambda_1] = eigs(C,1,'lm');
    t = sign(sum(v_1)) * sqrt((1-deltastar)*var_y / lambda_1);
    rho0 = t*lambda_1*v_1;
    
    %% Test (the following 2 lines generate C,diagonal C*, and rho that allow rho to be recovered):
%     fprintf(2,'====== TESTING ======\n');
%     v_true = 0.5*rand(m,1)-.25; rho_true = Ey2*(0.8+.1*rand(m,1)); C = diag(1:m) + v_true*v_true' + repmat(rho_true,1,m) + repmat(rho_true',m,1) - Ey2; 
%     % clear e; for m=2:40; for i=1:30; [~,~,~,rho,rho_true]=ER_Rank1Misfit(rand(m,1000), 0,1); e(i,m) = pdist([rho' ; rho_true'],'cosine'); end; end; figure; boxplot(e); xlabel('m'); ylabel('cos dist (rho, rho true)');
    
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
    
    % set inital value for the solver
    x0 = [rho0 ; zeros(m,1)]; % initial guess is independent misfit error
    %x0 = [Ey2*ones(m,1); 1*rand(m,1)];
    %x0 = zeros(2*m,1); % initial guess is independent misfit error
    
    %% Option 1 - fsolve (looks for a zero, which requires a perfect fit, should only exist in simulations)
    % define cost function and initial guess
%     function [fval] = offdiag_frobenius_norm (rho,v) 
%         P=v*v'; % rank-1 perturbation
%         fval = norm(P(idxs) + A*rho - Cshifted);
%     end
%     cost = @(x) offdiag_frobenius_norm(x(1:m),x(m+1:end));
%     options = optimoptions('fsolve','MaxFunEvals',1e6,'MaxIter',1e6,'TolX',eps);%1e-18);
%     [x,fval,exitflag]=fsolve(cost, x0,options); disp(exitflag);
    
    %% Option 2 - fminunc
%     options2 = optimoptions('fminunc','MaxFunEvals',1e6,'MaxIter',1e6,'TolX',eps);
%     [x,fval] = fminunc(cost,x0,options2);

    %% Option 3 - non-linear least-squares
    function [offdiag_residuals] = offdiag_elements(rho,v)
        P=v*v'; % rank-1 perturbation
        penalty_rho_amplitude = abs(norm(rho) - norm(rho0))/Ey2;
        penalty_rho_direction = abs(norm(rho/norm(rho) - rho0/norm(rho0))); % TRY A COST FUNCTION WHICH IS LSQ + L1_NORM(NORMALIZED RHO - RHO_0)
        penalty_residuals = ((P(idxs) + A*rho - Cshifted)/Ey2).^2;
        offdiag_residuals = sqrt(penalty_residuals + penalty_rho_direction + 0*penalty_rho_amplitude); 
    end
    options3 = optimoptions('lsqnonlin','MaxFunEvals',1e5,'MaxIter',1e5);%,'TolX',min(abs(Ey),Ey2)*1e-10,'TolFun',min(abs(Ey),Ey2)*1e-10);
    cost3 = @(x) offdiag_elements(x(1:m),x(m+1:end));
    [x,fval,residuals] = lsqnonlin(cost3, x0,[],[],options3);
    R=zeros(m); R(idxs)=residuals./Ey2; R=R+R'; imagesc(R); colormap(hot); colorbar;title('Rank-1 residuals'); 
    %fprintf('rank-1 norm(residuals) = %.2f\n',norm(R));
    
    rho = x(1:m); v = x(m+1:2*m); 
    P = v*v'; Cstar_offdiag = zeros(m); 
    Cstar_offdiag(idxs) = P(idxs); Cstar_offdiag = Cstar_offdiag + Cstar_offdiag'; % can use residuals instead
    fprintf('rank-1 fval = %.2f, Ey2 = %g\n',fval, Ey2);
    fprintf('---\nrank-1 mean residuals = %g\n',mean(residuals));
    fprintf('rank-1 mean penalty angle = %g\n',mean(abs(norm(rho/norm(rho) - rho0/norm(rho0)))));
    fprintf('rank-1 penalty amp = %g\n',abs(norm(rho) - norm(rho0)));
    fprintf('---\nrank-1 CosDist(rho,rho0) = %g\n',1-dot(rho/norm(rho),rho0/norm(rho0)));
    fprintf('rank-1 norm(rho)/norm(rho0) = %g\n',norm(rho)/norm(rho0));

    % Calculate weights based on the assumption of independent misfit errors
    Cinv = pinv(C,1e-5);
    w_lrm = Cinv*(rho - ones(m,1)*(ones(1,m)*(Cinv*rho)-1)/sum(sum(Cinv)));
    %w_lrm = Cinv*rho;
    y_lrm = Ey + Z0' * w_lrm;    
end

