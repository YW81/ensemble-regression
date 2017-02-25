% function [y_lrm, w_lrm] = ER_IndependentMisfits( Z, Ey, Ey2 )
% Assume independent misfit errors (diagonal C*), recover rho, find optimal weights.
function [g2_est] = estimate_g2( Z, Ey, Ey2 , loss)
    if ~exist('loss','var')  % loss can be 'l2','l1','huber'
        loss = 'l2';
    end;
    [m,n] = size(Z);
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z');
    
    %% Test (the following 2 lines generate C,diagonal C*, and rho that allow rho to be recovered):
    %m=5; Ey2 = 1;
    %v = 1+rand(m,1); rho_true = rand(m,1); C =  diag(v) + repmat(rho_true,1,m) + repmat(rho_true',m,1) - Ey2
    
    %% Solve linear equation  C_ij + Ey2 = rho_i + rho_j 
    % using the matrix A which selects the off diagonal elements of rho
    % such that A\rho = C + Ey2. A is m-choose-2 combinations of regressors (off-diagonal elements only)
    subs = nchoosek(1:m,2);
    idxs = sub2ind(size(C),subs(:,1),subs(:,2));

    XCov = (1 / n) * (squareform(pdist(Z)).^2); % XCov_{ij} = mean((Z(i,:) - Z(j,:)).^2) = E[(f_i - f_j)^2]
    XCov = XCov(idxs);
    A = zeros(size(subs,1), m);
    for i=1:size(subs,1)
        A(i,subs(i,1)) = 1;
        A(i,subs(i,2)) = 1;
    end;
    % Solve for rho: A*rho = C+Ey2
    if strcmp(loss,'huber')
        fval = @(rho) sum(huber(XCov - A*rho,1/(10*m))); % Huber-loss with delta = 1/10m
        b = fminunc(fval,zeros(m,1));
    elseif strcmp(loss,'l1')
        fval = @(rho) norm(XCov - A*rho,1);  % L1-loss
        b = fminunc(fval,zeros(m,1));
    else
        %fval = @(rho) norm(Cshifted - A*rho);  % L2-loss <==> Least-Squares
        %rho = A\Cshifted; % Least-Squares
        [b, resnorm, residuals] = lsqlin(A,XCov);
        R=zeros(m); R(idxs)=residuals./Ey2; R=R+R'; %imagesc(R); colormap(hot); colorbar;title('unconstrained residuals'); 
        fprintf('unconstraind norm(residuals) = %.2f',norm(R));
    end;

    g2_est = (trace(C) - sum(b))/m;
end

