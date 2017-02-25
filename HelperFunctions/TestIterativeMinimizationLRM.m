clear all;clc;

rng(0);

tic;

%% Create data
Ey = 0; Ey2 = 1;
m = 100;%20
n=10000;

Z = rand(m,n); Z = bsxfun(@minus,Z,mean(Z,2));
C = cov(Z');
v_true = rand(m,1);
Cstar_true = v_true*v_true'; % Cstar_true is a rank-1 matrix

% % Find rho by solving A*rho = gamma, where A is m-choose-2 combinations of regressors, 
% %  and gamma is a vector with the corresponding element in C + Ey2 - C*
% subs = nchoosek(1:m,2);
% idxs = sub2ind(size(C),subs(:,1),subs(:,2));
% gamma = C(idxs) + Ey2 - Cstar_true(idxs); %var_y;
% A = zeros(size(subs,1), m);
% for i=1:size(subs,1)
%     A(i,subs(i,1)) = 1;
%     A(i,subs(i,2)) = 1;
% end;
% % Solve for rho: A*rho = gamma
% rho_true = A\gamma;

rho_true = rand(m,1);
% Construct matrix C
for i=1:m;
    for j=1:m;
        C(i,j) = Cstar_true(i,j) + rho_true(i) + rho_true(j) - Ey2;
    end;
end;

%% Just another test
[v,e]=eigs(Cstar_true,1);
norm((C + Ey2*ones(m) - e*eye(m))\(rho_true*ones(1,m) + ones(m,1)*rho_true') * v  - v)
[a b]=fminsearch(@(rho) test_cost_func(C+.01*randn(m),Ey2,rho), Ey2*ones(m,1)),[rho_true a]

%% Test Splitting
alpha = 1;
cvx_begin
    variable Cstar(m,m) symmetric;
    variable rho(m)
    variable u(m,1)
    variable w(m,1)
    expression Gamma(m,m);

    for i=1:m; for j=1:m; Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j); end; end;

    minimize(norm(Gamma-u*w','fro'))% + alpha*norm(u - w))
cvx_end

% If the problem was not solved return nan;
if ~(strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved'))
    Cstar = nan;
    rho = nan;
else
    Cstar = u*w';
end;


%% No Iterations - Current ER_LowRankMisfitCVX implementation
alpha = 1;
cvx_begin sdp
    variable Cstar(m,m) symmetric;
    variable rho(m)
    expression Gamma(m,m);
    Cstar == semidefinite(m);

    for i=1:m; for j=1:m; Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j); end; end;

    minimize(norm(Gamma-Cstar,'fro') + alpha*norm_nuc(Cstar))
    subject to
        Cstar+Cstar' >= 0
cvx_end

% If the problem was not solved return nan;
if ~(strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved'))
    Cstar = nan;
    rho = nan;
end;

%% Find Rank-1 approximation
WANTED_RANK = 1;

%% Iterations - Test min(p,u) sum|| C_ij - rho_i - rho_j - u_i u_j ||^2
% Init
rho = Ey2 * ones(m,1); % Init rho = Ey2 (perfect predictions)
u = zeros(m,1); % Init u

for iter=1:10
    % Find rho given u
    cvx_begin % sdp 
        variable rho(m,1)
        expression Gamma(m,m);
        for i=1:m; 
            for j=1:m; 
                Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j) - u(i)*u(j); 
            end; 
        end;
        
        minimize(norm(Gamma,'fro'))
    cvx_end
    
    % Find u given rho
    cvx_begin 
        variable u(m,1);
        expression Gamma(m,m);
        for i=1:m; 
            for j=1:m; 
                Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j); 
            end; 
        end;
        
        minimize(norm(Gamma - u*u','fro'))
    cvx_end
    
    % Print results
    [rho_true,rho]
    Cstar_true, Cstar,
    fprintf(2,'MAX: %g\n',max(Cstar_true(:)));
    fprintf(2,'FRO: %g\n',(norm(Cstar - Cstar_true,'fro')));
    fprintf(2,'NUC: %g\n',sum(svd(Cstar))); % nuclear norm
end;

%% Iterations - Splitting: Test min(p,u,z) sum|| C_ij - rho_i - rho_j - u_i z_j ||^2 + lambda ||u - z||^2
% Init
rho = Ey2 * ones(m,1); % Init rho = Ey2 (perfect predictions)
v = zeros(m,1); % Init v
lambda = 1;

%for iter=1:10
    % Find rho given u,z
    cvx_begin % sdp 
        variable rho(m,1)
        expression Gamma(m,m);
        for i=1:m; 
            for j=1:m; 
                Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j) - v(i)*v(j); 
            end; 
        end;
        
        minimize(norm(Gamma,'fro'))
    cvx_end
    
    % Find u given rho,z
    Gamma = zeros(m);
    for i=1:m
        for j=1:m
            Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j); 
        end; 
    end;
    
    % analytical solution of min_u,e (A - uu')'(A - uu') = min ||A-uu'||^2
    [eigvec,eigval] = eigs(Gamma,1,'lm');
    u = eigvec * sqrt(eigval);
    z = u;
    
%     cvx_begin sdp
%         variable u(m,1);
%         minimize(norm(Gamma - u*z','fro') +  lambda*norm(u - z))
%         subject to
%             u*z' + z*u' >= 0
%     cvx_end
% 
%     % Find z given rho,u
%     cvx_begin sdp
%         variable z(m,1);
%         minimize(norm(Gamma - u*z','fro') +  lambda*norm(u - z))
%         subject to
%             u*z' + z*u' >= 0
%     cvx_end
     Cstar = u*z';
    
    % Print results
    [rho_true,rho]
    Cstar_true, Cstar,
    fprintf(2,'MAX: %g\n',max(Cstar_true(:)));
    fprintf(2,'FRO: %g\n',(norm(Cstar - Cstar_true,'fro')));
    fprintf(2,'NUC: %g\n',sum(svd(Cstar))); % nuclear norm
    [ v_true u z ]
%end;

toc;

%%
epsilon = 0; v = zeros(m,1); v(1) =1; Gamma = zeros(m);
%%
rho=(m*eye(m)+ones(m)) \ (C + Ey2 - epsilon*v*v')*ones(m,1)
for i=1:m; for j=1:m; Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j); end; end;

% find the eigenvector that gets the minimum score function
[V,D] = eig(Gamma);
S = Inf;
for i=1:m
    v_cur = V(:,i); 
    epsilon_cur = D(i,i);
    S_cur = norm(Gamma-epsilon_cur*v_cur*v_cur','fro');
    if S_cur < S
        S = S_cur;
        v = v_cur;
        epsilon = epsilon_cur;
        best_eig_idx = i;
    end;
end;


Cstar = epsilon*v*v';
S = norm(Gamma-epsilon*v*v','fro')

    [rho_true,rho]
    Cstar_true, Cstar,
    fprintf(2,'MAX: %g\n',max(Cstar_true(:)));
    fprintf(2,'FRO: %g\n',(norm(Cstar - Cstar_true,'fro')));
    fprintf(2,'NUC: %g\n',sum(svd(Cstar))); % nuclear norm
