clear all;
addpath ../Ensemble_Regressors/
addpath ../HelperFunctions/

rng(0);

tic;

%% Create data
Ey = 0; Ey2 = 1;
m = 5;
n=1000;

Z = rand(m,n);
C = cov(Z');
v = rand(m,1);
Cstar_true = v*v'; % Cstar_true is a rank-1 matrix

% Find rho by solving A*rho = gamma, where A is m-choose-2 combinations of regressors, 
%  and gamma is a vector with the corresponding element in C + Ey2 - C*
subs = nchoosek(1:m,2);
idxs = sub2ind(size(C),subs(:,1),subs(:,2));
gamma = C(idxs) + Ey2 - Cstar_true(idxs); %var_y;
A = zeros(size(subs,1), m);
for i=1:size(subs,1)
    A(i,subs(i,1)) = 1;
    A(i,subs(i,2)) = 1;
end;
% Solve for rho: A*rho = gamma
rho_true = A\gamma;

% rho_true = rand(m,1);
% % Construct matrix C
% for i=1:m;
%     for j=1:m;
%         C(i,j) = Cstar_true(i,j) + rho_true(i) + rho_true(j) - Ey2;
%     end;
% end;

%% Find Rank-1 approximation
WANTED_RANK = 1;
%f = @(x) abs(getApproxCstarRank(C,Ey2,x)-WANTED_RANK); % our error measure is 0 when rank C* = WANTED_RANK, > 0 otherwise.
f = @(x) LRMCostFunction(C,Ey2,x,WANTED_RANK); % our error measure is 0 when rank C* = WANTED_RANK, > 0 otherwise.
tol = 1e-15; lb = 0; ub = 1; % [0,1]
[alpha, fval, exitflag] = patternsearch(f,0,[],[],[],[],lb,ub,[], ...
                                        psoptimset('TolX',tol, 'TolMesh',tol,'TolFun',tol, ...
                                                    'PlotFcns',{@psplotbestf,@psplotbestx}));
if (exitflag ~= 1) %|| (fval ~= 0)
    error('Could not find rank-1 C* for alpha in the range [0,1]'); % raise exception
end;

[Cstar, rho] = FindLowRankCstarApprox(C,Ey2,alpha);

[rho_true,rho]
fprintf('FRO: %g\n',(norm(Cstar - Cstar_true,'fro')));
fprintf('NUC: %g\n',sum(svd(Cstar))); % nuclear norm


toc;
