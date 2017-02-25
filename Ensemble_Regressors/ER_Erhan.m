function [ens, mse, rank] = ER_Erhan(X, method)
% X - nxm matrix of predictions, m predictors
% method - 'ls', 'lad', 'sd', 'wls', 'wlad', 'als'
% ls - least squares
% lad - least absolute deviation
% sd - spectral decomposition
% wls - weighted least squares
% wlad - weighted least absolute deviation
% als - alternating least squares
% Returns:
% ens - ensemble prediction
% mse - infered mean squared errors
% rank - rank of predictions

switch method
    case 'ls'
        mse = fitMseLS(X);
        rank = StandardCompetitionRankings(mse);
        invmse = 1./mse;
        w = invmse / sum(invmse);
        ens = X * w;
    case 'lad'
        mse = fitMseLAD(X);
        rank = StandardCompetitionRankings(mse);
        invmse = 1./mse;
        w = invmse / sum(invmse);
        ens = X * w;       
    case 'sd'
        [mse, V] = fitMseSD(X);
        rank = StandardCompetitionRankings(V);
        invmse = 1./mse;
        w = invmse / sum(invmse);
        ens = X * w;
    case 'wls'
        mse = fitMseWLS(X);
        rank = StandardCompetitionRankings(mse);
        invmse = 1./mse;
        w = invmse / sum(invmse);
        ens = X * w;
    case 'wlad'
        mse = fitMseWLAD(X);
        rank = StandardCompetitionRankings(mse);
        invmse = 1./mse;
        w = invmse / sum(invmse);
        ens = X * w;    
    case 'als'
        mse = fitMseALS(X);
        rank = StandardCompetitionRankings(mse);
        invmse = 1./mse;
        w = invmse / sum(invmse);
        ens = X * w;
    otherwise    
        disp(['Invalid method ' method]);

end

end

function mse = fitMseALS(X)
    % X is nxm with m predictors and n variables
    
    [n, m] = size(X);
    
    % MSE matrix
    M = (1 / n) * (squareform(pdist(X')).^2);
    mse = fitMseLS(X);
    M = M + diag(2*mse);
    EM = exp(M);

    [x,y,z,err] = als(EM, 1, 1000, 0.1);
    mse = log(z);

end

function [mse, V] = fitMseSD(X)
% X is nxm with m predictors and n variables
    
    [n, m] = size(X);
    
    mse0 = fitMseLAD(X);
    rmse0 = sqrt(mse0);
    ioutlier = rmse0 > quantile(rmse0, 0.75) + 1.5*iqr(rmse0);
    M = (1 / n) * (squareform(pdist(X(:,~ioutlier)')).^2);  
    M = M + diag(2*mse0(~ioutlier));
    EM = exp(M);
    [V,Lambda] = eigs(EM,1,'lm');
    V(V==0) = eps;  % Fix when VV has zero elements
    V = abs(V);
    mse1 = log(sqrt(Lambda)*V);
    mse1 = fixMseLS(M, mse1); % make sure mse > 0
    mse = zeros(m,1);
    mse(ioutlier) = mse0(ioutlier);
    mse(~ioutlier) = mse1;
end

function out = fixMseLS(M,mse)
    % Use least squares to fit a bias term such that mse > 0
    m = length(M);
    n = 0.5*m*(m-1);
    d = zeros(n,1);
    C = zeros(n,2);
    b = zeros(m,1);
    A = -1*ones(m,2);
    A(:,1) = -1*mse;
    maxM = max(max(M));
    
    count = 0;
    for i = 1:(m-1)
        for j = (i+1):m
            count = count + 1;   
            w = sqrt(exp(M(i,j)-maxM));
            d(count) = w*M(i,j);
            C(count,1) = w * (mse(i) + mse(j));
            C(count,2) = w * 2;
        end
    end

    options = optimoptions(@lsqlin,'Display','off', 'Algorithm','active-set');
    x = lsqlin(C,d,A,b,[],[],[],[],[],options);
    out = mse*x(1) + x(2);
end

function mse = fitMseLAD(X)
% X is nxm with m predictors and n variables
    
    [n, m] = size(X);
    
    % MSE matrix
    M = (1 / n) * (squareform(pdist(X')).^2);  
    [f,A,b,lb,ub] = buildLadEq(M);

    options = optimoptions('linprog','Diagnostics','off','Display','off','Algorithm','dual-simplex'); % Try also interior-point
    prm = linprog(f,A,b,[],[],lb,ub,[],options);
    mse = prm(1:m);
end

function [f,A,b,lb,ub] = buildLadEq(M)
    
    m = length(M);
    n = 0.5*m*(m-1);
    f = [zeros(m,1); ones(n,1)];
        
    count = 0;
    X = sparse(n,m);  
    y = zeros(n,1);
    for i = 1:(m-1)
        for j = (i+1):m
            count = count + 1;
            X(count,i) = 1;
            X(count,j) = 1;
            y(count) = M(i,j);
        end
    end  
    
    A = [-X -1*speye(n); X -1*speye(n)];
    b = [-y; y];
    lb = [zeros(m,1); -inf*ones(n,1)];
    ub = [];
    f = sparse(f);
    b = sparse(b);
    lb = sparse(lb);
end

function mse = fitMseLS(X)
% X is nxm with m predictors and n variables
    
    [n, m] = size(X);
    
    % MSE matrix
    M = (1 / n) * (squareform(pdist(X')).^2);
    [C, d, lb] = buildLsEq(M);
    options = optimoptions(@lsqlin,'Display','off','Algorithm','interior-point');
    prm = lsqlin(C,d,[],[],[],[],lb,[],[],options);
    mse = prm(1:m);
end

function [C, d, lb] = buildLsEq(M)
    % Build the eq terms for lsqlin
    m = length(M);
    n = 0.5*m*(m-1);
    lb = zeros(m,1);
    d = zeros(0.5*m*(m-1), 1);
    I = [1:n 1:n]';
    J = zeros(2*n, 1);
    
    count = 0;
    for i = 1:(m-1)
        for j = (i+1):m
            count = count + 1;      
            d(count) = M(i,j);
            J(count) = i;
            J(count+n) = j;
        end
    end
    
    C = sparse(I,J,ones(2*n,1),n,m);
end

function mse = fitMseWLAD(X)
% X is nxm with m predictors and n variables
    
    [n, m] = size(X);
    % MSE matrix
    M = (1 / n) * (squareform(pdist(X')).^2);
    [f,A,b,lb,ub] = buildWLadEq(M);

    options = optimoptions('linprog','Diagnostics','off','Display','off','Algorithm','interior-point'); 
    prm = linprog(f,A,b,[],[],lb,ub,[],options);
    mse = prm(1:m);
end

function [f,A,b,lb,ub] = buildWLadEq(M)
    
    m = length(M);
    n = 0.5*m*(m-1);
    f = [zeros(m,1); ones(n,1)];
        
    count = 0;
    X = sparse(n,m);  
    y = zeros(n,1);
    for i = 1:(m-1)
        for j = (i+1):m
            count = count + 1;
            w = exp(M(i,j));
            X(count,i) = w;
            X(count,j) = w;
            y(count) = w*M(i,j);
        end
    end  
    
    A = [-X -1*speye(n); X -1*speye(n)];
    b = [-y; y];
    lb = [zeros(m,1); -inf*ones(n,1)];
    ub = [];
    f = sparse(f);
    b = sparse(b);
    lb = sparse(lb);
end

function mse = fitMseWLS(X)
% X is nxm with m predictors and n variables
    
    [n, m] = size(X);

    % MSE matrix
    M = (1 / n) * (squareform(pdist(X')).^2);
    [C, d, lb] = buildWLsEq(M);
    options = optimoptions(@lsqlin,'Display','off','Algorithm','interior-point');
    prm = lsqlin(C,d,[],[],[],[],lb,[],[],options);
    mse = prm(1:m);
end

function [C, d, lb] = buildWLsEq(M)
    % Build the eq terms for lsqlin
    m = length(M);
    n = 0.5*m*(m-1);
    lb = zeros(m,1);
    d = zeros(n, 1);
    v = zeros(2*n, 1);
    I = [1:n 1:n]';
    J = zeros(2*n, 1);
    
    count = 0;
    for i = 1:(m-1)
        for j = (i+1):m
            count = count + 1;    
            w = exp(M(i,j));
            d(count) = w*M(i,j);
            J(count) = i;
            J(count+n) = j;
            v(count) = w;
            v(count+n) = w;
        end
    end
    
    C = sparse(I,J,v,n,m);
end

function y = StandardCompetitionRankings(x)
    % Prepare data
    ctrl = isvector(x) & isnumeric(x);
    if ctrl
      x = x(:);
      x = x(~isnan(x) & ~isinf(x));
    else
      error('x is not a vector of numbers.')
    end
    % Find the Frequency Distribution
    [y, ind] = sort(x);
    FreqTab(:, 1) = y([find(diff(y)); end]);
    N1 = length(x);
    N2 = length(FreqTab(:, 1));
    if N1 == N2
      y(ind) = 1:N1;
      return
    end
    FreqTab(:, 2) = histc(y, FreqTab(:, 1));
    % Find the rankings
    y = (1:N1)';
    k = 1;
    for i = 1:N2
      if FreqTab(i, 2) > 1
        y(k:(k + FreqTab(i, 2) - 1)) = k;
      end
      k = k + FreqTab(i, 2);
    end
    y = sortrows([y, ind], 2);
    y(:, 2) = [];
end

function [x,y,z,err] = als(X, minval, iter, tol)
% Approximates positive symetric matrix X = z*z' where y > 1

    [n, m] = size(X);

    if n ~= m
        error('Matrix X should be square.');
    end

    if any(X < 0)
        error('Matrix X should have strictly positive elements.');
    end

    if ~issymmetric(X)
        error('Matrix X should be symetric.');
    end

    x = 1 + 1e-6*rand(n,1); 

    for k = 1:iter
        [C, d, lb] = buildEq1(X, x, minval);
        options = optimoptions(@lsqlin,'Display','off','Algorithm','interior-point');
        prm = lsqlin(C,d,[],[],[],[],lb,[],[],options);
        y = prm(1:m);

        [C, d, lb] = buildEq2(X, y, minval);
        options = optimoptions(@lsqlin,'Display','off','Algorithm','interior-point');
        prm = lsqlin(C,d,[],[],[],[],lb,[],[],options);
        oldx = x;
        x = prm(1:m);

        err = norm(x - oldx);
        if err < tol
            disp('Optimization finished: tolerance exceeded.');
            break;
        end
    end
    
    if k == iter
        disp('Optimization finished: Max iteration reached.');
    end
    
    z = sqrt(x.*y);
    err = norm(X - z*z');

end


function [C, d, lb] = buildEq1(M, a, minval)
    % Build the eq terms for lsqlin
    m = length(M);
    n = m^2;
    d = zeros(n, 1);
    I = (1:n)';
    J = zeros(n, 1);
    c = zeros(n, 1);
    lb = minval*ones(m,1);
    
    count = 0;
    for i = 1:m
        for j = 1:m
            count = count + 1;      
            d(count) = M(i,j);
            J(count) = j;
            c(count) = a(i);
        end
    end
    
    C = sparse(I,J,c,n,m);
end

function [C, d, lb] = buildEq2(M, b, minval)
    % Build the eq terms for lsqlin
    m = length(M);
    n = m^2;
    d = zeros(n, 1);
    I = (1:n)';
    J = zeros(n, 1);
    c = zeros(n, 1);
    lb = minval*ones(m,1);
    
    count = 0;
    for i = 1:m
        for j = 1:m
            count = count + 1;      
            d(count) = M(i,j);
            J(count) = i;
            c(count) = b(j);
        end
    end
    
    C = sparse(I,J,c,n,m);
end


