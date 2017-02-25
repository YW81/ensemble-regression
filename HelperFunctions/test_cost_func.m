function [ fval ] = test_cost_func( C,Ey2,rho )
    m = size(C,1);
    Gamma = zeros(m);
    for i=1:m; 
        for j=1:m; 
            Gamma(i,j) = C(i,j)+Ey2-rho(i)-rho(j); 
        end; 
    end; 
    [v,e] = eigs(Gamma,1); 
    fval = norm((C + Ey2*ones(m) - e*eye(m))\(rho*ones(1,m) + ones(m,1)*rho') * v  - v);
    %fval = det(rho*ones(1,m) + ones(m,1)*rho' - C -Ey2*ones(m) + e*eye(m));
end

