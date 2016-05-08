% function [y_pred,beta,rho_hat] = ER_SecondMoment(Z, Ey, Ey2)
% Calculate the second moment estimation of y
% Input:
%   Z = data matrix
%   Ey = E[y] (y being the response variable)
%   Ey2 = E[y^2]
%   
function [y_pred,beta] = ER_Boaz(Z, Ey, Ey2)


    [m n] = size(Z);
    var_y = Ey2 - Ey.^2;

    b_hat = mean(Z,2)-Ey; 
    Z_new = Z-b_hat * ones(1,n); 
    
    T = 1/n * Z_new * (Z_new'); 
    Q = T-Ey^2 * ones(size(T)); 
    w = var_y * (Q \ ones(m,1)); 
    %w = var_y *inv(Q) * ones(m,1); 
    
    w_0 = Ey * (1 - sum(w));
    
    y_pred = w_0 + (Z_new)'*w;
    beta = [w_0;w];
    
    %fprintf('Eigs(Q): '); eig(Q)
end