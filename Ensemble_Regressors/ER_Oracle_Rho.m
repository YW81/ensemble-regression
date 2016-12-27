% function [y_pred, w] = ER_Oracle_Rho(y_true, Z)
% Oracle of the form y = Ey + sum_i ( w_i (f_i-mu_i) ) where the sum of the weights equals 1
function [y_pred, w] = ER_Oracle_Rho(y_true, Z)
    [m n] = size(Z);
    Ey = mean(y_true);
    C = cov(Z');
    rho = mean(bsxfun(@times,bsxfun(@minus, Z, mean(Z,2)),y_true),2);
    
    %w = C\(rho - ones(m,1)*(ones(1,m)*(C\rho)-1)/sum(sum(pinv(C))));
    Cinv = pinv(C,1e-5);
    w = Cinv*(rho - ones(m,1)*(ones(1,m)*(Cinv*rho)-1)/sum(sum(Cinv)));
    
    %% Calculate oracle predictions
    y_pred = Ey + (Z - repmat(mean(Z,2),1,n))'*w;
end