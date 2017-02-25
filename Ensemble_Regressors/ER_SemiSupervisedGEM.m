% function [y_pred, w] = ER_SemiSupervisedGEM(Ztrain, ytrain, Ztest,Ey,Ey2)
% Semi-supervised GEM, estimator of the form y = Ey + sum_i ( w_i (f_i-mu_i) ) where the sum of the weights equals 1
% and rho is estimated using the training data
function [y_pred, w] = ER_SemiSupervisedGEM(Ztrain, ytrain, Zunlabeled, Ztest,Ey, Ey2)
    [m n] = size(Ztest);
    C = cov([Ztrain Zunlabeled]');
    rho = mean(bsxfun(@times,bsxfun(@minus, Ztrain, mean(Ztrain,2)),ytrain),2);

    %% find weights using the new representation
%     Cinv = pinv(C,1e-5);
%     w = Cinv*(rho - ones(m,1)*(ones(1,m)*(Cinv*rho)-1)/sum(sum(Cinv)));
    
    %% find weights using Perrone & Cooper representation
    Cstar_u = C - repmat(rho, 1,m) - repmat(rho',m,1) + Ey2;
    misfit_sgem = Ztrain - repmat(ytrain,m,1);
    Cstar_l = (misfit_sgem * misfit_sgem') / length(ytrain);
    Cstar = (Cstar_l*length(ytrain) + Cstar_u*size(Zunlabeled,1)) / (length(ytrain) + size(Zunlabeled,1));
    %Cstar = Cstar_u;
    Ci = inv(Cstar);
    w = zeros(m,1);
    for i=1:m
        w(i) = sum(Ci(i,:)) / sum(sum(Ci));
    end;

    
    %% Calculate oracle predictions
    y_pred = Ey + (Ztest - repmat(mean(Ztest,2),1,n))'*w;
end