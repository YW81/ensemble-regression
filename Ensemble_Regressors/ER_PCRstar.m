% function [y_pred,w,K] = ER_PCRstar(Ztrain, ytrain, Ztest)
function [y_pred,w,K] = ER_PCRstar(Ztrain, ytrain, Ztest)
    % Init params
    [m,n_train] = size(Ztrain);
    [m_test, n] = size(Ztest);
    ytrain = ytrain';
    
    assert(m == m_test, 'Ztrain and Ztest must have the same number of rows')
    
    %% PCR*
    A = Ztrain';
    C = cov(A);
    
    % with pcacov
    [Gamma, eigvals] = pcacov(C);

%     % with SVD
%     [U,D,V] = svd(C); % C PSD ==> U == V'
%     eigvals = diag(D);    
%     Gamma = U;
    
    %% PCR* with cutoff
    K = pcrstar_choose_cutoff(A, Gamma, ytrain);
    PC = A * Gamma(:,1:K); % = A * U;
    beta = (PC'*PC) \ (PC' * ytrain);
    alpha = Gamma(:,1:K)*beta;
    
%     %% TEST WITH ONLY 2 EIGS
%     K = pcrstar_choose_cutoff(A, Gamma, ytrain);
%     PC = A * Gamma(:,[1 K]); % = A * U;
%     beta = (PC'*PC) \ (PC' * ytrain);
%     alpha = Gamma(:,[1 K])*beta;
    
    % Calculate Predicted Response y
    y_pred = Ztest' * alpha;
    w = alpha; % terminology translation
    
%     %% PCR* with all possible values of k
%     %  returns alpha matrix with each column is the alphas with another PC added 
%     %  (alpha(:,1) is using 1 PC, alpha(:,3) is using 3 PCs).
%     for i = 1:m;
%         % transformed data matrix (named PC to be inline with Merz & Pazzani Eq. (4))
%         PC = A * Gamma(:,1:i); % = A * U;
% 
%         % coefficients
%         beta = (PC'*PC) \ (PC' * ytrain);
%         alpha(:,i) = Gamma(:,1:i)*beta;        
%     end;    
   
end