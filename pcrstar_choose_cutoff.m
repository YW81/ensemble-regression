% function [K] = pcrstar_choose_cutoff(A, Gamma, ytrain)
% Decide on a good number of principal components to use, based on Cross Validation on the training
% set. The choice here is the K that minimizes the average MSE using V-fold cross validation.
function [K] = pcrstar_choose_cutoff(A, Gamma, ytrain)
    V = 10; % V-Fold cross validation.
    [n,m] = size(A);
    Error = zeros(m,1);
    Error_plot = zeros(m,V);
    
    % Form V random partitions
    pos = randperm(n);
    edges = round(linspace(1,n+1,V+1));
    
    % for each partition v
    for v=1:V
        idxs = []; % get the v-1 indexes for training (1 part left for cross validation)
        for i=[1:v-1, v+1:V];
            idxs = [idxs, edges(i):edges(i+1)-1];
        end;
        shuffled_idxs = pos(idxs);
        
        A_no_v = A(shuffled_idxs,:);

        % for each possible cutoff

            %% CORRECT PCR* CODE
       for k=1:m            
            PC_no_v = A_no_v * Gamma(:,1:k);
            beta = (PC_no_v'*PC_no_v) \ (PC_no_v' * ytrain(shuffled_idxs));
            alpha = Gamma(:,1:k)*beta; % calculate coefficients

%             %% TEST ONLY 2 EIGS
%         for k=1:m            
%             PC_no_v = A_no_v * Gamma(:,unique([1 k]));
%             beta = (PC_no_v'*PC_no_v) \ (PC_no_v' * ytrain(shuffled_idxs));
%             alpha = Gamma(:,unique([1 k]))*beta; % calculate coefficients
            

            % CV Error
            cv_idxs = pos(edges(v):edges(v+1)-1);
            y_hat = A(cv_idxs,:)*alpha;
            Error(k) = Error(k) + sum((ytrain(cv_idxs) - y_hat).^2);            
            Error_plot(k,v) = sum((ytrain(cv_idxs) - y_hat).^2);
        end;
    end;
    
    %meanMSE = mean(Error_plot,2); figure; plot(Error_plot,'x-'); hold on; plot(meanMSE,'k--'); hold off;
    K = find(Error == min(Error));
end