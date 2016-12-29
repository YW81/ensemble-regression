% function [y_pred, reg_idxs] = MV_IndependentMisfits( Z, Ey, Ey2 , threshold_m, threshold_n, loss)
% Assume independent misfit errors (diagonal C*), recover rho, find optimal weights.
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan. If two predictors have less than threshold_n elements in common, we assume Cij cannot be
% calculated accurately and ignore that one of them completely. We ignore the one with a smaller
% total number of predictions.
% The function returns reg_idxs which is a vector containing the indexes of the regressors that were
% used to make the predictions (had enough common elements with the other predictors in the ensemble
% to accurately estimate Cij).
function [y_pred, reg_idxs] = MV_IndependentMisfits( Z, Ey, Ey2 , threshold_m, threshold_n, loss)
    %Ey = .5; Z = rand(5,10) + repmat(10*rand(5,1),1,10); Z(ceil(numel(Z)*rand(.5*numel(Z),1))) = nan;
    if ~exist('loss','var')  % loss can be 'l2','l1','huber'
        loss = 'l2';
    end;
    
    [C, Z, reg_idxs] = calc_C_with_missing_values(Z, threshold_n); % note this overwrites Z
    if length(reg_idxs) < 3
        throw(MException('MV_IndependentMisfits:tooSparse', ...
                         'Not enough common predictions to calculate the covariance terms. Try smaller threshold_n.'))
    end;
    [full_m, n] = size(Z);
    y_pred = nan*ones(n,1);
    Z0 = Z - nanmean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i    
    
    %% Test (the following 2 lines generate C,diagonal C*, and rho that allow rho to be recovered):
    %m=5; Ey2 = 1;
    %v = 1+rand(full_m,1); rho_true = rand(full_m,1); C =  diag(v) + repmat(rho_true,1,full_m) + repmat(rho_true',full_m,1) - Ey2

    %% Solve linear equation  C_ij + Ey2 = rho_i + rho_j only once on all the data
    % using the matrix A which selects the off diagonal elements of rho
    % such that A\rho = C + Ey2. A is m-choose-2 combinations of regressors (off-diagonal elements only)
    subs = nchoosek(1:full_m,2);
    lin_idxs = sub2ind(size(C),subs(:,1),subs(:,2));
    Cshifted = C(lin_idxs) + Ey2; %var_y;
    A = zeros(size(subs,1), full_m);
    for j=1:size(subs,1)
        A(j,subs(j,1)) = 1;
        A(j,subs(j,2)) = 1;
    end;
    % Solve for rho: A*rho = C+Ey2
    if strcmp(loss,'huber')
        fval = @(rho) sum(huber(Cshifted - A*rho,1/(10*full_m))); % Huber-loss with delta = 1/10m
        rho = fminunc(fval,zeros(full_m,1));
    elseif strcmp(loss,'l1')
        fval = @(rho) norm(Cshifted - A*rho,1);  % L1-loss
        rho = fminunc(fval,zeros(full_m,1));
    else
        %fval = @(rho) norm(Cshifted - A*rho);  % L2-loss <==> Least-Squares
        rho = A\Cshifted; % Least-Squares
    end;

    for i=1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Z(:,i)));   % indices of experts that provided prediction on stock i
        [curC,subidxs] = get_largest_dense_submatrix(C(idxs,idxs)); % remove NaNs, and update idxs
        idxs = idxs(subidxs); % update idxs, remove unchosen indexes
        m = numel(idxs);

        if m < threshold_m
            continue; 
        end

        % Calculate weights based on the assumption of independent misfit errors
        Cinv = pinv(C(idxs,idxs),1e-5);
        w = Cinv*(rho(idxs) - ones(m,1)*(ones(1,m)*(Cinv*rho(idxs))-1)/sum(sum(Cinv)));
        %%
        y_pred(i) = Ey + Z0(idxs,i)' * w;
    end;
end

