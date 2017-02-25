% function [y_lrm, w_lrm, y_oracle_rho, w_oracle_rho] = ER_LowRankMisfit( Z, Ey, Ey2, y_true )
function [y_lrm, w_lrm, y_oracle_rho, w_oracle_rho] = ER_LowRankMisfit( Z, Ey, Ey2, y_true )
    [m,n] = size(Z);
    var_y = Ey2 - Ey^2;
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    
    C = cov(Z');
    this minimizes all terms, while it should actually only minimize the off diagonal terms in the frobenious norm
    
    %% Solve A*rho = gamma, where A is m-choose-2 combinations of regressors, 
    %  and gamma is a vector with the corresponding element in C + Ey2
    subs = nchoosek(1:m,2);
    idxs = sub2ind(size(C),subs(:,1),subs(:,2));
    gamma = C(idxs) + Ey2; %var_y;
    A = zeros(size(subs,1), m);
    for i=1:size(subs,1)
        A(i,subs(i,1)) = 1;
        A(i,subs(i,2)) = 1;
    end;
    % Solve for rho: A*rho = gamma
    rho = A\gamma;
    max_rank = floor((nchoosek(m,2) - m)/m); % the maximum rank(Cstar) that can be reconstructed

    %% Pruning
    idxs_removed = []; idxs_selected = 1:m;
    %while rank(C(idxs_selected,idxs_selected)) < size(C(idxs_selected,idxs_selected),1)
    %while cond(C(idxs_selected,idxs_selected)) > 1e6
    while (length(idxs_selected) > max_rank) || (cond(C(idxs_selected,idxs_selected)) > 1e6)
        best_cond = Inf;
        for i=idxs_selected
            cur_idxs = idxs_selected;
            cur_idxs(cur_idxs == i) = [];
            if cond(C(cur_idxs,cur_idxs)) < best_cond
                best_cond = cond(C(cur_idxs,cur_idxs)); % rho(i)
                best_idx = i;
            end;
        end;
        idxs_selected(idxs_selected == best_idx) = [];
        idxs_removed = [idxs_removed best_idx];
    end;
    fprintf('LRM: Pruning [%s]\n',num2str(idxs_removed));
    C_pruned = C(idxs_selected,idxs_selected);
    rho_pruned = rho(idxs_selected);
    m_pruned = length(idxs_selected);
    w_lrm = zeros(m,1);
    lambda = ( ones(1,m_pruned)*(C_pruned\rho_pruned) - 1 ) / (ones(1,m_pruned)*(C_pruned\ones(m_pruned,1)));
    w_lrm(idxs_selected) = C_pruned\(rho_pruned - lambda*ones(m_pruned,1));
    y_lrm = Ey + Z0(idxs_selected,:)' * w_lrm(idxs_selected);
    
%     %% Without Pruning
%     % Calculate Low-Rank Misfit Covariance Estimation Weights and Predictions
%     %C = C + 0.001*eye(m);
%     %C = C + 0.1*var_y*eye(m); % diagonal loading    
%     lambda = ( ones(1,m)*(C\rho) - 1 ) / (ones(1,m)*(C\ones(m,1)));
%     w_lrm = C\(rho - lambda*ones(m,1));
%     y_lrm = Ey + Z0' * w_lrm;
    
%      %% Using GEM (C*)
%     Cstar = zeros(m);
%     for i=1:m
%         for j=1:m
%             Cstar(i,j) = C(i,j) - rho(i) - rho(j) + Ey2;
%         end;
%     end;
%     idxs_removed = []; idxs_selected = 1:m;
%     %while rank(Cstar(idxs_selected,idxs_selected)) > max_rank
%     while cond(Cstar(idxs_selected,idxs_selected)) > 1e6
%         best_cond = Inf;
%         for i=idxs_selected
%             cur_idxs = idxs_selected;
%             cur_idxs(cur_idxs == i) = [];
%             if cond(Cstar(cur_idxs,cur_idxs)) < best_cond
%                 best_cond = cond(Cstar(cur_idxs,cur_idxs)); % rho(i)
%                 best_idx = i;
%             end;
%         end;
%         idxs_selected(idxs_selected == best_idx) = [];
%         idxs_removed = [idxs_removed best_idx];
%     end;
%     Cstar_pruned = Cstar(idxs_selected,idxs_selected);
%     w_lrm_pruned = Cstar_pruned\ones(length(idxs_selected),1) / (sum(sum(inv(Cstar_pruned))));
%     w_lrm = zeros(m,1); w_lrm(idxs_selected) = w_lrm_pruned;
%     %w_lrm = Cstar\ones(m,1) / (sum(sum(pinv(Cstar))));
%     y_lrm = Ey + Z0' * w_lrm;

    %% Oracle rho
    % Calculate Oracle Rho Weights and Predictions. Note that for an ill-conditioned C, having an
    % oracle rho is still not great. Consider diagonally loading covariance C
    rho_true = Z0*y_true' / n;
warnmode = warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:singularMatrix');
    lambda_oracle_rho = ( (ones(1,m)*(C\rho_true) ) - 1) / (ones(1,m)*(C\ones(m,1)));
    w_oracle_rho = C \ (rho_true - lambda_oracle_rho * ones(m,1));
warning(warnmode);    
    y_oracle_rho = Ey + Z0' * w_oracle_rho;
    
    [rho rho_true]
end

