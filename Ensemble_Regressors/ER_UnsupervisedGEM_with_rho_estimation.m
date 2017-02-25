% function [y_pred, w] = ER_UnsupervisedGEM_with_rho_estimation(Z,Ey)
% Unsupervised GEM, estimator of the form y = Ey + sum_i ( w_i (f_i-mu_i) ) where the sum of the weights equals 1
% and rho is estimated using the mean predicted response
function [y_pred, w] = ER_UnsupervisedGEM_with_rho_estimation(Z,Ey)
    [m n] = size(Z);
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z0');
    rho = mean(bsxfun(@times,bsxfun(@minus, Z, mean(Z,2)),mean(Z)),2);
    
    %% Pruning
    idxs_removed = []; idxs_selected = 1:m;
%     while cond(C(idxs_selected,idxs_selected)) > 1e6
%         best_cond = Inf;
%         for i=idxs_selected
%             cur_idxs = idxs_selected;
%             cur_idxs(cur_idxs == i) = [];
%             if cond(C(cur_idxs,cur_idxs)) < best_cond
%                 best_cond = cond(C(cur_idxs,cur_idxs));
%                 best_idx = i;
%             end;
%         end;
%         idxs_selected(idxs_selected == best_idx) = [];
%         idxs_removed = [idxs_removed best_idx];
%     end;
%     
%     C_pruned = C(idxs_selected,idxs_selected);
%     Cinv = pinv(C_pruned); % no need for pseudo-inverse/diagonal-loading since the matrix is assumed to be well conditioned at this point (after pruning)
%     new_m = length(idxs_selected);
    Cinv = pinv(C,1e-5);
    new_m = m;
    %% end of Pruning
    
    %w = C\(rho - ones(m,1)*(ones(1,m)*(C\rho)-1)/sum(sum(pinv(C))));
    w = zeros(m,1);
    w(idxs_selected) = Cinv*(rho(idxs_selected) - ones(new_m,1)*(ones(1,new_m)*(Cinv*rho(idxs_selected))-1)/sum(sum(Cinv)));
    y_pred = Ey + Z0'*w;
end