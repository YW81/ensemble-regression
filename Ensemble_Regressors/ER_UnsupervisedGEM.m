% function [y_pred, beta] = ER_UnsupervisedGEM(Z, Ey, Ey2)
% Unsupervised estimator with weights that are inversly proportional to the variance
function [y_pred, beta] = ER_UnsupervisedGEM(Z, Ey, Ey2)
    [m,n] = size(Z);
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z0');

    
    %% Pruning
    idxs_removed = []; idxs_selected = 1:m;
%     %while rank(C(idxs_selected,idxs_selected)) < size(C(idxs_selected,idxs_selected),1)
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
%     Cinv = inv(C(idxs_selected,idxs_selected));
%     new_m = length(idxs_selected);

    Cinv = pinv(C,1e-5);
    new_m = m;
    %% end of Pruning
    
    w = zeros(m,1);
    w(idxs_selected) = (Cinv * ones(new_m,1)) / sum(sum(Cinv));
    beta = zeros(m+1,1); beta(1+idxs_selected) = w(idxs_selected);
    y_pred = Ey + Z0'*w;
end