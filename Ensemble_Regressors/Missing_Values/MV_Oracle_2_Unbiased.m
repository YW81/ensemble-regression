% function y_pred = MV_Oracle_2_Unbiased(y_true, Z, threshold_m, threshold_n)
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan. Otherwise, returns the median (after bias correction) for every sample.
function y_pred = MV_Oracle_2_Unbiased(y_true, Z, threshold_m, threshold_n)
    [C, Z, idxs_of_regressors_used ] = calc_C_with_missing_values( Z, threshold_n );
    if length(idxs_of_regressors_used) < 2
        throw(MException('MV_UnsupervisedPCRstar:tooSparse', ...
                         'Not enough common predictions to calculate for linear regression. Try smaller threshold_n.'))
    end;

    n = size(Z,2);
    y_pred = nan*ones(n,1);
    
    Ey = mean(y_true);
    Z0 = bsxfun(@minus,Z,nanmean(Z,2));
    y_centered = y_true - Ey;
    
    for i=1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Z0(:,i)));   % indices of experts that provided prediction on stock i
        [curC,subidxs] = get_largest_dense_submatrix(C(idxs,idxs)); % remove NaNs
        idxs = idxs(subidxs); % update idxs, remove unchosen indexes        
        
        m = numel(idxs);

        if m < threshold_m
            continue; 
        end
        
        %% Perform linear regression ignoring nans
        % linear regression Ax = B ==> x = A\B <==> x = (A'*A)\A'*B
        % A'*A == curC * (n-1) as curC is the unbiased sample covariance matrix
        % A'*B == Z0(idxs,:)* y_true, but need to ignore nans for every idx seperately:
        % Let r == A'*B
        
        DONT USE THIS ROUTINE - ITS INCORRECT
        r = nan*ones(m,1);
        for cur_row = 1:length(idxs)
            non_zero_entries = ~isnan(Z(idxs(cur_row),:));
            if sum(non_zero_entries) < threshold_n
                continue
            end;
            r(cur_row) = Z0(idxs(cur_row),non_zero_entries) * y_centered(non_zero_entries);
        end;
        %w = Z0(idxs,:)'\y_centered';
        w = (curC) \ r;
        y_pred(i) = Ey + Z0(idxs,i)'*w;
    end;
end