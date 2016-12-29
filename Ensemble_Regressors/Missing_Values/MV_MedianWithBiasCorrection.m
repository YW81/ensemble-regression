% function y_pred = MV_MedianWithBiasCorrection(Z, Ey)
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan. Otherwise, returns the median (after bias correction) for every sample.
function y_pred = MV_MedianWithBiasCorrection(Z, Ey, threshold_m)
    Z0 = bsxfun(@minus,Z,nanmean(Z,2));

    n = size(Z,2);
    y_pred = nan*ones(n,1);
    
    b_hat = nanmean(Z,2) - Ey;
    Zc = Z - b_hat * ones(1,n);
    for i=1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Zc(:,i)));   % indices of experts that provided prediction on stock i
        m = numel(idxs);

        if m < threshold_m
            continue; 
        end
        
        y_pred(i) = median(Zc(idxs,i));
    end;
end