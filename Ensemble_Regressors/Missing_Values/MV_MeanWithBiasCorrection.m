% function [y_pred] = MV_MeanWithBiasCorrection(Z, Ey, threshold_m)
% Assumes Z has nan entries and treats them as missing values
% if less than threshold_m values are given for a single prediction item, result for that element is
% nan.
function [y_pred] = MV_MeanWithBiasCorrection(Z, Ey, threshold_m)
    %Ey = .5; Z = rand(10,100) + repmat(10*rand(10,1),1,100); Z(ceil(numel(Z)*rand(.5*numel(Z),1))) = nan;
    n = size(Z,2);
    y_pred = nan*ones(n,1);
    
    b_hat = nanmean(Z,2) - Ey;           % This assumes we know Ey....
    Zc = Z - b_hat * ones(1,n);
    
    for i = 1:n
        % find indexes of relevant predictors
        idxs = find(~isnan(Zc(:,i)));   % indices of experts that provided prediction on stock i
        m = numel(idxs);

        if m < threshold_m
            continue; 
        end

        % Mean centered
        w = ones(m,1)/m;
        y_pred(i) = Zc(idxs,i)'*w;
    end
end