function [y_pred,w_new] = ER_AssumeNoCorrelation(Z, b_hat, prev_w)
    % y = Z' * w
    n = size(Z,2);
    y_pred = Z' * prev_w - repmat(prev_w',n,1)*b_hat;
    
    % update weights
    %w_uncorr{i+1} = (Z * y_uncorr{i}) ./ sum( Z.^ 2 ,2); % denominator = sum f_i^2 of all samples
    w_new = ((Z - repmat(b_hat,1,n)) * y_pred) ./ sum( (Z - repmat(b_hat,1,n)).^ 2 ,2); % denominator = sum f_i^2 of all samples
    w_new = w_new / sum(w_new); % keep sum(w) = 1.
end
