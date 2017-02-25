function rankX = getApproxRank(X)
    EIGVAL_NOISE_THRESHOLD = 0.01; % the minimal ratio between an eigval and lambda_1 for that
                                   % eigval to be considered non-zero.
    
    if all(isfinite(X(:)))
        e = eig(X); % get the eigvals in sorted lowest first.
        e = e(end:-1:1); % reverse order. put lambda_1 in e(1).
        e_ratio = e ./ e(1); % the ratio lambda_i / lambda_1
        rankX = find(e_ratio > EIGVAL_NOISE_THRESHOLD,1,'last');
    else
        rankX = Inf;
    end;
end