% function rankCstar = getApproxCstarRank(C,Ey2,alpha)
% Calls FindLowRankCstarApprox on C to get a low rank approximation to C* and then 
% calls getApproxRank on the resulting C* to get its approximate rank.
function rankCstar = getApproxCstarRank(C,Ey2,alpha)
    % Step 1: Find Low Rank C* Approximation
    [approxCstar, ~] = FindLowRankCstarApprox(C, Ey2, alpha);

    % Step 2: getApproxCstarRank
    rankCstar = getApproxRank(approxCstar);
    fprintf('Noisy Rank C* = %d for alpha = %.3g\n', rankCstar,alpha);
end % function getApproxCstarRank
