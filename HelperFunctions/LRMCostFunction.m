function fval=LRMCostFunction(C,Ey2,alpha,WANTED_RANK)
    [approxCstar, ~] = FindLowRankCstarApprox(C, Ey2, alpha);

    % Step 2: getApproxCstarRank
    rankCstar = getApproxRank(approxCstar);
    
    % Cost:
    % We want the condition number of the resulting matrix to be as high as possible. That means we
    % want to minimize rcond (1/cond). We also want to make sure the rank is equal to the wanted
    % rank. 
    % In the following cost function a mistake in the rank will cost 1, and rcond is < 1, so
    % minimizing this should mean first find the appropriate rank, and then maximize the condition#
    
    fval = rcond(approxCstar) + abs(rankCstar - WANTED_RANK);
end