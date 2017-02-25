% function [Cstar, rho] = FindLowRankCstarApprox(C, Ey2, alpha)
% Implements the SDP to find a low rank approximation for C* given C.
function [Cstar, rho] = FindLowRankCstarApprox(C, Ey2, alpha)
    m = size(C,1);
    cvx_begin sdp
        variable Cstar(m,m) symmetric;
        variable rho(m)
        expression Gamma(m,m);
        Cstar == semidefinite(m);

        for i=1:m; for j=1:m; Gamma(i,j) = C(i,j) + Ey2 - rho(i) - rho(j); end; end;

        minimize(norm(Gamma-Cstar,'fro') + alpha*norm_nuc(Cstar))
        subject to
            Cstar+Cstar' >= 0
    cvx_end
    % If the problem was not solved return nan;
    if ~(strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved'))
        Cstar = nan;
        rho = nan;
    end;
end % function FindLowRankCstarApprox
