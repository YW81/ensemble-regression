% function [y_pred, w, t] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2, deltastar)
% Chooses delta to minimize the mean regressor error weighted by v_1
function [y_pred, w, t] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2, deltastar)

    % basic variables
    [m,n] = size(Z);    
    var_y = Ey2 - Ey^2;
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z'); % cov(Z') == cov(Zc') == cov(Z0')...

    % Get leading eigenvector and eigenvalue
    [v_1,lambda_1] = eigs(C,1,'lm');
    t_sign = sign(sum(v_1));
    t = t_sign * sqrt((1-deltastar)*var_y / lambda_1);
    
    t_RE = sum(v_1)/m;
    t_WRE = 1/sum(v_1);
    fprintf('t=%g,t_RE=%g,t_WRE=%g\n',abs([t t_RE t_WRE]));
    
    delta_RE = 1 - lambda_1*sum(v_1)^2 / (var_y*m^2);
    delta_WRE = 1 - lambda_1 / (var_y*sum(v_1)^2);
    fprintf('delta=%g,delta_ER=%g,delta_WRE=%g\n',deltastar, delta_RE, delta_WRE);
    
    % Calculate predictions
    w = t_WRE * v_1;
    y_pred = Ey + Z0' * w;
end