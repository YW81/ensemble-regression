% DON'T USE, BUGGY CODE. Use ER_linear_regression_oracle instead.
function [y_pred,w,R,rho] = calculate_oracle(Ey, min_y, max_y, Sigma_true, biases_true, Z)
    rho = zeros(numel(biases_true),1);
    R = zeros(numel(biases_true));

    %% Calculate R, rho
    for i=1:numel(biases_true)
        for j=1:numel(biases_true)
            R(i,j) = Sigma_true(i,j) + ((Ey + biases_true(i)) * (Ey + biases_true(j)));
        end % j
        
        func = @(y) (y.^3/3 + biases_true(i)*y.^2/2);
        rho(i) = (func(max_y) - func(min_y)) / (max_y - min_y);
    end % i
    
    %% Calculate weights
    %w = inv(R)*rho;
    mu = Ey+biases_true;
    w = inv(R - biases_true*mu' - mu*biases_true' + biases_true*biases_true') ...
        *(rho - Ey * biases_true);
    
    %% Calculate oracle predictions
    y_pred = (Z - repmat(biases_true,1,size(Z,2)))'*w;
end