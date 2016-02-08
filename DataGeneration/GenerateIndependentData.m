function [Z,biases,variances,Sigma]= GenerateIndependentData(m ,n, true_y, ...
                                                        min_bias, max_bias, ...
                                                        min_var,  max_var)
    % Covariance matrix 
    variances = min_var + ((max_var - min_var) * rand(m,1));
    biases = min_bias + (max_bias - min_bias)*rand(m,1);    

    Sigma = diag(variances);    

    % MU is the true label + bias per regressor
    MU = (repmat(true_y,m,1) + repmat(biases,1,n))';
    Z = mvnrnd(MU, Sigma)';
end
