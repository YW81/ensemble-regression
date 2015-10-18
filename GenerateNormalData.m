function [Z,biases,variances,Sigma]= GenerateNormalData(m ,n, true_y, ...
                                                        min_bias, max_bias, ...
                                                        min_var,  max_var)
    % Covariance matrix 
    % (using Wishart distribution with variances around min_var/max_var)
    df = 2*m; % degrees of freedom for Wishart distribution (what's a good value?)
    variances = min_var + ((max_var - min_var) * rand(m,1));
    Sigma = wishrnd(diag(variances),df) / df;
    biases = min_bias + (max_bias - min_bias)*rand(m,1);    
%     % Block correlated covariance
    Sigma = diag(variances) + ...
            (min_var+ (max_var - min_var)*rand)*[ ones(9) zeros(9,m-9) ; ...
                                 zeros(4,9) ones(4) zeros(4,m-13); ...
                                 zeros(m-13,13) ones(m-13,m-13)];
    
    Sigma = wishrnd(Sigma,df) / df;
    % Uncorrelated Data
%     Sigma = diag(variances);
%     biases = biases - sum(biases); % make the sum = 0

    % MU is the true label + bias per regressor
    MU = (repmat(true_y,m,1) + repmat(biases,1,n))';
    Z = mvnrnd(MU, Sigma)';
end
