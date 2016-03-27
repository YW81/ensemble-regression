function [Z,biases,variances,Sigma]= GenerateDependentData(m ,n, true_y, ...
                                                        min_bias, max_bias, ...
                                                        min_var,  max_var)

    variances = min_var + ((max_var - min_var) * rand(m,1));
    %Sigma = wishrnd(diag(variances),df) / df;
    biases = min_bias + (max_bias - min_bias)*rand(m,1);    

%     % Block correlated covariance
%     df = 10*(max_var+min_var)/2; % degrees of freedom for Wishart distribution (what's a good value?)
%     real_Sigma = diag(variances) + ...
%             (rand*(max_var - min_var))*[ ones(9) zeros(9,m-9) ; ...
%                                  zeros(4,9) ones(4) zeros(4,m-13); ...
%                                  zeros(m-13,13) ones(m-13,m-13)];
%     
%     Sigma = wishrnd(real_Sigma,df) / df;

    % Uncorrelated Data
    Sigma = diag(variances);
    biases = biases - sum(biases); % make the sum = 0
    direction = 2*round(rand(m,1)) - 1; % +/-1 

    % MU is the true label + bias per regressor
    MU = (repmat(true_y,m,1) + repmat(direction.*biases,1,n))';
    Z = mvnrnd(MU, Sigma)';
end
