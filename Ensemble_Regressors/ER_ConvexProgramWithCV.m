function [y_pred,beta,lambda_mean,lambda_var,fval,mean_regressor_mse] = ER_ConvexProgramWithCV(Z, Ey, Ey2, beta_0)
% Convex program for optimizing the weights with penalties for deviation from mean and variance
% Input: Data matrix Z, Population mean Ey, Second moment Ey2, Inital guess for weights beta_0.
% Output: Response prediction y_pred, weights beta, chosen regularizers for mean and variance
% lambda_mean and lambda_var, result of the objective function at chosen lambdas fval, and the cost
% for choosing the lambdas, which is the mean regressor error (which is calculated given y=y_pred).

    % Init params
    [m,n] = size(Z);
    var_y = Ey2 - Ey.^2;
    min_cost = Inf;
    b_hat = mean(Z,2) - Ey; % approximate bias
    Zc = Z - repmat(b_hat,1,n);
    Z1c = [ones(1,n); Zc];
    search_num_of_values_mean = 1;
    search_num_of_values_var = 9;
    search_space_mean = linspace(0,0,search_num_of_values_mean);
    search_space_var = [.1 linspace(.5,4,search_num_of_values_var-1)]; % replace 0 with 0.1 in the search space to make sure that there's SOME regularization going on
    
    fprintf('\n');
    cost_surf = Inf*ones(search_num_of_values_mean,search_num_of_values_var);
    mean_regressor_mse_surf = Inf*ones(search_num_of_values_mean,search_num_of_values_var);
    for lambda_mean_cur = search_space_mean
        fprintf('#');
        for lambda_var_cur = search_space_var
            fprintf('.');
            [y_pred_cur, beta_cur, fval_cur] = ER_ConvexProgram(Z, Ey, Ey2, lambda_mean_cur, lambda_var_cur, beta_0);
            cur_mean = find(lambda_mean_cur == search_space_mean);
            cur_var  = find(lambda_var_cur == search_space_var);
            
            cost_surf(cur_mean, cur_var) = fval_cur; % use the convex program objective function as the cost
            mean_regressor_mse_surf(cur_mean, cur_var) = mean(mean((Zc - repmat(y_pred_cur',m,1)).^2,2)); % the mean regressor error
            %cost_surf(cur_mean, cur_var) = mean((Zc - repmat(y_pred_cur',m,1)).^2,2)'*beta_cur(2:end); % the weighted mean regressor error
            
            if (cost_surf(cur_mean, cur_var) < min_cost)
                min_cost = cost_surf(cur_mean, cur_var);
                y_pred = y_pred_cur;
                beta = beta_cur;
                lambda_mean = lambda_mean_cur;
                lambda_var = lambda_var_cur;
                fval = fval_cur;
                mean_regressor_mse = mean_regressor_mse_surf(cur_mean, cur_var);
            end
        end; % for lambda_var
    end; % for lambda_mean

    %% Plot cost surface
%     figure;
%     if search_num_of_values_mean > 1
%         surf(search_space_var, search_space_mean, cost_surf); ylabel('Lambda Mean'); xlabel('Lambda Var'); zlabel('Cost');
%     else
%         plot(search_space_var, cost_surf,'o-'); ylabel('Cost'); xlabel('Lambda Var');
%     end;
end