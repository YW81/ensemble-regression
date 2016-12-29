function [ C, newZ, chosen_indexes ] = calc_C_with_missing_values( Z, threshold_n )
%function [ C, newZ, chosen_indexes ] = calc_C_with_missing_values( Z, threshold_n )
%Calculates the elements of the covariance matrix C, given the data matrix Z in the presence of
%missing values (labeled by nan).
%   For every pair of predictors in Z, if less than threshold_n common elements exist in the data
%   matrix the one with less samples will be pruned. Otherwise, the covariance will be calculated
%   ignoring the missing values.
    m = size(Z,1);
    C = nancov(Z','pairwise');
    chosen_indexes = 1:m;
    for i=1:m
        for j=1:i-1
            num_common_elements = sum(all(~isnan(Z([i j],:))));
            if num_common_elements == 0
                % assumption: if 2 predictors have independent entries, they are independent
                C(i,j) = 0;
                C(j,i) = 0;
            elseif num_common_elements < threshold_n % if not enought elements in common
                C(i,j) = nan;
                C(j,i) = nan;
                regressor_to_eliminate = j;
                if diff(sum(~isnan(Z([i j],:)),2)) > 0  % if the j-th regressor has more elements
                    regressor_to_eliminate = i;
                end;
                chosen_indexes(chosen_indexes == regressor_to_eliminate) = [];
            end;
        end;
    end;
    
    C = C(chosen_indexes, chosen_indexes);
    %C(isnan(C)) = 0; % need to use re-weighted least squares, which will take into account the
                      % number of samples for every element of C when fitting/inverting
    newZ = Z(chosen_indexes,:);
end

