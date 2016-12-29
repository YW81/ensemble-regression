function [ denseC, chosen_indexes ] = get_largest_dense_submatrix( C )
%function [ C, chosen_indexes ] = get_largest_dense_submatrix( C )
%Returns the largest submatrix of C that does not have any nans
%   Removes the rows/cols with the most nans iteratively, until no nans are left.
%   Assumes C is square symmetric matrix.
    m = size(C,1);
    chosen_indexes = 1:m;
    for i=1:m
        [val,idx] = max(sum(isnan(C(chosen_indexes, chosen_indexes))));
        if val == 0 % exit the loop if no nans are found
            break;
        end;
        chosen_indexes(idx) = [];
    end;
    
    denseC = C(chosen_indexes, chosen_indexes);
end

