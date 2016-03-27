% returns:
%   C: the distances matrix, result of pdist, squareform(D)
%   idx: the new indexing
% In short for correlation matrices you can do this:
% D = pdist(Z,'correlation'); [~,idx] = sort(cluster(linkage(D),'maxclust',6)); C=squareform(D); imagesc(C(idx,idx));
function [C, idx] = plot_block_covariance(X,num_of_clusters_to_plot)

    function [d] = dist_func(XI,XJs)
        % mean(|xi-xj|) but only for the elements that both xi and xj are not 0.

        dist = abs(bsxfun(@minus, XI, XJs));
        both_nonzero = bsxfun(@times, XI, XJs) ~= 0;
        d = sum(dist .* both_nonzero,2) ./ sum(both_nonzero,2); % sprase mean over the stocks
        % explanation: dist is the absolute distance between each coordinate,
        % both_nonzero is 1 where both coordinates were nonzero.
    end

%     annonymized_dist_func = @(XI,XJs) ...
%         mean(abs(bsxfun(@minus, XI, XJs)) .* (bsxfun(@times, XI, XJs) ~= 0),2);
    D = pdist(X,@dist_func); %annonymized_dist_func);%'cityblock');%
    
    % Test Point - draw distances matrix before reordering
    %figure('Name','Distances Between Analysts'); imagesc(squareform(D));
    
    % Cluster the matrix
    clusterTree = linkage(D,'average');
    for m=3:10;  % search for the best number of clusters
        clusters = cluster(clusterTree, 'maxclust', m);
        fprintf('Cluster sizes for %d clusters: \t', m);
        uniq = unique(clusters)';
        for i=uniq;
            fprintf('%4d', sum(clusters == uniq(i)));
        end;
        fprintf('\n');
    end;
    clusters = cluster(clusterTree, 'maxclust', num_of_clusters_to_plot);


    % Visualize Correlations as a Heat Map
    % Construct the block data matrix by clusters
    C = squareform(D); % use the distances for correlations
    [~,idx] = sort(clusters);
    
    figure;
    imagesc(C(idx,idx));
end
