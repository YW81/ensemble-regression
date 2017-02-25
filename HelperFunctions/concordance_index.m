function [c] = concordance_index(y_true, y_pred)
    c=0;
    [~,idxs] = sort(y_true);
    yhat = y_pred(idxs);
    for i=1:length(idxs)
        for j=i:length(idxs)
            c = c + double(yhat(i) < yhat(j));
        end;
    end;
    c=c/nchoosek(length(idxs),2);
end