function [Ztrain, ytrain, Ztest, ytest]= ...
                                TrainTestSplit(Z, y, trainset_percentage)
    n = numel(y);
    idxs = randperm(n);
    train_idxs = idxs(1:round(n * trainset_percentage));
    test_idxs = idxs(round(n * trainset_percentage)+1:end);
    
    Ztrain = Z(:,train_idxs);
    Ztest = Z(:,test_idxs);
    ytrain = y(:,train_idxs);
    ytest = y(:,test_idxs);
end