% function [y_pred, w, s] = ER_SemiSupervisedWeightsComposition(Ztrain, ytrain, Ztest)
function [y_pred] = ER_SemiSupervisedWeightsComposition(Ztrain, ytrain, Ztest)
    [m,n_train] = size(Ztrain);
    [m_test, n] = size(Ztest);
    
    assert(m == m_test, 'Ztrain and Ztest must have the same number of rows')
   
    % Get PCR* weights
    [y_pcrstar,w_pcrstar,~] = ER_PCRstar(Ztrain, ytrain, Ztest);
    %[y_semi_s,w_semi_s,~] = ER_SemiSupervised_S_SelectionSpectralApproach(Ztrain, ytrain, Ztest);
    [y_us,w_us,~] = ER_SpectralApproach(Ztest, mean(ytrain), mean(ytrain.^2));
    
    % return results
    y_pred = (y_pcrstar + y_us)/2;
end