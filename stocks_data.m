clear all; close all;
addpath Ensemble_Regressors/
addpath Ensemble_Regressors/Missing_Values/
%load ~/code/github/stocks/data/parsed_data_matrix.mat;
load ./Datasets/RealWorld/parsed_data_matrix.mat;   % After transpose, Z_{i,j} = Prediction of Target_Price(Stock_j)/Current_Price(Stock_j) BY expert i

min_ensemble_m = 4;  %require at least m predictions for given stock price target
min_predictions_per_expert = 3;

% Some ordering
Z(Z == 0) = nan;  % missing values should be NaN (instead of zeros)
Z = Z';

Z(:,strcmp(labels_tickers,'NFLX')) = nan;
Z(:,strcmp(labels_tickers,'EXXI')) = nan;
Z(:,strcmp(labels_tickers,'ANGI')) = nan;
Z(:,strcmp(labels_tickers,'BBEP')) = nan;

Z(:,strcmp(labels_tickers,'FFIV')) = nan;
Z(:,strcmp(labels_tickers,'MU')) = nan;
Z(:,strcmp(labels_tickers,'WYNN')) = nan;

Z(:,strcmp(labels_tickers,'AMZN')) = nan;
Z(:,strcmp(labels_tickers,'CTRP')) = nan;
Z(:,strcmp(labels_tickers,'ATVI')) = nan;
Z(:,strcmp(labels_tickers,'NVDA')) = nan;

Z(:,strcmp(labels_tickers,'ZNGA')) = nan;
Z(:,strcmp(labels_tickers,'PTEN')) = nan;
Z(:,strcmp(labels_tickers,'MDVN')) = nan;
Z(:,strcmp(labels_tickers,'FUEL')) = nan;
Z(:,strcmp(labels_tickers,'EBAY')) = nan;
Z(:,strcmp(labels_tickers,'DISCA')) = nan;

Z(:,strcmp(labels_tickers,'GPRO')) = nan;
Z(:,strcmp(labels_tickers,'GRPN')) = nan;
Z(:,strcmp(labels_tickers,'SSYS')) = nan;
Z(:,strcmp(labels_tickers,'ZUMZ')) = nan;
Z(:,find((y > 1.5) + (y < 0.5))) = nan;

Z((sum(~isnan(Z),2) < min_predictions_per_expert),:) = NaN; % eliminate predictors with not enough preds.
Z(:,(sum(~isnan(Z),1) < min_ensemble_m)) = NaN; % eliminate stocks with not enough preds.

% [v,ix] = sort(sum(~isnan(Z),2),'descend');
% Z = Z(ix(1:10),:);
C = nancov(Z','pairwise');
[Cdense,ix] = get_largest_dense_submatrix(C);
Z = Z(ix,:);

stocks_with_preds = ~all(isnan(Z)); y = y(stocks_with_preds); labels_tickers = labels_tickers(stocks_with_preds);
Z=Z(~all(isnan(Z),2),stocks_with_preds); % remove all nan rows and columns
y_true = y'; clear y;

% mean centering
n = size(Z,2);                       % n = number of stocks
y_true = y_true - nanmean(y_true);
Zorig = Z;
Z = bsxfun(@minus, Z, nanmean(Z,2));

Ey = nanmean(y_true);
Ey2 = nanmean(y_true.^2);
var_y = var(y_true);
mse = @(x) (nanmean((y_true'-x).^2))/var_y;

%% Calculate predictions ignoring missing values
%y_oracle = MV_Oracle_2_Unbiased(y, Z, min_ensemble_m, min_predictions_per_expert);
y_biasedmean = nanmean(Zorig)';
y_naive = Ey * ones(n,1);
y_mean = MV_MeanWithBiasCorrection(Z,Ey,min_ensemble_m);
y_median = MV_MedianWithBiasCorrection(Z,Ey,min_ensemble_m);
y_upcr = MV_UnsupervisedPCRstar(Z,Ey,Ey2,min_ensemble_m,min_predictions_per_expert);
y_upcrgivend = MV_UnsupervisedPCRgivendelta(y_true,Z,Ey,Ey2,min_ensemble_m,min_predictions_per_expert);
[y_upcrRE, y_upcrWRE] = MV_UnsupervisedPCRdeltaWRE(Z,Ey,Ey2,min_ensemble_m,min_predictions_per_expert);
%y_ugem = MV_UnsupervisedGEM(Z,Ey,min_ensemble_m,min_predictions_per_expert);
%y_ugem_with_rho_est = MV_UnsupervisedGEM_with_rho_estimation(Z,Ey,min_ensemble_m,min_predictions_per_expert);
y_dgem = MV_UnsupervisedDiagonalGEM(Z,Ey,min_ensemble_m);
y_indepmisfit = MV_IndependentMisfits(Z,Ey,Ey2,min_ensemble_m,min_predictions_per_expert);

%% Print results
L1err = @(x) nanmean(abs(y_true' - x));
for alg=who('y_*')'
    if ~strcmp(alg{1}, 'y_true')
        cur_MSE = mse(eval(alg{1}));
        cur_n = sum(~isnan(eval(alg{1})));
        fprintf('RMSE=%.02f \tL1=%.02f \t%s \t%d Not NaNs (%.02f,%.02f)\n', ...
                sqrt(cur_MSE), L1err(eval(alg{1})), alg{1}, cur_n, ...
                sqrt(cur_MSE .* cur_n ./ chi2inv([.025,.975],cur_n) )); % 95% CI = https://stats.stackexchange.com/questions/78079/confidence-interval-of-rmse

        % Confidence Interval for MSE is MSE * (n/chi2(1-alpha/2, n), n/chi2(alpha/2,n))
        % where chi2(a,n) is the chi2 distribution with n degrees of freedom, and a (1-a)
        % probability of being within the range
    end;
end;

for alg=who('y_*')'
    if ~strcmp(alg{1}, 'y_true')
        cur_MSE = mse(eval(alg{1}));
        cur_n = sum(~isnan(eval(alg{1})));
        fprintf('%20s \t%.02f (%.02f,%.02f) & %.02f & %d\n', ...
                alg{1}, sqrt(cur_MSE), ...
                sqrt(cur_MSE .* cur_n ./ chi2inv([.975,.025],cur_n) ),... % 95% CI = https://stats.stackexchange.com/questions/78079/confidence-interval-of-rmse
                L1err(eval(alg{1})), cur_n);
    end;
end;


%% Plots
figure; 
plot(y_true,y_true,'k-',y_true,y_mean,'x',y_true,y_upcrgivend,'s',y_true,y_indepmisfit,'v');
grid on; grid minor; axis tight;
legend('y=y','MED','U-PCR','SIE');
xlabel('True response');
ylabel('Prediction');

%%
%corrplot([y_true y_mean y_ugem y_dgem y_upcrgivend y_lrm],'varnames',{'truth','mean','uGEM','D-GEM','spectral','LRM'})

m=size(Z,1); figure; hold on;for i=1:m; plot(y_true,Z(i,:),'.'); end; plot(y_true,y_true,'x'); hold off; grid on; grid minor;