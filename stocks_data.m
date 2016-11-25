clear all;
%load ~/code/github/stocks/data/parsed_data_matrix.mat;
load ./Datasets/RealWorld/parsed_data_matrix.mat;   % After transpose, Z_{i,j} = Prediction of Target_Price(Stock_j)/Current_Price(Stock_j) BY expert i

min_ensemble_m = 4;  %require at least m predictions for given stock price target
min_predictions_per_expert = 3;

% Some ordering
Z(Z == 0) = NaN;  % missing values should be NaN (instead of zeros)
Z = Z';
Z((sum(Z > 0,2) < min_predictions_per_expert),:) = NaN; % eliminate predictors with not enough preds.
Ey = mean(y);
Ey2 = mean(y.^2);
var_y = var(y);

% Some common variables
n = size(Z,2);                       % n = number of stocks
b_hat = nanmean(Z,2) - Ey;           % This assumes we know Ey....
Z0 = Z - nanmean(Z,2)*ones(1,n);     % Z_ij = f_i(x_j) - \mu_i
mse = @(x) (nanmean((y-x).^2));

C = nancov(Z','pairwise');

%%
% Initialization: All arrays correct size, with NaN entries
y_mean = zeros(n,1)*NaN;
y_varw = zeros(n,1)*NaN;
y_invc = zeros(n,1)*NaN;
y_spectral = zeros(n,1)*NaN;
y_lrm = zeros(n,1)*NaN;
y_dgem = zeros(n,1)*NaN;

%%
% For every stock, calculate the predictions ignoring missing values
for i = 1:n
    % find indexes of relevant predictors
    idxs = find(~isnan(Z(:,i)));   % indices of experts that provided prediction on stock i
    m = numel(idxs);
    
    if m < 4
        continue; 
    end
    
    % Mean centered
    w_mean = ones(m,1)/m;
    y_mean(i) = Ey + Z0(idxs,i)'*w_mean;
    
    % Variance Weighted
    cur_vars = diag(C(idxs,idxs));
    w_varw = cur_vars / sum(cur_vars);
    y_varw(i) = Ey + Z0(idxs,i)'*w_varw;
    
    % Unsupervised GEM (InverseCov)
    cur_C = C(idxs,idxs);
    cur_C(isnan(cur_C)) = 0;  % assumption: if 2 predictors have independent entries, they are independent
    cur_Cinv = pinv(cur_C,.01);
    w_invc = cur_Cinv*ones(m,1) / (ones(1,m) * cur_Cinv * ones(m,1));
    y_invc(i) = Ey + Z0(idxs,i)'*w_invc;
    
    % Spectral
    cur_C = C(idxs,idxs);
    cur_C(isnan(cur_C)) = 0;  % assumption: if 2 predictors have independent entries, they are independent
    [v_1,lambda_1] = eigs(cur_C,1,'lm');
    t = sign(sum(v_1)) * sqrt(var_y / lambda_1);
    w_spectral = t * v_1;
    y_spectral(i) = Ey + Z0(idxs,i)' * w_spectral;
    
    % Low Rank Misfit
    if m >= 4
        cur_C = C(idxs,idxs);
        lrm_subs = nchoosek(1:m,2);
        lrm_idxs = sub2ind(size(cur_C),lrm_subs(:,1),lrm_subs(:,2));
        gamma = cur_C(lrm_idxs) + Ey2; %var_y;
        A = zeros(size(lrm_subs,1), m);
        for j=1:size(lrm_subs,1)
            A(j,lrm_subs(j,1)) = 1;
            A(j,lrm_subs(j,2)) = 1;
        end;
        rho = A\gamma;
        lambda = ( ones(1,m)*(cur_C\rho) - 1 ) / (ones(1,m)*(cur_C\ones(m,1)));
        w_lrm = cur_C\(rho - lambda*ones(m,1));
        y_lrm(i) = Ey + Z0(idxs,i)' * w_lrm;    
    end;
end;

% Unsupervised D-GEM Initialization (requires results for y_mean)
MSEs = nanmean((repmat(y_mean',size(Z,1),1) - Z).^2,2); % Estimate MSE per regressor
for i=1:n
    idxs = find(~isnan(Z(:,i)));
    m = numel(idxs);
    w_dgem = 1 ./ MSEs(idxs);
    w_dgem = w_dgem ./ sum(w_dgem);
    y_dgem(i) = Ey + Z0(idxs,i)' * w_dgem;
end;


%% Print results
mse = @(x) nanmean((y - x).^2);
L1err = @(x) nanmean(abs(y - x));
for alg=who('y_*')'
    if ~strcmp(alg{1}, 'y_true')
        fprintf('MSE=%.02f \tL1=%.02f \t%s \t%d NaNs\n',mse(eval(alg{1})),L1err(eval(alg{1})), alg{1}, sum(isnan(eval(alg{1}))));
    end;
end;

%% Plots
figure;
plot(y,y,'k-',y,y_mean,'x',y,y_varw,'o',y,y_invc,'d',y,y_spectral,'s',y,y_lrm,'v');
legend('y=y','mean','varw','invC','spectral','LRM');
xlabel('True response');
ylabel('Prediction');

figure;
corrplot([y y_mean y_varw y_invc y_dgem y_spectral y_lrm],'varnames',{'truth','mean','varw','invc','D-GEM','spectral','LRM'})

