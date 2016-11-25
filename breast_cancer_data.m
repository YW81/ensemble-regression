%% Read data from files
clear all; close all;
addpath Ensemble_Regressors/

T = readtable('~/Downloads/Datasets/DREAM/BreastCancer/finalMetabricPredictions[3].txt','Delimiter','\t');
Z = table2array(T(:,2:end))';
team_names = T.Properties.VariableNames(2:end);
patientIDs = table2array(T(:,1));

T2 = readtable('~/Downloads/Datasets/DREAM/BreastCancer/Clinical_Overall_Survival_Data_from_METABRIC.txt','Delimiter',',');
y = table2array(T2(:,2))';
patientIDs2 = table2array(T2(:,1));

[m,n] = size(Z);
y_true = zeros(1,n);

for i=1:n
    y_true(i) = y(find(strcmp(patientIDs2, patientIDs(i))));
end;

% get ranking from values
tmp = sort(y_true,'descend');
[~,y_rank] = ismember(y_true,tmp);
y_orig = y_true'; y_true = y_rank; clear y_rank;

%% Transform ranking to continuous response (number of days expected survival)
assert(max(y_true) <= 480,'Cannot run this block twice without reloading the data');
% assumes y_true is ranks and transforms it to continuous response, which means you can't run this
% block twice in a row

y_rank = y_true';
y_true = y_orig';
ranking = -1*ones(size(y_true));
ranking(y_rank) = y_true;

% some y's are the same (different patients, same #days survival). This means that some ranks need
% to get the same truth #days (e.g. ranking(50) == ranking(51)).
for i=2:length(ranking)
    if ranking(i) == -1
        ranking(i) = ranking(i-1);
    end;
end;

% Map Z to continuous variables
Z_orig = Z;
Z = floor(Z); % some groups gave ranks that are actually half-ranks (e.g. rank 27.5), round down
for i=1:size(Z,1);
    for j=1:size(Z,2);
        Z(i,j) = ranking(Z(i,j));
    end;
end;

%clear T T2 y m n i patientIDs patientIDs2

%% Plot
% plot(mean((Z - repmat(y_true,m,1)).^2,2),'o');
% grid on;
% title('Is MSE a good measure of regressor success?');
% xlabel({'\bf Regressor Rank';'\rm (best regressor on the left, worst on the right)'});
% ylabel('\bf MSE');


%% Estimators
%Zfull = Z; Z = Z(1:20,:); [m,n] = size(Z);
Ey = mean(y_true);
Ey2 = mean(y_true.^2);

y_mean = ER_MeanWithBiasCorrection(Z, Ey);
y_dgem = ER_UnsupervisedDiagonalGEM(Z, Ey);
y_gem  = ER_UnsupervisedGEM(Z, Ey,Ey2);
y_spectral = ER_SpectralApproach(Z, Ey, Ey2);
y_zeromisfit = ER_LowRankMisfit(Z,Ey,Ey2,y_true);
%[y_lrm,~,y_oracle_rho] = ER_LowRankMisfitCVX(Z, Ey, Ey2, y_true);
y_rank1 = ER_Rank1Misfit(Z,Ey,Ey2,y_true,1);
%y_rank2 = ER_Rank1Misfit(Z,Ey,Ey2,y_true,2);

%% Print results
mse = @(x) mean((y_true' - x).^2);
for alg=who('y_*')'
    if ~strcmp(alg{1}, 'y_true')
        fprintf('SQRT(MSE): %g \tSPEARMAN_CORR: %g \t%s\n',sqrt(mse(eval(alg{1}))), corr(eval(alg{1}),y_true','type','spearman'),alg{1});
    end;
end;
fprintf('Best individual SQRT(MSE): %g\nBest individual SPEARMAN_CORR: %g\n', sqrt(min(mean((Z - repmat(y_true,m,1)).^2,2))), max(corr(y_true',Z')));