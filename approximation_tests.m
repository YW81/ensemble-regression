%%
clear all; close all; 
%clc;
addpath './Ensemble_Regressors'
%DataFilename = 'ensemble_white_wine.mat';
%DataFilename = 'ensemble.mat'; 
%DataFilename = 'ensemble_CCPP.mat'; 
%DataFilename = 'ensemble_abalone.mat'; 
%DataFilename = 'ensemble_bike_sharing.mat'; 
%DataFilename = 'MVGaussianDepData.mat'; 
%load(['./Datasets/' DataFilename]); %y=y_true;

DataFilename = '/home/omer/code/github/ensemble-regression/ensemble.mat';
load(DataFilename);

[m,n] = size(Z);
var_y = Ey2 - Ey.^2;

%% Print data desc
fprintf('Data desc:\n');
fprintf('m = %d, n = %d, Ey = %g, Ey2 = %g, Var(y) = %g\n', m, n, Ey, Ey2, var_y);


%% f = y + b + e, check assumption b_hat = b (bias estimation error)
b = mean(Z - repmat(y,m,1),2); % real bias
b_hat = mean(Z,2) - Ey; % approximate bias
fprintf('Bias estimation error =\t');
fprintf('%.2g;\t', b - b_hat); fprintf('\n');
fprintf('\t\t\t%% of E[y] =\t');
fprintf('%.2f%%;\t', 100*(b - b_hat) / Ey); fprintf('\n');

%% f = y + b + e, check assumption that g_i = E[e] = 0, given b_hat instead of b
e = Z - repmat(y, m,1) - repmat(b_hat,1,n); % approximate error term
fprintf('\ng_i(x) = E[y e_i]:\n');
g = mean(repmat(y,m,1) .* e, 2);
fprintf('\tg_i \tg_i / var(y)\n');
disp([g, g ./ var_y])

%% f = y + b + e, check assumption that E[e_i e_j] is small
% fprintf('\n');
% E = e*e' /n;
% disp(E)

%% g
fprintf('\nmax g ratio = %f\n',max(g/max(max(cov(Z') - Ey))));
fprintf('max g / eig = %f\n',max(abs((g/(max(eig(cov(Z') - Ey)))))));
% figure; plot(y,e','o','MarkerSize',10); legend('show'); grid on; axis tight; set(gca,'FontSize',14);
% xlabel('True Response (y)'); ylabel('Error i (\epsilon_i)'); title(strrep(DataFilename,'_',' '));