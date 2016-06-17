%%
clear all; close all; 
clc;
addpath './Ensemble_Regressors'
%DataFilename = 'ensemble_white_wine.mat';
%DataFilename = 'ensemble.mat'; 
%DataFilename = 'ensemble_CCPP.mat'; 
%DataFilename = 'ensemble_abalone.mat'; 
%DataFilename = 'ensemble_bike_sharing.mat'; 
%DataFilename = 'MVGaussianDepData.mat'; 
%load(['./Datasets/' DataFilename]); %y=y_true;

% DataFilename = 'auto_basketball.mat'; 
% load(['./Datasets/auto/' DataFilename]); %y=y_true;

% DataFilename = '/home/omer/code/github/ensemble-regression/ensemble.mat';
% load(DataFilename);

%%
ROOT = './Datasets/auto/';
file_list = dir([ROOT '*.mat']);
datasets = cell(1,length(file_list));
for i=1:length(file_list)
    datasets{i}=[ROOT file_list(i).name];
end;

for dataset_name=datasets
    DataFilename = dataset_name{1};
    load(DataFilename)

    if isempty(strfind(dataset_name{1},'MVGaussianDepData')) % if ~strcmp(dataset_name{1},'./Datasets/MVGaussianDepData.mat') % 
        y_true = double(y); clear y; % renmae y to y_true and make sure both y and y_true are double
        ytrain = double(ytrain);     % (in the sweets dataset y is integer)
    end;
    y = y_true;
    
    
%% Print data desc
[m,n] = size(Z);
var_y = Ey2 - Ey.^2;

fprintf('%s\n',DataFilename)
fprintf('m = %d, n = %d, Ey = %g, Ey2 = %g, Var(y) = %g\n', m, n, Ey, Ey2, var_y);


%% f = y + b + e, check assumption b_hat = b (bias estimation error)
b = mean(Z - repmat(y,m,1),2); % real bias
b_hat = mean(Z,2) - Ey; % approximate bias
% fprintf('Bias estimation error =\t');
% fprintf('%.2g;\t', b - b_hat); fprintf('\n');
% fprintf('\t\t\t%% of E[y] =\t');
% fprintf('%.2f%%;\t', 100*(b - b_hat) / Ey); fprintf('\n');

%% Plot eCDF
% Zc = Z - repmat(b_hat,1,n); 
% figure; hold all; ecdf(y); 
% for i=1:m; ecdf(Zc(i,:)); end; 
% grid on; ylim([-.1 1.1]); title(DataFilename,'Interpreter','none');
% if isempty(strfind(dataset_name{1},'MVGaussianDepData'));
%     legend(['y';mat2cell(names,ones(6,1), 320)],'Location','southoutside'); 
% end;


%% f = y + b + e, check assumption that g_i = E[e] = 0, given b_hat instead of b
e = Z - repmat(y, m,1) - repmat(b_hat,1,n); % approximate error term
% fprintf('\ng_i(x) = E[y e_i]:\n');
g = mean(repmat(y,m,1) .* e, 2);
% fprintf('\tg_i \tg_i / var(y)\n');
% disp([g, g ./ var_y])

n_sample = floor(.95*n);
idxs = randsample(n,n_sample);
y_sample = y(idxs);
e_sample = Z(:,idxs) - repmat(y_sample,m,1) - repmat(b_hat,1,n_sample);
g_sample = mean(repmat(y_sample,m,1) .* e_sample,2);

fprintf('g sample MSE:\t %f\n', mean((g-g_sample).^2));
fprintf('g vs g_sample\n');
[g g_sample]

%% f = y + b + e, check assumption that E[e_i e_j] is small
% fprintf('\n');
% E = e*e' /n;
% disp(E)

%% g_i = E[y * (y + b_i + e_i)], if we omit the last term, how much are we changing g_i?
(mean(repmat(Ey2 + Ey*b_hat,1,n) .* repmat(y,m,1),2) ./ g)


end; % for dataset_name in datasets
