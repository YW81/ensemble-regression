clear all; close all; close all hidden;
addpath Ensemble_Regressors;
addpath HelperFunctions;
rng(0)
%ROOT = './Datasets/RealWorld/CCLE/mydata/EC50/';
ROOT = './Datasets/final/misc/';
%ROOT = './Datasets/final/rf/';
%ROOT = './Datasets/final/mlp/';
%ROOT = './Datasets/final/mlp_different/';
files = dir([ROOT '*.mat']);
%files = struct(struct('name','ratings_of_sweets.mat'));

RHO_EST=zeros(3,length(files));
results_summary = {};
for file_idx=1:length(files)
    load([ROOT files(file_idx).name]);
    %fprintf('FILE: %s\n', files(file_idx).name);
    Desc = strsplit(Description,'\n'); Desc = Desc{1}; %fprintf('%s\n',Desc);

    y_true = y;
    clear y;
    y_true = double(y_true) - mean(y_true);
    Z = bsxfun(@minus, Z, mean(Z,2));
    [m,n] = size(Z);
    d = str2num(Desc(regexp(Desc,'features=')+length('features='):regexp(Desc,', n_tot=')));
    ntrain = str2num(Desc(regexp(Desc,'regressors on ')+length('regressors on '):regexp(Desc,', n=')));
    var_y = var(y_true);
    mse = @(x) mean((y_true' - x).^2 / var_y);        

    MSE_orig = zeros(m,1);
    for i=1:m
        MSE_orig(i) = mse(Z(i,:)');
    end;
    avgRE = mean(MSE_orig);

    [y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z);
    deltastar = mse(y_oracle2);
    
    fprintf('%s & %d & %d & %.2f & %d & %.2f & %.4f \\\\ \\hline\n', ...
        files(file_idx).name(1:end-4),n,ntrain,var_y,d,avgRE,deltastar);
    
end;
