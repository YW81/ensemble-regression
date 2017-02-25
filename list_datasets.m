clear all; close all; close all hidden;
addpath Ensemble_Regressors;
addpath HelperFunctions;
rng(0)
%ROOT = './Datasets/RealWorld/CCLE/mydata/EC50/';
%ROOT = './Datasets/final/misc/';
ROOT = './Datasets/final/repeat/misc/';
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
    minRE = min(MSE_orig);

    [y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z);
    deltastar = mse(y_oracle2);
    
    fprintf('%s & %d & %d & %d & %.2f & %.2f & %.4f \\\\ \\hline\n', ...
        files(file_idx).name(1:end-4),n,ntrain,d,avgRE,minRE,deltastar);
    
    seps = strfind(files(file_idx).name,'_');
    last_separator = seps(end);
    dataset_name = files(file_idx).name;
    dataset_name = dataset_name(1:last_separator-1);
    dataset_name=strrep(dataset_name,'_',' '); dataset_name(1) = upper(dataset_name(1));
    results_summary = [results_summary; {files(file_idx).name(1:end-4),dataset_name,n,ntrain,d,avgRE,minRE,deltastar}];
end;

fprintf('Dataset \t\t& n\t& n_{train}\t& d\t& avgRE (+/-sd)\t& minRE (+/-sd)\t& deltastar (+/-sd)\n');
p=pivottable(results_summary,2,[3:8 6:8],{@mean, @mean, @mean, @mean, @mean, @mean, @std, @std, @std});
for i=1:size(p,1)
    fprintf('%20s &\t %7d &\t %6d &\t %3d &\t %.2f &\t %.2f &\t %.3f ($\\pm %.3f$) \\\\ \\hline\n',...
    p{i,1},p{i,2},p{i,3},p{i,4},p{i,5},p{i,6},p{i,7},p{i,10});
end;