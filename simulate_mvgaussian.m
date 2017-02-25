clear all; close all; 
clc;
addpath 'Ensemble_Regressors'
addpath 'DataGeneration'
[~,hostname] = system('hostname'); hostname = strtrim(hostname);

n = 1e3; m = 6;
y = linspace(100,200,n);
Ey = mean(y); Ey2 = mean(y.^2);
min_bias = -50; max_bias = 150; min_var = 100; max_var = 200;

for dep_level=0:0.1:1
    for iter=1:10
        Z = GenerateNormalData(m ,n, y, min_bias, max_bias, min_var,  max_var, dep_level);
        
    end;
end