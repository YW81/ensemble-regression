clc; close all; clear all;

n_list=[100 1000 10000 20000 50000 80000 100000]
results = cell(numel(n_list),1);
for n_itr=1:numel(n_list)

    n = n_list(n_itr);
    for itration=1:10
        dontPlot = 1;
        main_gem
        res_opt(itration) = MSE_opt;
        res_gem(itration) = MSE_gem;
        res_uncorr(itration) = MSE_uncorr;
        res_mean_f_i(itration) = MSE_mean_f_i;
        res_supervised(itration) = MSE_supervised;
    end

    clc;
    fprintf('MMSE[opt] = %g\nMMSE[gem] = %g\nMMSE[uncorr] = %g\nMMSE[mean f_i] = %g\nMMSE[supervised] = %g\n',MSE_opt,MSE_gem,MSE_uncorr,MSE_mean_f_i,MSE_supervised);

    figure('Name',['n=' num2str(n)]); 
    subplot(211);
    set(gca,'fontsize',24);
    results{n_itr} = [res_supervised' res_opt' res_gem' res_uncorr' res_mean_f_i'];
    boxplot(results{n_itr}, ...
            'labels',{'supervised', 'opt','gem','uncorr','mean f_i'});
    set(gca,'fontsize',24);
	title('MSE','fontsize',22);
    subplot(212);
    imagesc(real_Sigma); colormap(gray); colorbar;
    title('population covariance in the last iteration','fontsize',22);
    

end;