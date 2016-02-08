clc; close all; clear all;

%%

vars_list=[100 1000 10000 20000 50000 80000 100000];
results = cell(numel(vars_list),1);
for v_itr=1:numel(vars_list)

    for iteration=1:10
        %% main_gem params init
        n = 1000;
        m = 15;
        num_iterations = 20;

        n_training_set = 200;
        y_true = linspace(100,200,n + n_training_set);
        Ey = 150;

        % min/max bias/variance
        min_bias = -.75*(max(y_true) - min(y_true)); % -75
        max_bias = 1.5*(max(y_true) - min(y_true));  %+150
        max_var = vars_list(v_itr);     % 100^2
        min_var = max_var/16;                         % 

        A_dontPlot = 1;
        A_outsideParams = 1;
        
        %% Call main
        single_run;
        
        clear A_dontPlot;
        clear A_outsideParams;
        
        %% Collect results
        
        res_uncentered_gem(iteration) = MSE_uncentered_gem;
        res_gem(iteration) = MSE_gem;
        res_uncorr(iteration) = MSE_uncorr;
        res_mean_f_i(iteration) = MSE_mean_f_i;
        res_supervised(iteration) = MSE_supervised;
        res_oracle(iteration) = MSE_oracle;
%         res_oracle2(iteration) = MSE_oracle2;
        res_f_best(iteration) = MSE_f_best;
        res_2me(iteration) = MSE_2me;
        res_pcGEM(iteration) = MSE_pcGEM;
    end

    fprintf('%s\n',repmat('=',1,79)); % print separator
    fprintf(['MMSE[uncentered gem] = %g\nMMSE[gem] = %g\nMMSE[uncorr] = %g\n' ...
             'MMSE[mean f_i] = %g\nMMSE[2 me] = %g\nMMSE[supervised] = %g\nMMSE[oracle] = %g\n' ...
             ... %MMSE[oracle2] = %g\n
             'MMSE[best f] = %g\nMMSE[PC GEM] = %g\n'], ...
             min(res_uncentered_gem), min(res_gem), min(res_uncorr), min(res_mean_f_i), ...
             min(res_2me), min(res_supervised), min(res_oracle), ...%min(res_oracle2), 
             min(res_f_best), min(res_pcGEM));

    figure('Name',['max var=' num2str(vars_list(v_itr))]); 
    subplot(121);
    %set(gca,'fontsize',24);
    results{v_itr} = [res_oracle' ...
                      ...%res_oracle2' 
                      res_supervised' res_pcGEM' res_2me' res_uncentered_gem' ...
                      res_gem' res_uncorr' res_mean_f_i', res_f_best'];
    boxplot(log10(results{v_itr}), ...
            'labels',{'oracle', ...%'Emprical R/rho', 
                      'supervised', 'Perrone & Cooper GEM','2nd Moment', ...
                      'uncentered-gem','gem','uncorr', 'mean f_i','best f_i'}, ...
            'labelorientation','inline');
    %set(gca,'fontsize',24);
    ylim([log10(min(vars_list)) log10(max(vars_list))]);
	title('log_{10}(MSE)','fontsize',22); grid on;
    subplot(122);
    imagesc(real_Sigma); colormap(gray); colorbar;
    title({'population covariance';'in the last iteration'},'fontsize',22);
    
    %print('-deps', ['results/n=' num2str(n_list(n_itr)) '.eps']);
end;

for i=1:numel(results)
    s = size(results{i});
    res_MSE(i,:,:) = results{i}; % size(res) = [ numel(n_list)=7 iterations=10 types=8 ]
end

res_MSE = permute(res_MSE, [ 3 2 1]); % size = [ types=8 iterations=10 numel(n_list)=7 ]
figure('Name','MSE by max variance'); hold all;
%set(gcf,'DefaultMarker','x');
for i=1:size(res_MSE,1)-1 % for each type except for best_f_i
    plot(log10(vars_list), reshape(log10(nanmean(res_MSE(i,:,:),2)),numel(vars_list),1),'d-');
end;
h=legend('oracle', 'supervised', 'Perrone \& Cooper GEM', '2nd Moment', 'uncentered-gem','gem', ...
       'uncorr', 'bias corrected $\bar f_i$');%,'best f_i');
set(h,'interpreter','latex');
ylabel('log_{10}(MSE)'); xlabel('log_{10} Variance');
title('MSE vs max variance'); grid on;
%ylim([0 2*log10(max_var)]);