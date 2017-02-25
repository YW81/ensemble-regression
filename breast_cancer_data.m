%% Read data from files
clear all; close all;
addpath Ensemble_Regressors/
addpath HelperFunctions/

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

% %% Transform ranking to continuous response (number of days expected survival)
% assert(max(y_true) <= 480,'Cannot run this block twice without reloading the data');
% % assumes y_true is ranks and transforms it to continuous response, which means you can't run this
% % block twice in a row
% 
% y_rank = y_true';
% y_true = y_orig';
% ranking = -1*ones(size(y_true));
% ranking(y_rank) = y_true;
% 
% % some y's are the same (different patients, same #days survival). This means that some ranks need
% % to get the same truth #days (e.g. ranking(50) == ranking(51)).
% for i=2:length(ranking)
%     if ranking(i) == -1
%         ranking(i) = ranking(i-1);
%     end;
% end;
% 
% % Map Z to continuous variables
% Z_orig = Z;
% Z = floor(Z); % some groups gave ranks that are actually half-ranks (e.g. rank 27.5), round down
% for i=1:size(Z,1);
%     for j=1:size(Z,2);
%         Z(i,j) = ranking(Z(i,j));
%     end;
% end;

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
var_y = Ey2 - Ey.^2;
mse = @(x) mean((y_true' - x).^2 / var_y);    

[y_oracle2, w_oracle2] = ER_Oracle_2_Unbiased(y_true, Z);
[y_best,w_best] = ER_BestRegressor(y_true,Z);
[y_mean,beta_mean] = ER_MeanWithBiasCorrection(Z, Ey);
y_biasedmean = mean(Z)'; % most teams are unbiased, so this should be equivalent to the mean
y_median = ER_MedianWithBiasCorrection(Z, Ey);
[y_dgem,w_dgem] = ER_UnsupervisedDiagonalGEM(Z, Ey);
y_gem  = ER_UnsupervisedGEM(Z, Ey,Ey2);
y_spectral = ER_SpectralApproach(Z, Ey, Ey2);
[y_spectralgivend,w_spectralgivend] = ER_SpectralApproachGivenDeltaStar(Z, Ey, Ey2,mse(y_oracle2));
[y_spectralmre,w_spectralmre] = ER_SpectralApproachDeltaMinMRE(Z, Ey, Ey2,mse(y_oracle2));
[y_spectralwmre,w_spectralwmre] = ER_SpectralApproachDeltaMinWMRE(Z, Ey, Ey2,mse(y_oracle2));
[y_indepmisfit,w_indepmisfit] = ER_IndependentMisfits(Z,Ey, Ey2); 

%% Concordance index calculation
CONCs = zeros(m,1);
for i=1:m
    CONCs(i) = concordance_index(y_true,Z(i,:));
end;
best_idx = find(CONCs == max(CONCs),1);
y_winning_method = Z(best_idx,:)';


%% Print results
mse = @(x) mean((y_true' - x).^2);
mad = @(x) mean(abs(y_true' - x));

for alg=who('y_*')'
    if ((~strcmp(alg{1}, 'y_true')) && (~strcmp(alg{1}, 'y_true_orig')))
        %fprintf('SQRT(MSE): %g \tSPEARMAN_CORR: %g \t%s\n',sqrt(mse(eval(alg{1}))), corr(eval(alg{1}),y_true','type','spearman'),alg{1});
        fprintf('MAD: %g \tCONCORDANCE: %g \t%s\n',mad(eval(alg{1})), concordance_index(y_true,eval(alg{1})),alg{1});
    end;
end;
%fprintf('Best individual SQRT(MSE): %g\nBest individual SPEARMAN_CORR: %g\n', sqrt(min(mean((Z - repmat(y_true,m,1)).^2,2))), max(corr(y_true',Z')));

fprintf('\n\n\nAlgorithm\t& $\\text{MAD}$\t& Concordance\n');
for alg=who('y_*')'
    if ((~strcmp(alg{1}, 'y_true')) && (~strcmp(alg{1}, 'y_true_orig')))
        algstr = alg{1}; algstr = algstr(3:end); algstr(1) = upper(algstr(1));
        fprintf('\\multicolumn{1}{|l|}{%s} \t& %.2f & %.3f \\\\ \\hline\n',algstr,mad(eval(alg{1})),concordance_index(y_true,eval(alg{1})));
    end;
end;

fprintf('\n\nConcordance of U-PCR weights to final standings of teams in the competition: %.3f\n', concordance_index(1:154,-abs(w_spectralgivend)));
fprintf('Concordance of D-GEM weights to final standings of teams in the competition: %.3f\n', concordance_index(1:154,-abs(w_dgem)));
fprintf('Concordance of IND weights to final standings of teams in the competition: %.3f\n', concordance_index(1:154,-abs(w_indepmisfit)));


%% MSE Results
% mse = @(x) mean((y_true' - x).^2) / var_y;
% for alg=who('y_*')'
%     if ((~strcmp(alg{1}, 'y_true')) && (~strcmp(alg{1}, 'y_true_orig')))
%         fprintf('%s\t%.2f\n',alg{1},mse(eval(alg{1})));        
%     end;
% end;
% fprintf('Best individual MSE: %g\n', min(mean((Z - repmat(y_true,m,1)).^2,2))/var_y);
% 
% % fprintf('\n\n\nAlgorithm\t& $\\text{MAD}$\t& Concordance\n');
% % for alg=who('y_*')'
% %     if ((~strcmp(alg{1}, 'y_true')) && (~strcmp(alg{1}, 'y_true_orig')))
% %         algstr = alg{1}; algstr = algstr(3:end); algstr(1) = upper(algstr(1));
% %         fprintf('\\multicolumn{1}{|l|}{%s} \t& %.2f & %.3f \\\\ \\hline\n',algstr,mad(eval(alg{1})),concordance_index(y_true,eval(alg{1})));
% %     end;
% % end;
% 
% %% Some plots
% figure(3);
% plot(gs',Zorig','.')
% grid minor; grid on;
% hold; plot(gs',gs','b-');
% title('Competitor Predictions vs. True Response');
% xlim([min(gs) max(gs)]); ylim([min(Zorig(:)) max(max(Zorig([1:12 14],:)))]);
% xlabel('Y');
% ylabel('f_i(X)')
% 
% figure(1);
% plot(y_true',Z','.')
% grid minor; grid on;
% hold; plot(y_true',y_true','b-');
% title('Unbiased Competitor Predictions vs. True Response');
% axis([-5 5 -5 50]);
% xlabel('Y');
% ylabel('$$\hat{y}(X)$$','interpreter','latex')
% 
% figure(2);
% plot(y_true',y_true','b-'); hold;
% grid minor; grid on;
% leg = {'true response'};
% for alg=who('y_*')'
%     if ((strcmp(alg{1}, 'y_dgem')) || (strcmp(alg{1}, 'y_mean')) || (strcmp(alg{1}, 'y_median')) ...
%          || (strcmp(alg{1}, 'y_spectralgivend')) || (strcmp(alg{1}, 'y_indepmisfit')))
%         plot(y_true',eval(alg{1}),'.');
%         leg = [leg, alg{1}];
%     end;
% end;
% legend(leg,'interpreter','none');
% axis([-5 5 -5 50]);
% title('Unsupervised Ensemble Methods vs. True Response');
% xlabel('Y');
% ylabel('$$\hat{y}(X)$$','interpreter','latex')