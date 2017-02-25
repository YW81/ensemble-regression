clear all; close all;
addpath HelperFunctions/

filename_contains_index = false;
%dataset = 'main.mat';
%dataset = 'icml.mat';
%dataset = 'icml_repeat.mat'; filename_contains_index = true;
%dataset = 'bounding_boxes_results.mat';
dataset = 'dream8_results.mat';
load(dataset); 

%c=pivottable(results_summary,2,4,[3 3 3],{@std, @(x) std(x .* (x < 1)), @(x) sum(x > 1)} ); c([1 end-3 end],:);
%p=pivottable(results_summary,2,4,[3 3],{@mean,@std})
if filename_contains_index
    p=pivottable(results_summary,2,4,3,@mean);
else
    p=pivottable(results_summary,2,1,3,@mean);
end
a=cell2mat(p(2:end,2:end))
p(:,1)

%% for paper
idx_orc      = find(strcmp(p(2:end,1),'y_oracle2'));
idx_mean     = find(strcmp(p(2:end,1),'y_MEAN'));
idx_med      = find(strcmp(p(2:end,1),'y_MED'));
idx_upcr     = find(strcmp(p(2:end,1),'y_UPCRrhoINDB'));
idx_upcr2c   = find(strcmp(p(2:end,1),'y_UPCRrhoINDB2c'));
idx_upcr_ss  = find(strcmp(p(2:end,1),'y_UPCRrhoINDB_ss'));
idx_upcr2c_ss= find(strcmp(p(2:end,1),'y_UPCRrhoINDB2c_ss'));
idx_best     = find(strcmp(p(2:end,1),'best'));
idx_best_mse = find(strcmp(p(2:end,1),'y_BEST_MSEhat'));
idx_best_rho = find(strcmp(p(2:end,1),'y_BEST_RHOhat'));

diff_ss = a(idx_upcr2c_ss,:) - a(idx_upcr2c,:) > 1e-2;
diff_2c = a(idx_upcr2c_ss,:) - a(idx_upcr_ss,:) - a(idx_upcr2c_ss,:) > 5e-3;
diff_ss_2c = a(idx_upcr2c_ss,:) - a(idx_upcr,:) - a(idx_upcr2c_ss,:) > 5e-3;

if strcmp(dataset,'icml_repeat.mat') diff_ss = false(size(diff_ss)); end;

%% g2 vs. delta
t=pivottable(results_summary,1,2,3,@mean);
t = [t, ['g2';num2cell(G2_EST./VAR_Y)]]; % add a column with g2/var_y
j = size(t,2)+1; 

if filename_contains_index  % only on repeated experiments, where the std can be calculated
    t(1,j) = cellstr('dataset');
    for i=2:size(t,1)
        dataset_name = t{i,1};
        seps = strfind(dataset_name,'_');
        last_separator = seps(end);
        t(i,j) = cellstr(dataset_name(1:last_separator-1));
    end;
    t=t(2:end,[1 j-2 j-1 j]); % columns:     filename,'y_oracle2','g2','dataset'
    t=pivottable(t,4,[2 3],{@mean, @mean}); % columns:'dataset','y_oracle2','g2'
else
    t=pivottable(t(2:end,[1 j-2 j-1]),1,[2 3],{@mean, @mean}); % columns:'dataset','y_oracle2','g2'
end;
delta=cell2mat(t(:,2));
g2=cell2mat(t(:,3));

fig = 3; figure(fig); clf; set(gca,'fontsize',16);  set(fig,'Name','g2 vs. delta* (gs_vs_delta)')
hold on; grid on; grid minor; msize = 8;
plot(delta,g2,'ko','markerfacecolor','k','markersize',msize); %mean
xlabel('\delta_{OR}=MSE(oracle)/Var(Y)'); ylabel('g_2/Var(Y)');
axis([0 1 0 1]);

set(fig,'PaperPositionMode','auto');
set(fig,'Position',[500 700 900 378]);
set(gca,'Position', [.1 .2 .85 .70]);

%% Perfect Prediction Possible (category 1)
type1 = a(idx_orc,:) < .1;
if strcmp(dataset,'bounding_boxes_results.mat'); type1=true(size(a(idx_orc,:))); end;
if any(type1)
    fig = 1;
    figure(fig); clf;  msize = 8; set(fig,'Name','Perfect Prediction Possible (cat1_comparison)')
    set(gca,'fontsize',14); 
    hold on; grid on; grid minor; 
    plot(a(idx_orc,type1),a(idx_mean,type1)-a(idx_orc,type1),'ko','markerfacecolor','k','markersize',msize); %mean
    plot(a(idx_orc,type1),a(idx_med,type1)-a(idx_orc,type1),'k>','markersize',msize,'markerfacecolor','k');   %median
    plot(a(idx_orc,type1),a(idx_upcr2c_ss,type1)-a(idx_orc,type1),'ms','markersize',msize,'markerfacecolor','m');   %UPCR
    plot(a(idx_orc,diff_2c & type1),a(idx_upcr_ss,diff_2c & type1)-a(idx_orc,diff_2c & type1),'r<','markersize',msize,'markerfacecolor','r');   %UPCR 2C
    plot(a(idx_orc,diff_ss & type1),a(idx_upcr2c,diff_ss & type1)-a(idx_orc,diff_ss & type1),'o','markersize',msize,'linewidth',2,'markeredgecolor',[0 0 .9]);   %UPCR 1C with SUBSET
    plot(a(idx_orc,diff_ss_2c & type1),a(idx_upcr,diff_ss_2c & type1)-a(idx_orc,diff_ss_2c & type1),'v','markersize',msize,'linewidth',2,'markeredgecolor',[0 .7 0]);   %UPCR 2C SUBSET

    leg = {'MEAN','MED','UPCR'};
    if any(diff_2c & type1); leg = [leg,'UPCR 1-PC']; end;
    if any(diff_ss & type1); leg = [leg, 'UPCR (ALL REGRs)']; end;
    if any(diff_ss_2c & type1); leg = [leg, 'UPCR 1-PC (ALL REGRs)']; end;
    legend(leg,'Location','NorthWest'); 
    xlabel('MSE(oracle)/Var(Y)'); ylabel('EXCESS RISK');
    box on;

    if strcmp(dataset,'bounding_boxes_results.mat')
        xlim([0.02 0.19]);
        set(gca,'fontsize',10);
        set(fig,'PaperPositionMode','auto'); 
        set(fig,'Position',[1000 200 400 200]); set(gca,'Position', [.18 .2 .80 .75]);
        print -depsc 'plots/icml/bounding_box_accuracy.eps';
        fprintf(2,'SAVED PLOT 2 TO plots/icml/bounding_box_accuracy.eps\n');
    elseif strcmp(dataset,'icml_repeat.mat')
        %xlim([0.02 0.19]);
        set(gca,'fontsize',10);
        set(fig,'PaperPositionMode','auto'); 
        set(fig,'Position',[1000 200 400 200]); set(gca,'Position', [.17 .2 .8 .75]);%set(gca,'Position', [.17 .2 .8 .75]);
        %print -depsc 'plots/icml/cat1_comparison.eps';
        %fprintf(2,'SAVED PLOT 2 TO plots/icml/cat1_comparison.eps\n');
    else
        %axis([0 0.1 0 .1+max(max(a(:,type1)))]); 
        set(fig,'PaperPositionMode','auto');
        set(fig,'Position',[500 700 500 378]);
        set(gca,'Position', [.2 .22 .70 .70]);
    end
end;

%% Difficult but solvable (category 2)
type2 = (a(idx_orc,:) > .1) & (a(idx_orc,:) < .8);
if any(type2)    
    msize = 7;
    fig = 2; h=figure(fig); clf; set(gca,'fontsize',20);  set(fig,'Name','Difficult but doable (cat2_comparison)')
    plot(a(idx_orc,type2),a(idx_mean,type2)-a(idx_orc,type2),'ko','markerfacecolor','k','markersize',msize); %mean
    hold on; grid on; grid minor; 
    plot(a(idx_orc,type2),a(idx_med,type2)-a(idx_orc,type2),'k>','markersize',msize,'markerfacecolor','k');   %median
    plot(a(idx_orc,type2),a(idx_upcr2c_ss,type2)-a(idx_orc,type2),'ms','markersize',msize,'markerfacecolor','m');   %UPCR
    plot(a(idx_orc,diff_2c & type2),a(idx_upcr_ss,diff_2c & type2)-a(idx_orc,diff_2c & type2),'r<','markersize',msize,'markerfacecolor','r');   %UPCR 2C
    plot(a(idx_orc,diff_ss & type2),a(idx_upcr2c,diff_ss & type2)-a(idx_orc,diff_ss & type2),'o','markersize',msize,'linewidth',2,'markeredgecolor',[0 0 .9]);   %UPCR 1C with SUBSET
    plot(a(idx_orc,diff_ss_2c & type2),a(idx_upcr,diff_ss_2c & type2)-a(idx_orc,diff_ss_2c & type2),'v','markersize',msize,'linewidth',2,'markeredgecolor',[0 .7 0]);   %UPCR 2C SUBSET

    leg = {'MEAN','MED','UPCR'};
    if any(diff_2c & type2); leg = [leg,'UPCR 1-PC']; end;
    if any(diff_ss & type2); leg = [leg, 'UPCR (ALL REGRs)']; end;
    if any(diff_ss_2c & type2); leg = [leg, 'UPCR 1-PC (ALL REGRs)']; end;
    legend(leg,'Location','NorthWest'); 
    xlabel('MSE(oracle) / Var(Y)'); ylabel('EXCESS RISK');
    %set(gca,'yscale','log'); set(gca,'ytick',[.01 .1])

    if strcmp(dataset,'dream8_results.mat')
        axis([.1 .35 0 .225]); 
        set(fig,'PaperPositionMode','auto'); set(fig,'Position',[300 500 400 200]); set(gca,'Position', [.17 .2 .8 .78]);
        print -depsc 'plots/icml/dream8_accuracy.eps';
        fprintf(2,'SAVED PLOT 2 TO plots/icml/dream8_accuracy.eps\n');
    elseif strcmp(dataset,'icml_repeat.mat')
        %axis([.1 .35 0 .225]); 
        set(fig,'PaperPositionMode','auto'); set(fig,'Position',[300 500 400 200]); set(gca,'Position', [.17 .2 .8 .75]);
%        print -depsc 'plots/icml/cat2_comparison.eps';
%        fprintf(2,'SAVED PLOT 2 TO plots/icml/cat2_comparison.eps\n');
    else
        axis([0 .8 0 .42]); 
        set(fig,'PaperPositionMode','auto');
        set(fig,'Position',[500 700 900 378]);
        set(gca,'Position', [.12 .22 .70 .70]);
    end;
end;
% %% Print Results Table
% if filename_contains_index  % only on repeated experiments, where the std can be calculated
%     t=pivottable(results_summary,1,2,3,@mean);
%     t = [t, ['n';num2cell(G2_EST./VAR_Y)]]; % add a column with g2/var_y
%     j = size(t,2)+1; 
%     t(1,j) = cellstr('dataset');
%     for i=2:size(t,1)
%         dataset_name = t{i,1};
%         seps = strfind(dataset_name,'_');
%         last_separator = seps(end);
%         t(i,j) = cellstr(dataset_name(1:last_separator-1));
%     end;
%     t=t(2:end,[1 j-2 j-1 j]); % columns:     filename,'y_oracle2','g2','dataset'
%     t=pivottable(t,4,[2 3],{@mean, @mean}); % columns:'dataset','y_oracle2','g2'
% 
%     delta=cell2mat(t(:,2));
%     g2=cell2mat(t(:,3));
%     
%     
%     
%     t = pivottable(results_summary,4,2,3,@mean);
%     e = pivottable(results_summary,4,2,3,@std);
%     fprintf('\bf Name & \multicolumn{1}{c|}{$n$} & $n_{\mbox{\tiny train}}$ &  \multicolumn{1}{c|}{$d$}  & $\overline{\mbox{MSE}}(f)$ & $\min_i \mbox{MSE}(f_i)$ & $\mbox{MSE}_{\mbox{\tiny oracle}}$ \\\n');
%     for i=2:size(t,1)
%         dataset_name = t{i,1};
%         dataset_name=strrep(dataset_name,'_',' '); dataset_name(1) = upper(dataset_name(1));
% 
%         fprintf('%20s &\t %d &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) \\\\ \\hline\n',...
%         dataset_name, ...
%                      t{i,1+idx_orc},e{i,1+idx_orc}, ...
%                      t{i,1+idx_mean},e{i,1+idx_mean}, ...
%                      t{i,idx_med},e{i,idx_med}, ... 
%                      t{i,idx_upcr},e{i,1+2+is_RF}, ...
%                      t{i,1+5+is_RF},e{i,1+5+is_RF},t{i,end-2},e{i,end-2});
%     end;
% end;

%% PRINT EXCESS RISK IN SELECTING THE BEST MSE REGRESSOR
if filename_contains_index  % only on repeated experiments, where the std can be calculated
    t = pivottable(results_summary,4,2,3,@mean);
    e = pivottable(results_summary,4,2,3,@std);
    fprintf('Dataset \t\t& Oracle MSE \t& Best Regressor MSE & Excess Risk $\\min_i \\widehat{\\mbox{MSE}_i} $ \t& Excess Risk $\\max_i \\hat\\rho_i $ \n');
    for i=2:size(t,1)
        dataset_name = t{i,1};
        dataset_name=strrep(dataset_name,'_',' '); dataset_name(1) = upper(dataset_name(1));

        fprintf('%20s &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$)&\t %.2f ($\\pm %.2f$)  \\\\ \\hline\n',...
        dataset_name,t{i,1+idx_orc},e{i,1+idx_orc},... % Oracle MSE
                     t{i,1+idx_best},e{i,1+idx_best}, ... % Best Regressor MSE
                     t{i,1+idx_best_mse},e{i,1+idx_best_mse}, ... % Excess Risk MSE
                     t{i,1+idx_best_rho},e{i,1+idx_best_rho}); % Excess Risk RHO
    end;
    
end;

%% PRINT RESULTS
if filename_contains_index  % only on repeated experiments, where the std can be calculated
    t = pivottable(results_summary,4,2,3,@mean);
    e = pivottable(results_summary,4,2,3,@std);
    fprintf('Dataset \t\t& Oracle MSE \t& U-PCR MSE & Mean MSE \t& Median MSE \n');
    for i=2:size(t,1)
        dataset_name = t{i,1};
        dataset_name=strrep(dataset_name,'_',' '); dataset_name(1) = upper(dataset_name(1));

        fprintf('%20s &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$) &\t %.2f ($\\pm %.2f$)&\t %.2f ($\\pm %.2f$)  \\\\ \\hline\n',...
        dataset_name,t{i,1+idx_orc},e{i,1+idx_orc},... % Oracle MSE
                     t{i,1+idx_upcr2c_ss},e{i,1+idx_upcr2c_ss}, ... 
                     t{i,1+idx_mean},e{i,1+idx_mean}, ... 
                     t{i,1+idx_med},e{i,1+idx_med}); 
    end;
end;
