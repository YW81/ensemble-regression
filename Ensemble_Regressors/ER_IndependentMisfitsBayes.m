% function [y_ind_bayes, w_ind_bayes,rho_hat, MSE_hat,chosen_g2] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2, loss,showFig)
% Assume independent misfit errors (diagonal C*), recover rho, find optimal weights.
% Done by minimizing |Aa - Cshifted| where rho_i = g_2 + a_i, and Cshifted = C - g_2.
% use isInit for subset selection and problem difficulty classification. It does not use lambda_1/m
% for rho estimation, and therefor is more robust to outliers (is doesn't assume accurate f_i).
function [y_ind_bayes, w_ind_bayes,rho_hat, MSE_hat,g2_chosen,g2_list,MSE,SCORE] = ER_IndependentMisfitsBayes(y_true, Z, Ey, Ey2, loss,showFig,isInit)
    if ~exist('loss','var')  % loss can be 'l2','l1','huber'
        loss = 'l2';
    end;
    if ~exist('showFig','var')
        showFig = 0;
    end;
    if ~exist('isInit','var')
        isInit = false;
    end;

    [m,n] = size(Z);
    C = cov(Z');
    [v_1,lambda_1] = eigs(C,1,'lm');
    var_y = Ey2 - Ey.^2;
    mse = @(x) mean((y_true' - x).^2 / var_y);
    
    [y_INDB_g0, w_INDB_g0,rho_INDB_g0,a_vec_INDB_g0] = getINDBayesPredictions( Z, Ey, 0, 'l2'); % function below
    [y_INDB_g1, w_INDB_g1,rho_INDB_g1,a_vec_INDB_g1] = getINDBayesPredictions( Z, Ey, 0, 'l1'); % function below

    max_g = min(1,2 * (1 - max(a_vec_INDB_g0/var_y))); 
    g2_list = linspace(0,max_g * Ey2,80);
    a_vec = zeros(m,length(g2_list));
    rhoINDB = zeros(m,length(g2_list));
    yINDB = zeros(n,length(g2_list));
    wINDB = zeros(m,length(g2_list));
    MSE = zeros(length(g2_list),1);
    SCORE = zeros(length(g2_list),1);

    rhoINDB_1 = zeros(m,length(g2_list));
    yINDB_1   = zeros(n,length(g2_list));
    wINDB_1   = zeros(m,length(g2_list));
    MSE_1     = zeros(length(g2_list),1);
    SCORE_1   = zeros(length(g2_list),1);
    
    SCORE_2C = zeros(length(g2_list),1);
    SCORE_2C_1 = zeros(length(g2_list),1);

    for i=1:length(g2_list);
        rhoINDB(:,i) = rho_INDB_g0 + ones(m,1)*g2_list(i)/2;
        [yINDB(:,i), wINDB(:,i), ~, SCORE(i)] = ER_UPCRgivenRho(Z,Ey,Ey2,rhoINDB(:,i));
        MSE(i) = mse(yINDB(:,i));
        
        rhoINDB_1(:,i) = rho_INDB_g1 + ones(m,1)*g2_list(i)/2;
        [yINDB_1(:,i), wINDB_1(:,i), ~, SCORE_1(i)] = ER_UPCRgivenRho(Z,Ey,Ey2,rhoINDB_1(:,i));
        MSE_1(i) = mse(yINDB_1(:,i));
    
        [~, ~, ~, SCORE_2C(i)] = ER_UPCRgivenRho2Components(Z,Ey,Ey2,rhoINDB(:,i));
        [~, ~, ~, SCORE_2C_1(i)] = ER_UPCRgivenRho2Components(Z,Ey,Ey2,rhoINDB_1(:,i));
        
        %SCORE(i) = v_1'*rhoINDB(:,i); %mean(mean((abs(Z - repmat(yINDB(:,i)',m,1))))); % mean absolute error
    end;
    REL_SCORE  = SCORE'./sqrt(sum(rhoINDB.^2));
    REL_SCORE_1= SCORE_1'./sqrt(sum(rhoINDB_1.^2));
    REL_SCORE_2C  = SCORE_2C'./sqrt(sum(rhoINDB.^2));
    REL_SCORE_2C_1= SCORE_2C_1'./sqrt(sum(rhoINDB_1.^2));
    
    [val0, g0_opt_indx] = min(SCORE);
    [val1, g1_opt_indx] = min(SCORE_1);
    [rval0, g0_r_indx] = min(REL_SCORE);
    [rval1, g1_r_indx] = min(REL_SCORE_1);
    [rval0_2c, g0_r_indx_2c] = min(REL_SCORE_2C);
    [rval1_2c, g1_r_indx_2c] = min(REL_SCORE_2C_1);
    
    % Check which is better - L1 or L2
    isL2chosen = g2_list(g0_r_indx) >= g2_list(g1_r_indx); %isL2chosen = g2_list(g0_opt_indx) >= g2_list(g1_opt_indx);
    if isL2chosen
        g2_chosen_idx = g0_r_indx; %g0_opt_indx;
        g2_chosen = g2_list(g0_r_indx);% g0_opt_indx);
    else
        g2_chosen_idx = g1_r_indx;%    g1_opt_indx;
        g2_chosen = g2_list(g1_r_indx);% g1_opt_indx);
    end;
   
    if ~isInit && (g2_chosen < lambda_1/m) % only check for lambda_1/m if we're not during subset selection or classification
        fprintf(2,'Using g2_chosen = lambda_1/m\n');
        g2_chosen = min(lambda_1/m, max_g*Ey2);
        [y_ind_bayes, w_ind_bayes,rho_hat] = getINDBayesPredictions( Z, Ey, g2_chosen, 'l2'); % function below

    % Take the values corresponding to the better minimization scheme, at the optimal g2 of that
    % scheme OR at lambda_1/m if lambda_1/m > g2_hat
    elseif isL2chosen
        y_ind_bayes = yINDB(:,g2_chosen_idx);
        w_ind_bayes = wINDB(:,g2_chosen_idx);
        rho_hat = rhoINDB(:,g2_chosen_idx);
    else
        y_ind_bayes = yINDB_1(:,g2_chosen_idx);
        w_ind_bayes = wINDB_1(:,g2_chosen_idx);
        rho_hat = rhoINDB_1(:,g2_chosen_idx);
    end;

    MSE_hat = zeros(m,1); % we return MSE_hat, but nobody uses this
    for i=1:m
        MSE_hat(i) = (var_y - 2*rho_hat(i) + C(i,i)) / var_y;
    end;
    
    
 %% PLOTS
 
    if showFig
        %figure(301); clf; set(gca,'fontsize',24); 
        %plot(g2_list/var_y,min(

        clf; set(gca,'fontsize',24);
        plot(g2_list/var_y,MSE,'bs-'); hold on; 
        plot(g2_list/var_y,REL_SCORE/max(REL_SCORE),'r>-'); grid on;
        plot(g2_list/var_y,MSE_1,'bo-'); hold on; 
        plot(g2_list/var_y,REL_SCORE_1/max(REL_SCORE_1),'m.-'); grid on;
        title({['MSE(min) ' num2str(MSE(g0_opt_indx)) ' MSE_1(min) ' num2str(MSE_1(g1_opt_indx))]; ...
               [' MSE(REL) ' num2str(MSE(g0_r_indx))  ' MSE(REL_1) ' num2str(MSE(g1_r_indx))]}); 
        if 1 
            plot(g2_list/var_y,REL_SCORE_2C/max(REL_SCORE_2C),'k','linewidth',2);
            plot(g2_list/var_y,REL_SCORE_2C_1/max(REL_SCORE_2C_1),'k--','linewidth',2);
        end;
        legend('MSE','REL_SCORE','MSE-1','REL_SCORE_1','2C','2C_1'); xlabel('g_2');        
        plot(g2_list/var_y,min(rhoINDB)/var_y,'c--',g2_list/var_y,max(rhoINDB)/var_y,'g--');
        
        if 1
            a_hat = rho_hat - g2_chosen;
                
            plot([g2_chosen/var_y g2_chosen/var_y], [0 1],'r--',[min(lambda_1/m, max_g) min(lambda_1/m, max_g)],[0 1],'b-');
            disp(a_hat')
        end;

        %% FOR PAPER
%         fig = gcf; set(fig,'Name','G2 Estimation');
%         clf; set(gca,'fontsize',20);
%         plot(g2_list/var_y,MSE,'k-','linewidth',2); hold on; 
% 
%         if isL2chosen
%             plot(g2_list/var_y,REL_SCORE/max(REL_SCORE),'b-','linewidth',2); grid on;
%             plot([g2_list(g0_r_indx)/var_y g2_list(g0_r_indx)/var_y], [0 max(1,max(MSE))],'m--','linewidth',2);
%         else
%             plot(g2_list/var_y,REL_SCORE_1/max(REL_SCORE_1),'b-','linewidth',2); grid on;
%             plot([g2_list(g1_r_indx)/var_y g2_list(g1_r_indx)/var_y], [0 max(1,max(MSE))],'m--','linewidth',2);
%         end
%         
%         %if g2_list(g1_r_indx)/var_y > .1 && g2_list(g1_r_indx)/var_y < .8
%             plot([lambda_1/m/var_y lambda_1/m/var_y], [0 max(1,max(MSE))],'r-.','linewidth',2);
%         %end
%         
%         ylim([0 max(1,max(MSE))]);
%         legend('MSE','RESIDUAL','MIN RES','\lambda_1/m'); xlabel('q / Var(Y)');
%         
%         set(fig,'PaperPositionMode','auto');
%         set(fig,'Position',[500 700 300 200]);
%         set(gca,'Position', [.1 .2 .85 .70]);        
%         fprintf('PAUSE FOR PLOT\n'); pause;        
    end;
    
end

function [y_ind_bayes, w_ind_bayes,rho_hat,a] = getINDBayesPredictions(Z,Ey,g2,loss)
    [m,n] = size(Z);
    Z0 = Z - mean(Z,2)*ones(1,n); % Z_ij = f_i(x_j) - \mu_i
    C = cov(Z');
    
    %% Test (the following 2 lines generate C,diagonal C*, and rho that allow rho to be recovered):
    %m=5; Ey2 = 1;
    %v = 1+rand(m,1); rho_true = rand(m,1); C =  diag(v) + repmat(rho_true,1,m) + repmat(rho_true',m,1) - Ey2
    
    %% Solve linear equation  C_ij + Ey2 = rho_i + rho_j 
    % using the matrix A which selects the off diagonal elements of rho
    % such that A\rho = C + Ey2. A is m-choose-2 combinations of regressors (off-diagonal elements only)
    subs = nchoosek(1:m,2);
    idxs = sub2ind(size(C),subs(:,1),subs(:,2));
    Cshifted = C(idxs) - g2; %+Ey2
    A = zeros(size(subs,1), m);
    for i=1:size(subs,1)
        A(i,subs(i,1)) = 1;
        A(i,subs(i,2)) = 1;
    end;
    
    % Solve for rho: A*rho = C+Ey2
    if strcmp(loss,'huber')
        fval = @(a) sum(huber(Cshifted - A*a,1/(10*m))); % Huber-loss with delta = 1/10m
        a = fminunc(fval,zeros(m,1));
    elseif strcmp(loss,'l1')
        fval = @(a) norm(Cshifted - A*a,1);  % L1-loss
        options = optimoptions(@fminunc,'Algorithm','quasi-newton','Display','off');
        a = fminunc(fval,zeros(m,1),options);
    else
        %fval = @(a) norm(Cshifted - A*a);  % L2-loss <==> Least-Squares
        %a = A\Cshifted; % Least-Squares
        options = optimoptions(@lsqlin,'Algorithm','active-set','Display','off');
        [a, resnorm, residuals] = lsqlin(A,Cshifted,[],[],[],[],[],[],[],options);
        R=zeros(m); R(idxs)=residuals./g2; R=R+R'; %imagesc(R); colormap(hot); colorbar;title('unconstrained residuals'); 
        %fprintf('unconstraind norm(residuals) = %.2f',norm(R));
    end;

    % Calculate weights based on the assumption of independent misfit errors
    Cinv = pinv(C,1e-5);
    rho = g2 + a;
    %w_ind_bayes = Cinv*(rho - ones(m,1)*(ones(1,m)*(Cinv*rho)-1)/sum(sum(Cinv))); % constrain sum(w)=1
    w_ind_bayes = Cinv*rho;
    %%
    y_ind_bayes = Ey + Z0' * w_ind_bayes;    
    rho_hat = rho;
end