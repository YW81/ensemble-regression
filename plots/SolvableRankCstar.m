clear all; 
max_m = 20;
eqs = zeros(max_m,1); max_rank = zeros(max_m,1); 
for m=2:max_m; 
    eqs(m) = nchoosek(m,2); 
    max_rank(m) = floor((eqs(m) - m)/m); 
end;
plot(3:max_m,max_rank(3:end),'s'); 
grid on; axis tight; 
xlabel('number of regressors m'); ylabel('max rank C*');

hold on; plot(1:max_m, [1:max_m]/2,'k--');
legend('Max rank C*','y=m/2','Location','SouthEast');
