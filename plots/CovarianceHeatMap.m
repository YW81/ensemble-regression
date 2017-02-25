clear all; close all; close all hidden;
C1 = [1 .98 .97 ; .98 1 .96;  .97 .96 1]
C2 = [1 .99; .99 1]
C3 = .9*ones(2,3);
C = [C1 C3'; C3 C2]
Ci = inv(C)
sum(Ci,2) ./ sum(sum(Ci))
Cbad = blkdiag(C,20,1); Cbad(Cbad == 0) = 0.7; Cbadi = inv(Cbad);
Cbad,
HeatMap(Cbad(end:-1:1,:),'Annotate',true,'Symmetric',false,'ColorMap',redbluecmap,'DisplayRange',2);
sum(Cbadi,2) ./ sum(sum(Cbadi))
