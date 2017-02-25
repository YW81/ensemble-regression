clear all; close all;
addpath Ensemble_Regressors/

% [im_id, cls_idx, pred_box[:4], gt_box[:4],  probability]
KEY_IDXS = [1:2,7:10]; % uniqueness is defined by im_id and gtybox
ROOT = './Datasets/RealWorld/Moshik/mat/';
files = dir([ROOT '*.mat']);
%files = struct(struct('name','friedman3.mat'));

%% LOAD ALLDATA

% First, find the size of matrix that needs to be allocated
n = 0; num_of_fields_list = zeros(length(files),1);
for file_idx=1:length(files)
    load([ROOT files(file_idx).name]);
    fprintf('SIZE: (%d,%d), FILE: %s\n', size(data), files(file_idx).name);
    [n_cur, num_of_fields_list(file_idx)] = size(data);
    if n_cur > n % found a longer data file
        n = n_cur;
    end
end;

num_of_fields = min(num_of_fields_list);
assert(num_of_fields == 11);
ALLDATA = nan*ones(length(files), n, num_of_fields);
models = cell(length(files),1);

% Populate ALLDATA matrix, dedup the entries in each file
for file_idx=1:length(files)
    load([ROOT files(file_idx).name]);
    fprintf('FILE: %s\n', files(file_idx).name);
    models{file_idx} = files(file_idx).name;
    
    keys = data(:,KEY_IDXS);
    [~,uniq_idx]=unique(keys,'rows'); % U = keys(uniq_keys,:)
    ALLDATA(file_idx,1:length(uniq_idx),:) = data(uniq_idx,:); % dedup (and sort, since uniq_idx is sorted)
end;

% Create a filtered data matrix, only taking the rows that show up in the shortest data file
[n loc] = min(sum(~isnan(ALLDATA(:,:,1)),2)); % find the minimal data file
ALLDATA_FILTERED = nan*ones(length(files),n,num_of_fields);
keys = squeeze(ALLDATA(loc,1:n,KEY_IDXS)); % get its keys
for file_idx=1:length(files)
    [val loc] = ismember(keys, squeeze(ALLDATA(file_idx,:,KEY_IDXS)),'rows');
    assert(sum(val) == n); % make sure all keys exist in this data file
    ALLDATA_FILTERED(file_idx,:,:) = ALLDATA(file_idx,loc(val),:); % take only ismember rows
end
ALLDATA = ALLDATA_FILTERED;
clear ALLDATA_FILTERED;

%% VERIFY SORTING OF ROWS MATCHES BETWEEN DATA FILES
keys_match = true;
for file_idx=2:length(files); 
    keys_match = keys_match & isequal(ALLDATA(1,:,KEY_IDXS),ALLDATA(file_idx,:,KEY_IDXS)); 
    assert(keys_match);
end; 

%% CREATE DATA MATRIX Z, AND GROUND TRUTH Y_TRUE
class_id = ALLDATA(:,:,2);
assert(size(unique(class_id,'rows'),1) == 1);
class_id = class_id(1,:)';

% X1
Z_x1 = squeeze(ALLDATA(:,:,3));
yy_true_x1 = squeeze(ALLDATA(:,:,7));
assert(size(unique(yy_true_x1,'rows'),1) == 1) % make sure that all rows exactly match
yy_true_x1 = yy_true_x1(1,:);

% Y1
Z_y1 = squeeze(ALLDATA(:,:,4));
yy_true_y1 = squeeze(ALLDATA(:,:,8));
assert(size(unique(yy_true_y1,'rows'),1) == 1) % make sure that all rows exactly match
yy_true_y1 = yy_true_y1(1,:);

% X2
Z_x2 = squeeze(ALLDATA(:,:, 5));
yy_true_x2 = squeeze(ALLDATA(:,:, 9));
assert(size(unique(yy_true_x2,'rows'),1) == 1) % make sure that all rows exactly match
yy_true_x2 = yy_true_x2(1,:);

% Y2
Z_y2 = squeeze(ALLDATA(:,:, 6));
yy_true_y2 = squeeze(ALLDATA(:,:, 10));
assert(size(unique(yy_true_y2,'rows'),1) == 1) % make sure that all rows exactly match
yy_true_y2 = yy_true_y2(1,:);

%% SAVE DATA MAT FILE
%save('box_regression_coco4.mat','Z*','y*','class_id','models');
save('box_regression_coco4_alldata.mat','Z*','y*','class_id','models','ALLDATA');
