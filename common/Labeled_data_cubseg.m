function [data, label,label_cub] = Labeled_data_cubseg(data3D, label_gt,label_cubseg)
% Data Preprocessing
[m, n, p] = size(data3D);
data_col = reshape(data3D,m*n,p);
label_col = reshape(label_gt,m*n,1);
label_cub_col=reshape(label_cubseg,m*n,1);

idx = find(label_col~=0);

data = data_col(idx,:);
data = normalize(data);

label = label_col(idx,:);
label_cub=label_cub_col(idx,:);
end
