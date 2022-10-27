function [S] = cubseg_Gen_adj_2D(data2D,labels)
% [S] = cubseg_Gen_adj_2D(data2D,labels)
% Build a weight matrix from the constructed superpixel segmentation map
data_col=data2D;
[m,~]=size(data_col);

gt_col=labels;
gt_cla=unique(gt_col);
gt_num = length(gt_cla);

dist_seg=[];

i_seg=[];
j_seg=[];

fprintf('---The weight matrix is being constructed...\n');
for i=1:gt_num
    %disp(['---Processing split area: ',num2str(i)]);
    [v]=find(gt_col==gt_cla(i)); 
    ci = length(v);   
    if ci==1
        continue
    end
    datai = data_col(v,:); 
    dist_seg_temp=[];
    for j=1:ci
        temp=datai(j,:);
        % Calculate Euclidean distance
        [dist_temp]=temp.^2*ones(size(datai'))+ones(size(temp))*(datai').^2-2*temp*datai';
        dist_temp(dist_temp<0)=0;
        dist_temp(j)=0;
        dist_temp=sqrt(dist_temp);
        dist_seg_temp=[dist_seg_temp,dist_temp];
        i_seg_tmp=[];
        i_seg_tmp(1:ci)=v(j);
        i_seg=[i_seg i_seg_tmp];
        j_seg=[j_seg v'];
    end
    temp=[];
    if ci>1
        dist_a=var(dist_seg_temp); %Calculate the variance of the current region
        temp=exp(-(dist_seg_temp.^2)./(2*dist_a^2));
    else
        temp=dist_seg_temp;
    end
    dist_seg=[dist_seg,temp];
end
S=sparse(i_seg,j_seg,dist_seg,m,m);
% S = max(S, S');
    if  numel(find(isnan(S)))== 0 && sum((S==S')==0,'all')==0
        disp('The weight matrix was constructed successfully');
    else
        disp('There is a problem with the weight matrix construction');
    end
end
