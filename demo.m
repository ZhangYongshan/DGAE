% =========================================================================
% A simple demo for Spectral-Spatial Feature Extraction with Dual Graph Autoencoder 
% for Hyperspectral Image Clustering
%
% version 1:2022.08.08
%
% Reference: Y. Zhang, Y. Wang, X. Chen, X. Jiang and Y. Zhou,
%"Spectral-Spatial Feature Extraction with Dual Graph Autoencoder 
% for Hyperspectral Image Clustering," in IEEE Transactions on Circuits 
% and Systems for Video Technology, 2022, doi: 10.1109/TCSVT.2022.3196679.
%=========================================================================
clc;clear;close all

addpath(genpath('common'));
addpath(genpath('Data'));
addpath(genpath('DGAE_lib'));
%addpath(genpath('drtoolbox'));
addpath(genpath('Entropy Rate Superpixel Segmentation'));
%addpath(genpath(cd));
%RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
%% load data
data_name = 'Indian';

load stream;  %
if strcmp(data_name,'Indian')
    load indian_pines_corrected;load indian_pines_gt
    data3D = indian_pines_corrected;        label_gt = indian_pines_gt;
    lamda=1e0;    % the penalty factor β for regularization term
    num_Pixel=20; % the number of superpixels S for the superpixel-based similarity graph
    num_embed=70; % the reduced dimension
end
%% Hyperparameters
nKmeans = 10;          
learning_rate = 1e-4;   % The alpha parameter in the ADAM optimizer
num_epoch=200;          % Network iteration parameters

labels_ERS = cubseg(data3D,num_Pixel);   % super-pixels segmentation
[X, Y,labels_cubseg] = Labeled_data_cubseg(data3D, label_gt,labels_ERS);

[m, n]=size(X);
nClass = length(unique(Y));

%print the setup information
disp(['Dataset: ',data_name]);
disp(['class_num=',num2str(nClass),',','num_kmeans=',num2str(nKmeans)]);

% Construction of dual graph
[~,B] =construct_A_B(X,m,n);
A=cubseg_Gen_adj_2D(X,labels_cubseg);
A=normalizeSparseA(A);
disp('The dual graph was constructed successfully');
%% Data ingress
self.A = A;
self.B = B;
self.X = X;
self.Y = Y;

self.num_epoch = num_epoch;
self.learning_rate = learning_rate;   % for adma optimization (value for alpha)

self.lamda = lamda;
self.num_embed = num_embed;

RandStream.setGlobalStream(stream);
%method='DGAE';

[loss_history, embedding,Network] = DGAE_Model(self);
%% Clustering Analysis
%Evaluation_sturct=["ACC", "MIhat", "ARI", "F1_score", "Precision", "Recall", "Purity"];
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
result= zeros(nKmeans,7);
for i = 1:nKmeans
    %fprintf('nKmeans: %d\n',i);
    label = kmeans(embedding,nClass,'Replicates',1);
    [result(i,:),~]= ClusteringMeasure(Y,label);
end
mean_results=mean(result);

fprintf('=============================================================\n');
fprintf(['The average ACC (10 iterations) of DGAE for ',data_name,' is %0.4f\n'],mean_results(1));
fprintf('=============================================================\n');
