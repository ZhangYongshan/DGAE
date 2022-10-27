function [A_n,B_n] = construct_A_B(fea,nSmp,nFea,k)
%function [A_n,B_n] = construct_A_B(fea,nSmp,nFea£¬k)
%   X n*d
%   A n*n
%   B d*d
% construct A
    if ~exist('k','var')                     
        k=5;
    end
       options = [];
       options.NeighborMode = 'KNN';
       options.k = k;
%       options.WeightMode = 'HeatKernel';
%       options.t = 1;
%S1 = constructW(fea,options);
S1 = constructW(fea);
A_bar = S1 + speye(nSmp);
d = sum(A_bar);
d_sqrt = 1.0./sqrt(d);
d_sqrt(d_sqrt == Inf) = 0;
DH = diag(d_sqrt);
DH = sparse(DH);
A_n = DH * sparse(A_bar) * DH;
% construct B

S2 = constructW(fea',options);
%S2 = constructW(fea');
B_bar = S2 + speye(nFea);
d2 = sum(B_bar);
d_sqrt2 = 1.0./sqrt(d2);
d_sqrt2(d_sqrt2 == Inf) = 0;
DH2 = diag(d_sqrt2);
DH2 = sparse(DH2);
B_n = DH2 * sparse(B_bar) * DH2;
end

