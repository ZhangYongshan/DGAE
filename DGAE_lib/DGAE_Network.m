function [L,r_Xo,r_Xi,Zo,Zi] = DGAE_Network(self,W1,W2,B1,B2)
% Forward Propagation of Network
X = self.X;
M=self.M;
%% Encoder
    Zi=M*W1+B1; 
    Zo= Zi.*(Zi>=0);  %ReLu function;
%% Decoder
    r_Xi=Zo*W2+B2;
    r_Xo=1 ./ (1 + exp(-r_Xi));  % sigmoid function 
%% loss function:
    L =mean((X(:) - r_Xo(:)) .^ 2);%+lamda_test*(mean(W1(:) .^ 2)+mean(W2(:) .^ 2));
