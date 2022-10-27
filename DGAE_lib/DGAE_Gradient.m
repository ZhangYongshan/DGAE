function [grad_W1, grad_W2,grad_B1,grad_B2] = DGAE_Gradient(self, r_Xo,r_Xi, Zo ,Zi , W1, W2)
% Back Propagation 
X = self.X;
lamda = self.lamda;
Xo=self.M;
%% dL/dXi
    error_o=2*(r_Xo-X).*(1 ./ (1 + exp(-r_Xi))).*(1-1 ./ (1 + exp(-r_Xi)));
%% backward for output layer::compute gradient W2 & gradient B2 (dL/dW2 & dL/dB2)
    grad_W2=Zo'*error_o+2*lamda*W2;
    grad_B2=error_o;
%% dL/dZi
    error_h=error_o*W2'.*(Zi >= 0);
%% backward for hidden layer:compute gradient W1 & gradient B1 (dL/dW1 & dL/dB1)
    grad_W1=Xo'*error_h+2*lamda*W1;
    grad_B1=error_h;