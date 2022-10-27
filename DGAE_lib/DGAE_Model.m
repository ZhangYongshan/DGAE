function [loss_history, embedding, NetWork] = DGAE_Model(self)
%DGAE_MODEL 
num_epoch = self.num_epoch;
A = self.A;
B = self.B;
X = self.X;
M = A*X*B;
self.M = M;
 
loss_history = zeros(num_epoch, 1);
num_input = size(X, 2);

numsap= size(self.X, 1);
num_embed = self.num_embed;
learning_rate = self.learning_rate;

%% weight initialization: glorot glorot initialization
W1 = (2 * rand(num_input, num_embed) - 1) * sqrt(6 / (num_input + num_embed));
W2 = (2 * rand(num_embed, num_input) - 1) * sqrt(6 / (num_embed + num_input));
B1 = (2 * rand(numsap, num_embed) - 1) * sqrt(6 / (numsap + num_input));
B2 = (2 * rand(numsap, num_input) - 1) * sqrt(6 / (numsap + num_input));
%% Initialize ADAM 
adma_para = adam_init_all(size(W1),size(W2),size(B1),size(B2),learning_rate);

%% Network Training
fprintf('---DGAE network is being trained...\n');
for epoch = 1:num_epoch
    % fprintf('optimization epoch %g\n', epoch);
    [L,r_Xo,r_Xi,Zo,Zi] = DGAE_Network(self,W1,W2,B1,B2);
    [grad_W1,grad_W2,grad_B1,grad_B2] = DGAE_Gradient(self,r_Xo,r_Xi,Zo,Zi,W1,W2);
    [W1, W2,B1, B2, adma_para] = DGAE_Optimize_all(W1,grad_W1,W2,grad_W2,B1,grad_B1,B2,grad_B2,adma_para);
    loss_history(epoch, 1) = L;
end
 
embedding = Zo;
NetWork = cell(2, 1);
NetWork{1}.W=W1;
NetWork{1}.bias_upW =B1;
NetWork{2}.W=W2;
NetWork{2}.bias_upW =B2;
