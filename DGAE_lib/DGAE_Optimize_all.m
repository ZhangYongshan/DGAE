function [W1,W2,B1,B2,adma_para] = DGAE_Optimize_all(W1,grad_W1,W2,grad_W2,B1,grad_B1,B2,grad_B2, adma_para)
%DGAE_Optimize_all
% One step of ADAM

alpha = adma_para.alpha;
beta1 = adma_para.beta1;
beta2 = adma_para.beta2;
epsilon = adma_para.epsilon;
t = adma_para.t;
m = adma_para.m;
v = adma_para.v;

% fprintf('t = %g\n', t);
% fprintf('mean_m = %g\n', mean(m(:)));
% fprintf('mean_v = %g\n', mean(v(:)));

szW1 = size(W1);
szW2 = size(W2);
szB1 = size(B1);
szB2 = size(B2);
w = [W1(:); W2(:); B1(:); B2(:)];   %theta_t
g = [grad_W1(:); grad_W2(:); grad_B1(:); grad_B2(:)]; % g_t

%% Paper version
%
% t = t + 1;
% m = beta1 * m + (1-beta1) * g;
% v = beta2 * v + (1-beta2) * (g.^2);
% m_hat = m / (1 - beta1^t);
% v_hat = v / (1 - beta2^t);
% w = w - alpha * m_hat ./ (sqrt(v_hat) + epsilon);

%% Tensorflow version
t = t + 1;
lr = alpha * sqrt(1 - beta2^t) / (1 - beta1^t);
m = beta1 * m + (1-beta1) * g;
v = beta2 * v + (1-beta2) * (g.^2);
w = w - lr * m ./ (sqrt(v) + epsilon);

adma_para.alpha = alpha;
adma_para.beta1 = beta1;
adma_para.beta2 = beta2;
adma_para.epsilon = epsilon;
adma_para.t = t;
adma_para.m = m;
adma_para.v = v;

% fprintf('lr = %g\n', lr);
% fprintf('t = %g\n', t);
% fprintf('mean_m = %g\n', mean(m(:)));
% fprintf('mean_v = %g\n', mean(v(:)));
% fprintf('mean_w = %g\n', mean(w(:)));

W1 = reshape( w(1:prod(szW1)), szW1 );
W2 = reshape( w(1+prod(szW1):prod(szW1)+prod(szW2)), szW2 );
B1 = reshape( w(1+prod(szW1)+prod(szW2):prod(szW1)+prod(szW2)+prod(szB1)), szB1 );
B2 = reshape( w(1+prod(szW1)+prod(szW2)+prod(szB1):end), szB2 );