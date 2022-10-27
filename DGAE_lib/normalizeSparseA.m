function A = normalizeSparseA(A)
% Normalize sparse A.(renormalization)

n = size(A,1);
A = A + speye(n);           % A~=A+In
sqrt_diagD = sqrt(full(sum(A,2)));      % D~(-1/2)

[ii, jj, vv] = find(A);
vv = vv ./ (sqrt_diagD(ii) .* sqrt_diagD(jj));  % A^=D~(-1/2)A~D~(-1/2)

A = sparse(ii, jj, vv, n, n);
