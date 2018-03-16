function [ varargout ] = KP_RankTest( Pi, V, T, q, F, G )
%Implementation of the Kleibergen-Paap Rank Test. 
%Description of test can be found in Journal of Econometrics 133 p97-126.
%Tests the hypothesis that the stochastic KxM matrix Pi has rank q < min(K,M).
%
%Input arguments: Pi, matrix to test.  
%V:  sqrt(T) * vec(\hat{Pi} - Pi) -> N(0, V)  (Asymptotic covariance)
%T: number of observations used in estimating Pi.
%q: Rank to test against.  
%F, G: KxK and MxM scaling matrices.  If not given, assumed identity.
%Try to choose F and G s.t. kron(F, G) * V * kron(F, G)' -> eye(k*m).
%e.g. if Pi is regression coefficient Y = Z * Pi + e, 
%set F = chol(Z'Z) and G = inv(chol(Y'Y)).
%
%Michael Abrahams, 2011

[k, m] = size(Pi);

if q >= min(k, m)
    error('Cannot test against full rank.')
end

if nargin < 5
    F = eye(k);
    G = eye(m);
end

%Rescaled estimate and variance
Theta = G * Pi * F';
W = kron(F, G) * V * kron(F, G)';



%SVD calculation method
[u, ~, v] = svd(Theta);
v = v';

u22 = u(q+1:end, q+1:end);
A_Perp = u(:,q+1:end) / u22 * sqrtm(u22 * u22');

v22 = v(q+1:end, q+1:end);
B_Perp = sqrtm(v22' * v22) / v22 * v(q+1:end,:);


%Transformed Singular Values and Variance
lambda_hat = vec(A_Perp' * Theta * B_Perp');
Omega = kron(B_Perp, A_Perp') * W * kron(B_Perp, A_Perp')';

%Test statistic
rk = T * lambda_hat' / Omega * lambda_hat;
p = 1 - chi2cdf(rk, (k-q)*(m-q));

switch nargout
    case 0
        disp(sprintf('Kleibergen-Paap Statistic for rank = %i:\n\trk=%f, Chi-sq(%i) p=%g.', q, rk, (k-q)*(m-q), p));
    case 1
        varargout = {rk};
    case 2
        varargout = {rk (k-q)*(m-q)};
    case 3
        varargout = {rk (k-q)*(m-q) p};
end
        
        

end

