function V  = cluster_twoway(Y, X, C1, C2, u)
% Computes two-way clustered standard errors. 
% Matches results from Stata routineprovided by Petersen (2008).
% Extends routine by Ian Gow to large multidimensional datasets. 
% Michael Abrahams, 2014
%
%Model: Y=[1 X]B+U (observations in rows), with U_i~N(0,Sig).
% ---Input---
%Y: TxP matrix of outcome variables. 
%X: TxK matrix predictor variables. 
%U: 
% 
% ---Output--- 
% V: Cluster-robust estimate of variance of vec(B_hat)

if nargin < 4 
    error('Not enough inputs.');
end

[T, K]          = size(X);  
[~, P]          = size(Y);
upsInv          = inv(X'*X); %Upsilon inverse


if nargin < 6
    u           = (eye(T) - X*upsInv*X')*Y;
end


%%% Compute clustered standard errors
V_C1         = clustered_var(upsInv, X, u, C1);
V_C2         = clustered_var(upsInv, X, u, C2);
V_Cross      = clustered_var(upsInv, X, u, [C1 C2]);
V            = (V_C1 + V_C2 - V_Cross);

  
% Calculates clustered standard errors
    function V = clustered_var(upsInv, X, U, cl)

    
    % Identify unique clusters
    [~, ~, iC] = unique(cl, 'rows');  %iC: index of which group each element is in
    [cl_sort, IX] = sort(iC); %Find indices of elements in matching groups
    [~, a] = unique(cl_sort); %Find groups of observations
    M = size(a,1); %Number of clusters
    a = [a; T+1]; 

    
    w = waitbar(0, 'Calculating Clustered Standard Errors');
    V = [];    
    if M == size(X,1) %Full disaggregation - use White estimator
        for j = 1:P
            crossterm = X .* (U(:,j).^2*ones(1,size(X,2)));
            crossterm = crossterm' * X;
            V = blkdiag(V, upsInv * crossterm * upsInv);
        end 
    else
        X = X(IX,:);
        U = U(IX,:);
        
        
        %%% Loop over clusters to sum individual contribution to variance
        crossterms = zeros(P*K, M);
        for i=1:M
            ind = a(i):a(i+1)-1;
            e_g = X(ind,:)' * U(ind,:);
            crossterms(:,i) = vec(e_g);
            waitbar(i/M, w);
        end
        
        for j = 1:size(U,2)
            V = blkdiag(V, upsInv * ...
                crossterms((j-1)*K+1:j*K,:) * crossterms((j-1)*K+1:j*K,:)' * upsInv);
        end
        
    end
    close(w);
    V = (T-1)/(T-K)*M/(M-1)*V;
end


end
