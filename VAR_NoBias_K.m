function varargout = VAR_NoBias_K( X, includeMean, reps )
% [Mu, Phi, Nu, Sig] = Var_NoBias_K(X, includeMean, reps)
%
%Implements the Nicholls and Pope (1988) bias correction using
%the stationarity restriction proposed by Kilian (1998).
%See Engsted and Pedersen (2011).
%X and Nu are in row-major form (i.e. TxK) so that the regression is
%X_(t+1)' = Phi * X_t' + Nu_(t+1)';
%
%Input arguments: X: TxK matrix of data.
%   includeMean: logical.  Estimate mu?
%   reps: integer.  Number of repetitions for NP correction.
%Output in order of preference: {Mu, Phi, Nu, Sigma}.
%
%Michael Abrahams, 2011


    if nargin < 3
        reps = 0;
    end
    if nargin == 1
        includeMean = true;
    end
    if nargin == 0
        error('Not enough input arguments');
    end;

    %Init
    K = size(X, 2);
    T = size(X, 1);
    sigmaX = cov(X);
    I = eye(K);
    
    if (includeMean); XT = [ones(T-1, 1), X(1:T-1,:)];
    else XT = X(1:T-1,:);
    end

    

    %Calculate OLS estimator    
    Phi = (XT' * XT) \ XT' * X(2:end,:);    
    if (includeMean)
        Mu = Phi(1,:)';
        Phi = Phi(2:end,:);        
    else 
        Mu = zeros(K, 1);
    end
    
        
    %Innovations
    Nu = X(2:end, :) - X(1:T-1, :) * Phi - repmat(Mu, 1, T-1)';
    sigmaNu = cov(Nu);
    
    %This algorithm is written for a LHS VAR
%     Phi = Phi';    
    if (reps > 0)
        for n = 1:reps
            %Calculate bias
            B = inv(I - Phi) + Phi / (I - (Phi)^2);
            spec = eig(Phi);
            for i = 1:K
                B = B + (I * spec(i)) / inv(I - spec(i)*Phi);
            end        
            B = sigmaNu * B / sigmaX;

            
            %Kilian's method to ensure stationarity
            if max(abs(eig(Phi))) < 1;
                k = 1;
                m_eig = max(abs(eig(Phi + B'/T * k)));
                while m_eig >= 1;
                    k = k - .001;
                    m_eig = max(abs(eig(Phi + B'/T * k)));
                end                
                Phi = Phi + B'/T * k;
                disp(k);
            end
                    
            

            Nu = X(2:end, :) - X(1:T-1, :) * Phi - repmat(Mu, 1, T-1)';
            sigmaNu = cov(Nu);
        end
    end
        
    %Covariance
    Sig = Nu' * Nu / (T-1);
    
    Phi = Phi';   
    
    
    switch nargout
        case 0
            varargout = {Mu, Phi};
        case 1
            varargout = {[Mu Phi]};
        case 2
            varargout = {Mu, Phi};
        case 3
            varargout = {Mu, Phi, Nu};
        case 4
            varargout = {Mu, Phi, Nu, Sig};
    end
    
end

