function [varargout] = ols(Y, X, includeMean, robust)
%Usage:
%   [B_hat, U_hat, Sig, R2, Yhat, varBHat] = ols(Y, X)
% 
%Simple, no-nonsense OLS.  
%Model: Y=[1 X]B+U (observations in rows), with U_i~N(0,Sig).
%
%If third argument is false do not append a constant regressor.
%If fourth argument is true use White (1980) standard errors. 
%If Y contains more than one column, last output is var(vec(B_hat)))
%
%
%Michael Abrahams, 2011



    if (nargin < 3)
        includeMean = true;
    end
    if nargin < 4
        robust = false;
    end
    
    if includeMean 
        X = [ones(size(X,1), 1) X];
    end
    
    b = (X' * X) \ X' * Y; 
    u = Y - X*b;    
    sig = u' * u / (length(Y) - size(X, 2));
    

    
    switch nargout
        case 1
            varargout = {b};
        case 2
            varargout = {b, u};
        case 3
            varargout = {b, u, sig};
        case 4            
            SST = sum(demean(Y).^2);
            SSE = sum(u.^2);
            R2 = 1 - SSE./SST;
            varargout = {b, u, sig, R2};
        case 5
            SST = sum(demean(Y).^2);
            SSE = sum(u.^2);
            R2 = 1 - SSE./SST;
            Yhat = X * b;
            varargout = {b, u, sig, R2, Yhat};
        case 6
            SST = sum(demean(Y).^2);
            SSE = sum(u.^2);
            R2 = 1 - SSE./SST;
            Yhat = X * b;
            upsInv = inv(X'*X);
            if robust %Does not support multiple LHS
                varBHat = upsInv * (X' * (X .* repmat(u.^2, 1,size(X, 2)))) * upsInv;
            else
                varBHat = kron(sig, upsInv);
            end
            varargout = {b, u, sig, R2, Yhat, varBHat};
        otherwise
            varargout = {b};
            
    end
    
end