function h = modap(x)
%modap approximates the absolute value of x, |x|.
%
%   Parameters
%   ----------
%   x : int/double or symbol
%   
%   Returns
%   -------
%   h : double (if x is double); symbolic function (if x is symbolic)

    s = 0.0001; % set approximation parameter s (explained in report)
    h = x.*erf(x./s); % use error function to approximate |x|
end