function m = maxap(x,a)
%maxap uses approximation of the l1 norm in modap to approximate the 
%   max(x,b) function
%
%   Parameters
%   ----------
%   x : int/double or symbol
%   a : int/double
%   
%   Returns
%   -------
%   m : double (if x is double); symbolic function (if x is symbolic)
    m = (x + a + modap(x - a))./2;
end