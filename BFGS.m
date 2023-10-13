function [xstar,fstar,iter] = BFGS(f,n,x0)
%BFGS uses the Broyden–Fletcher–Goldfarb–Shanno unconstrained optimisation
%   method to find the local minimum of a function.
%
%   Parameters
%   ----------
%   f : symbolic function 
%       (for example, syms x; f = x^2)
%   n : int
%       size of symbolic x (or in the context of our project, 
%                           number of ads)
%   x0 : vector of size 1xn
%       initial point for descent search
%   
%   Returns
%   -------
%   xstar : vector of size 1xn
%       this is the minimiser
%   fstar : double
%       this is the value of f at the minimiser
%   iter : int
%       this is the number of iterations that the search algorithm took to
%       find the minimiser
    
    x = sym('x', [1, n]);
    syms t

    % BFGS Step 1
    H = eye(n); % initial BFGS H matrix
    tol = 0.01; % stopping tolerance
    gradf = gradient(f); % exact gradient of f using symbolic toolbox
    solfound = false;
    iter = 1; % starting iteration number

    xs = zeros(n,10); % keep track of x at each step of descent
    xs(:,iter) = x0; % set initial x value

    while true
        % BFGS Step 2
        initial_f_grad_value = double(subs(gradf,x,xs(:,iter).'));
        initial_abs_f_grad_value = norm(initial_f_grad_value);

        if initial_abs_f_grad_value < tol
           solfound = true;
           break; 
        end

        d = -H*initial_f_grad_value;
        
        % if d is NaN then something went wrong; return last x
        if isnan(d)
           solfound = false;
           break
        end
        
        % BFGS Step 3
        q = subs(f, x, (xs(:,iter) + t*d).');
        qfun = matlabFunction(q);
        [tval,fval,exitflag,output] = fminunc(qfun,0,optimoptions('fminunc','Display','none'));
        
        % BFGS Step 3
        xs(:,iter+1) = xs(:,iter) + tval*d;

        %%% BFGS Update
        s = xs(:,iter+1) - xs(:,iter);
        g = double(subs(gradf, x, xs(:,iter+1).')) - double(subs(gradf, x, xs(:,iter).'));
        r = (H*g) / (dot(s,g));
        H = H + ( (1 + dot(r,g))/(dot(s,g)) ) * (s*s.') - (s*r.' + r*s.');
        iter = iter+1;
    end
    
    % returns
    if solfound == true
        xstar = xs(:,iter);
        fstar = 1*double(subs(f,x,xs(:,iter).'));
    else
       xstar = xs(:,iter-1);
       fstar = 1*double(subs(f,x,xs(:,iter-1).'));
    end
end
