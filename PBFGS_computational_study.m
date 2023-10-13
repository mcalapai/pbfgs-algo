n_param_sets = 5; % number of randomly generated parameter sets
n_runs = 5; % number of randomly generated initial points for each parameter set
n = 2; % same as m
m = n; % number of ads to be displayed over time T
T = 10; % time T
k = 1; % constant k > 0
x = sym('x', [1, m]); % symbolic x for PBFGS and MATLAB fmin
tol = 0.1; % tolerance for classifying a found minimiser as existing vs new
disp_comparison = true; % display setting:
% if true then will display the algorithm comparison (section 5.1)
% else, display scaling speed test results (section 5.2)

%rng(43); % seed for controlled experimentation (comment out)

% initialise random point matrices
as = zeros(n_param_sets,n); 
bs = zeros(n_param_sets,n);
cs = zeros(n_param_sets,n);
x0s = zeros(n_runs,n);

% populate random point matrices
for i = 1:n_param_sets
    as(i,:) = unifrnd(1,10,1,m);
    bs(i,:) = unifrnd(1,10,1,m);
    cs(i,:) = unifrnd(0,1,1,m);
end
for i = 1:n_runs
    x0s(i,:) = unifrnd(0,5,1,m);
end

disp("Testing values m = " + m + ", k = " + k + " and T = " + T + " for " + n_param_sets + ...
    " parameter sets with " + n_runs + " random initial points each set.");
% array for recording times when disp_comparison = false 
timer_array = zeros(n_param_sets*n_runs, 5);
for z = 1:5
    current_iteration = 1; % record total iterations
    for i = 1:n_param_sets % run over each parameter set
        a = as(i,:);
        b = bs(i,:);
        c = cs(i,:);
        n_min = 0; % number of minimums found for particular parameter set and x0
        solutions = zeros(10, m+4); % xstar (takes m values), fmin, n_min, n_iter, tElapsed
        for j = 1:n_runs % test several x0 values for given set of parameters
            tStart = tic; % start recording time
            x0 = x0s(j,:); 
            switch z
                case 1
                    % penalty function custom BFGS
                    [xstar, fstar, n_iter] = BFGS(objfun(x,n,a,b,c,k,T),n,x0);
                    xstar = xstar.';
                    fstar = conobjfun(xstar,a,b,k);
                case 2
                    % penalty function MATLAB fminunc
                    [xstar,fstar,exitflag,output] = fminunc(@(x)sobjfun(x,m,a,b,c,k,T),x0,optimoptions('fminunc','Display','none'));
                    n_iter = output.iterations;
                    fstar = conobjfun(xstar,a,b,k);
                case 3
                    % fmincon-activeset with ineq constraints in 2.2.1
                    A = [-1*eye(n);ones(1,n);-1*eye(n)];
                    B = [-1.*c, T, zeros(1,n)];
                    options = optimoptions(@fmincon,'Algorithm','active-set','Display','none');
                    [xstar,fstar,exitflag,output] = fmincon(@(x)conobjfun(x,a,b,k),x0,A,B,[],[],[],[],[],options);
                    n_iter = output.iterations;
                case 4
                    % fmincon-sqp with ineq constraints in 2.2.1
                    A = [-1*eye(n);ones(1,n);-1*eye(n)];
                    B = [-1.*c, T, zeros(1,n)];
                    options = optimoptions(@fmincon,'Algorithm','sqp','Display','none');
                    [xstar,fstar,exitflag,output] = fmincon(@(x)conobjfun(x,a,b,k),x0,A,B,[],[],[],[],[],options);
                    n_iter = output.iterations;
                case 5
                    % fmincon-interiorpoint with ineq constraints in 2.2.1
                    A = [-1*eye(n);ones(1,n);-1*eye(n)];
                    B = [-1.*c, T, zeros(1,n)];
                    options = optimoptions(@fmincon,'Algorithm','interior-point','Display','none');
                    [xstar,fstar,exitflag,output] = fmincon(@(x)conobjfun(x,a,b,k),x0,A,B,[],[],[],[],[],options);
                    n_iter = output.iterations;
            end
            tElapsed = toc(tStart); % finish recording time
            timer_array(current_iteration, z) = tElapsed; % append time to timer_array
            fstar = -fstar; % take negative to get maximum
            
            % handle for new/old min finding
            oldmin = false;
            for p = 1:n_min % iterate over each min already found
                if abs(fstar - solutions(p,m+1)) < tol % if new min is within tolerance of found min
                    % then append to solutions found for that min
                    solutions(p,m+2) = solutions(p,m+2) + 1; 
                    solutions(p,m+3) = solutions(p,m+3) + k;
                    solutions(p,m+4) = solutions(p,m+4) + tElapsed;
                    oldmin = true; % found min already exists
                    break;
                end
            end
            if oldmin == false % if a new min is found
                n_min = n_min + 1; % increase number of minds
                solutions(n_min,:) = [xstar, [fstar, 1, n_iter, tElapsed]]; % record new min
            end 
            current_iteration = current_iteration + 1; % increase iteration
        end
        if disp_comparison
            switch z
                case 1
                    % penalty function BFGS
                    fprintf("PBGFS \n")
                    fprintf("parameters: \na = [%s] \nb = [%s] \nc = [%s]\n", ...
                        sprintf('%f, ', a), sprintf('%f, ', b), sprintf('%f, ', c));
                case 2
                    % penalty function MATLAB fminunc
                    fprintf("fminunc \n")
                    fprintf("parameters: \na = [%s] \nb = [%s] \nc = [%s]\n", ...
                        sprintf('%f, ', a), sprintf('%f, ', b), sprintf('%f, ', c));
                case 3
                    % penalty function MATLAB fminunc
                    fprintf("fmincon-activeset \n")
                    fprintf("parameters: \na = [%s] \nb = [%s] \nc = [%s]\n", ...
                        sprintf('%f, ', a), sprintf('%f, ', b), sprintf('%f, ', c));
                case 4
                    % penalty function MATLAB fminunc
                    fprintf("fmincon-sqp \n")
                    fprintf("parameters: \na = [%s] \nb = [%s] \nc = [%s]\n", ...
                        sprintf('%f, ', a), sprintf('%f, ', b), sprintf('%f, ', c));
                case 5
                    % penalty function MATLAB fminunc
                    fprintf("fmincon-interiorpoint\n")
                    fprintf("parameters: \na = [%s] \nb = [%s] \nc = [%s]\n", ...
                        sprintf('%f, ', a), sprintf('%f, ', b), sprintf('%f, ', c));
                    
            end
            for u = 1:n_min
                avg_iter = solutions(u, m + 3)/solutions(u,m+2);
                avg_time = solutions(u, m + 4)/solutions(u,m+2);
                to_display = sprintf('%f, ', solutions(u,1:m));
                fprintf("F max %f at x = [%s]; \nfound for %i times, avg iterations of %f and avg time of %f \n", ...
                    solutions(u,m+1), to_display, solutions(u, m + 2), avg_iter, avg_time)
            end
            disp("---------------------------------------------------------------------------------------------------")
        end
    end
end

algorithm_times = mean(timer_array,1); % average tElapsed for each algorithm over all parameter sets AND x0

if ~disp_comparison % display for speed test
    for i = 1:5
        switch i
            case 1
                disp("PBFGS took an average of " + algorithm_times(i) + " seconds.")
            case 2
                disp("fminunc took an average of " + algorithm_times(i) + " seconds.")
            case 3
                disp("fmincon-activeset took an average of " + algorithm_times(i) + " seconds.")
            case 4
                disp("fmincon-sqp took an average of " + algorithm_times(i) + " seconds.")
            case 5
                disp("fmincon-interiorpoint took an average of " + algorithm_times(i) + " seconds.")
        end
    end
end


% symbolic function for our algorithm
function f = objfun(x,n,a,b,c,k,T)
%objfun is the approximated penalty function described in the report
%   (section 3.1.3). This MATLAB function specifically returns a symbolic
%   function for use with MATLAB's Symbolic Toolbox.
%
%   Parameters
%   ----------
%   x : 1xn symbol vector
%   n : int
%       size of symbolic x (or in the context of our project, 
%                           number of ads) 
%   a : 1xn vector
%       parameter set for a values, a > 0
%   b : 1xn vector
%       parameter set for b v alues
%   c : 1xn vector
%       parameter set for c values
%   k : double
%       must have k > 0
%   T : double
%       total ad time allocation constraint
%
%   Returns
%   -------
%   f : symbolic function
%       this is the penalty function from 3.1.3

    m = n;
    of = sym('of', [1, m]);
    cons = sym('cons', [1, 2*m+1]);
    f = sym('f', [1, m]);
    
    % Penalty Function 
    alpha = 10;
    for i = 1:m
        of(i) = maxap(-a(i)*k*x(i).^2,-b(i)); % objective function
        cons(i) = modap(maxap(c(i)-x(i),0)); % constraints
        cons(i+m) = modap(maxap(-x(i), 0));
    end
    cons(end) = modap(maxap(sum(x) - T,0)); % final constraint

    f = sum(of) + alpha*sum(cons); % penalty function
end

% scalar function for fminunc
function f = sobjfun(x,m,a,b,c,k,T)
%objfun is the approximated penalty function described in the report
%   (section 3.1.3). This MATLAB function specifically returns a scalar
%   value of f. Therefore all the inputs are the same as objfun (note m =
%   n) except that x is a scalar 1xn vector.

    n_cons = 2*m+1;
    alpha = 10; % penalty parameter
    of = zeros(m,1);
    cons = zeros(n_cons,1);
    for i = 1:m
        of(i) = maxap(-a(i)*k*x(i).^2,-b(i));
        cons(i) = modap(maxap(c(i)-x(i),0));
        cons(i+m) = modap(maxap(-x(i), 0));
    end
    cons(end) = modap(maxap(sum(x) - T,0));

    f = sum(of) + alpha*sum(cons);
end

% constrained objective function for fmincon
function f = conobjfun(x,a,b,k)
%objfun is the approximated objective function for the initial NLP
%   described in section 2.2. Inputs are the same as sobjfun.

    f = sum(max(-a.*k.*x.^2, -b));
end