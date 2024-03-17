function PMSS_Feedback_Controller_KSE

% This MATLAB code computes a feedback controller for the Kuramoto Sivashinsky 
% Equation with zero Dirichlet and Neumann boundary conditions, discretised
% using finite differences. The objective is the space-time averaged energy

% Gradient descent is used to find the optimum controller. The derivatives are 
% provided by the function 'MSS', which uses adjoint Preconditioned Multiple 
% Shooting Shadowing. The time stepper ODE45 is used to integrate the primal, 
% tangent and adjoint equations. 'svds' is used to compute the singular value 
% decomposition of the state transition matrices (for compute the
% preconditioner). GMRES is used for the solution to the PMSS matrix system
% (in MATVEC form). 

% The code outputs the uncontrolled solution contour, and the objective on 
% each iteration. Upon convergence of the algorithm, the controlled
% solution contour is also plotted, as well as the feedback matrix

T      = 50;                         % Time Horizon
T_min  = -1000;                      % Integrate from T_min
dt     = 1e-1;                       
t_span = (T_min:dt:T);               % Integrate in [T_min,T] and interpolate every dt
P      = 5;                          % Number of segments
tp     = (0:T/P:T)';
L      = 128;                        % Length of domain
N      = 127;                        % # Interior nodes, 1,2,...,n
dx     = L/(N+1);                    % Node spacing
init_cond = linspace(1,1,N);         % Initial condition vector
x      = dx:dx:L-dx;
c      = 0;                          % KS Parameter
opts   = odeset('RelTol',1e-4,'AbsTol',1e-4);
  
% Algorithm parameters
eps    = 0.01;                       % Target cost function tol
a      = 10;                         % Descent Step Size
xi_p   = 2;                          % Restrict feedback matrix kernel to distance xi_p

% Preconditioner parameters
No_sing  = 15;                       % Number of PC singular values
Maxit    = 1;                        % # Diagonalisation iterations
Subspace = No_sing +2;               % Subspace dimension

% GMRES Solver Parameters
tol    = 1e-6;                       % Schur Complement solver tolerance
gamma  = 0.1;                        % Regularisation parameter
maxit  = 60;                         % Maximum # GMRES Iterations

%%
% t_span for the constraints and adjoints
for p=1:1:length(tp)-1
    tspan_con = tp(p):dt:tp(p+1);                                            
    t_span_con(:,p) = tspan_con';
end

% Initialise the trajectory and discard solution in [T_min,0]
[t,u]  = ode45(@(t,u) ks_solve(t,u,dx,N,c), t_span, init_cond, opts);       % Solve the primal (KSE)
t0_loc = ceil((length(t))*(-T_min/(T-T_min)));
t      = t(t0_loc:end); u      = u(t0_loc:end,:);                           % Store trajectory in [0,T]
init_cond = u(1,:)';                                                        % Store initial condition

% Initialise the algorithm
K = zeros(N);                                                               % Feedback matrix

% Compute Initial cost function 
J_x    = (1/L)*trapz(x,(u).^2,2); J   = (1/T)*trapz(t,J_x);                 % Compute objective J

Init_cond = zeros(N,1);     % Zero initial condition to compute RHS
Prj   = cell(P,1);
f_p   = cell(P,1);
g     = cell(P,1);
phi_g = cell(P,1);

figure(1)
contourf(x,0:dt:T,u)
title('Initial Solution')

figure(2)
semilogy(0,abs(J),'bo')
hold on
xlabel('Iteration #')
ylabel('J')
title('Cost Function')

k=0; JJ=0;

%%
% Start optimisation loop
while abs(J-JJ)/JJ > eps

k=k+1;

DJ_dK = MSS;                                                                % Call MSS function to obtain sensitivity matrix DJ_dK

% Use backtracking to compute the step size
JJ = J;
J = costfun(a);
while J>JJ
a = 0.5*a;
J = costfun(a);
end

K = K - a*DJ_dK;                                                            % Update feedback matrix

[t,u]  = ode45(@(t,u) ks_solve1(t,u,dx,N,c,K), 0:dt:T, init_cond, opts);    % Update primal solution

J_x = (1/L)*trapz(x,u.^2,2);
J   = (1/T)*trapz(t,J_x);                                                   % Compute objective J

figure(2)
semilogy(k,abs(J),'bo')
hold on
xlabel('Iteration #')
ylabel('J')
title('Cost Function')

end

figure(3)
contourf(x,0:dt:T,u)
title('Controlled Solution')

figure(4)
imagesc(K)
xlabel('Column Number (j)')
ylabel('Row Number (i)')
title('Converged Feedback Matrix')

%%
% Function to compute objective function
function J = costfun(a)

KK = K - a*(DJ_dK);

[t,u_test]  = ode45(@(t,u) ks_solve1(t,u,dx,N,c,KK), 0:dt:T, init_cond, opts);

J_x = (1/L)*trapz(x,u_test.^2,2);
J   = (1/T)*trapz(t,J_x);

end

% Function to compute Sensitivity matrix
function DJ_dK = MSS
             
f1 = (2/(dx^2)-7/(dx^4))*u(:,1)  + (-(2*c+u(:,2))/(4*dx)-1/(dx^2)+4/(dx^4)).*u(:,2) - (1/(dx^4))*u(:,3) + sum(K(1,:).*(u),2);
f2 = ((2*c+u(:,1))/(4*dx)-1/(dx^2)+4/(dx^4)).*u(:,1) + (2/(dx^2)-6/(dx^4))*u(:,2) + (-(2*c+u(:,3))/(4*dx)-1/(dx^2)+4/(dx^4)).*u(:,3) -(1/(dx^4))*u(:,4) + sum(K(2,:).*(u),2);
FF = zeros((T/dt)+1,N-4);
for i = 3:N-2
    f = (-1/(dx^4))*u(:,i-2) + ((2*c+u(:,i-1))/(4*dx)-1/(dx^2)+4/(dx^4)).*u(:,i-1)  +  (2/(dx^2)-6/(dx^4))*u(:,i)  +  (-(2*c+u(:,i+1))/(4*dx)-1/(dx^2)+4/(dx^4)).*u(:,i+1)  -  (1/(dx^4))*u(:,i+2) + sum(K(i,:).*(u),2);
    FF(:,i-2) =f;
end
fNminus1 = -(1/(dx^4))*u(:,N-3) + ((2*c+u(:,N-2))/(4*dx)-1/(dx^2)+4/(dx^4)).*u(:,N-2) +  (2/(dx^2)-6/(dx^4))*u(:,N-1)  +   (-(2*c+u(:,N))/(4*dx)-1/(dx^2)+4/(dx^4)).*u(:,N) + sum(K(N-1,:).*(u),2);
fN = -(1/(dx^4))*u(:,N-2)   + ((2*c+u(:,N-1))/(4*dx)-1/(dx^2)+4/(dx^4)).*u(:,N-1)   +  (2/(dx^2)-7/(dx^4))*u(:,N) + sum(K(N,:).*(u),2);
F = [f1 f2 FF fNminus1 fN];

for p = 1:1:P
    F_p = F(ceil(1+(T/dt)*(p/P)),:)';     %f(u,t) at the checkpoints (p=1,2,...,P)
    j_p = J_x(ceil(1+(T/dt)*(p/P))) ; 
    
    [ta,w] = ode45(@(ta,w) adj_solve(ta,w,t,u,dx,N,c,K), flipud(t_span_con(:,p)), Init_cond, opts);   % Solve Adjoint Equations with zero IC
    gp = w(end,:)';
    g{p} = gp;
    
    [tc,v] = ode45(@(tc,v) con_hom_solve(tc,v,t,u,dx,N,c,K), t_span_con(:,p), gp, opts);    % Solve homogeneous Constraint Equations  
    phi_g{p} = v(end,:)';
    
    Prj{p} = eye(N) - (F_p*F_p')/(F_p'*F_p);                                                % Store projections 
    f_p{p} = F_p;                                                                           % Store f(u,t) at the checkpoints (p=1,2,...,P)
    J_p(p,1) = j_p;
    
    % Compute SVD of the state transition matrices
    [U,S,V] = svds(@SVD_phi,[N N],No_sing,'largest','Tolerance',5,'MaxIterations',Maxit,'SubspaceDimension',Subspace);
    
    % Compute Preconditioner
    M2 = U*(S^-2)*U' + (eye(N)-U*U');
    M_2{p} = M2;
end

Phi_g   = cell2mat(phi_g);
G       = cell2mat(g);
RHS     = -(Phi_g - [G(N+1:end);zeros(N,1)]);

for p = 1:1:P
    rhs_pc{p} = (M_2{p}*RHS(1+N*(p-1):N*p))';
end

RHS_PC = cell2mat(rhs_pc);

Prj0 = mat2cell(eye(N) - (F(1,:)'*F(1,:))/(F(1,:)*F(1,:)'),N,N);            % Projection at t=0
Prj  = [Prj0;Prj];                                                          % Store projections at checkpoints (t0,t1,...,tP)

[ws,flag,relres,iter,resvec] = gmres(@Sx_vector,RHS_PC',[],tol,maxit);      % Solve Schur Complement using GMRES

dJ_dK = zeros(N); DJ_dK = zeros(N); DJ_DK = zeros(N);
for p = 0:1:P-1
    
    W = Prj{p+2}*ws(1+N*(p):N*(p+1)) + (1/T)*((J-J_p(p+1))/(f_p{p+1}'*f_p{p+1}))*f_p{p+1};
    
    [ta,w] = ode45(@(ta,w) adj_solve(ta,w,t,u,dx,N,c,K), flipud(t_span_con(:,p+1)), W, opts);   % Solve Adjoint Equations    
    w = flipud(w);

    a1 = 1-p+p*(length(t_span_con(:,1))); a2 = -p+(p+1)*(length(t_span_con(:,1)));
    
    for i=1:N
    dJ_dK(i,:) = trapz(tc,w(:,i).*(u(a1:a2,:)));
    end
    
    DJ_dK = DJ_dK + dJ_dK;
    
end

for i  = -xi_p/dx:xi_p/dx
    DJ_DK = DJ_DK + diag(diag(DJ_dK,i),i);
end
DJ_dK = DJ_DK;
end
M_2;

% Function to compute Schur complement MATVEC products
function Sw = Sx_vector(W)
    
    for p =1:P
    [ta,w] = ode45(@(ta,w) adj_hom_solve(ta,w,t,u,dx,N,c,K), flipud(t_span_con(:,p)), Prj{p+1}*W(1+N*(p-1):N*p), opts);   % Solve adjoint Equations 
    w_end      = w(end,:)';
    phiTw{p,:} = w_end;                                       % Adjoint Propagator MATVECs
    end
    phiT_w = cell2mat(phiTw);
    ATw    = [-phiT_w(1:N); W(1:N*(P-1))-phiT_w(N+1:end); W(end-(N-1):end)];  % MATVEC Product (A'w)
    
    for p=1:P
    [tc,v] = ode45(@(tc,v) con_hom_solve(tc,v,t,u,dx,N,c,K), t_span_con(:,p), ATw(1+N*(p-1):N*p), opts);   % Solve homogeneous Constraint Equations  
    v_end   = v(end,:)';                                        % Constraint Propagator MATVECs
    Sw{p,:} = M_2{p}*(-(Prj{p+1}*v_end)+ATw(1+(N*p):N*(p+1)));           % MATVEC product (AA'w) 
    end
    Sw = cell2mat(Sw);
    Sw = gamma*W + Sw;
end

% Function to compute SVD of the state transition matrices
function Phix = SVD_phi(x,tflag)
         if strcmp(tflag,'notransp')
         [tc,v] = ode45(@(tc,v) con_hom_solve(tc,v,t,u,dx,N,c,K), t_span_con(:,p), x, opts);   % Solve homogeneous Constraint Equations  
         Phix   = Prj{p}*v(end,:)';                                        % Constraint Propagator MATVECs 
         else
         [ta,w] = ode45(@(ta,w) adj_hom_solve(ta,w,t,u,dx,N,c,K), flipud(t_span_con(:,p)), Prj{p}*x, opts);   % Solve adjoint Equations 
         Phix   = w(end,:)'; 
         end
end

% Function to compute uncontrolled KSE
function dudt = ks_solve(t,u,dx,N,c)
       
         dudt = zeros(N,1);
         dudt(1) = (2/(dx^2)-7/(dx^4))*u(1)                   + (-(2*c+u(2))/(4*dx)-1/(dx^2)+4/(dx^4))*u(2) - (1/(dx^4))*u(3);
         dudt(2) = ((2*c+u(1))/(4*dx)-1/(dx^2)+4/(dx^4))*u(1) + (2/(dx^2)-6/(dx^4))*u(2) + (-(2*c+u(3))/(4*dx)-1/(dx^2)+4/(dx^4))*u(3) -(1/(dx^4))*u(4);
         for i = 3:N-2
         dudt(i) = (-1/(dx^4))*u(i-2) + ((2*c+u(i-1))/(4*dx)-1/(dx^2)+4/(dx^4))*u(i-1)  +  (2/(dx^2)-6/(dx^4))*u(i)  +  (-(2*c+u(i+1))/(4*dx)-1/(dx^2)+4/(dx^4))*u(i+1)  -  (1/(dx^4))*u(i+2); 
         end
         dudt(N-1) = -(1/(dx^4))*u(N-3) + ((2*c+u(N-2))/(4*dx)-1/(dx^2)+4/(dx^4))*u(N-2) +  (2/(dx^2)-6/(dx^4))*u(N-1)  +   (-(2*c+u(N))/(4*dx)-1/(dx^2)+4/(dx^4))*u(N);
         dudt(N) = -(1/(dx^4))*u(N-2)   + ((2*c+u(N-1))/(4*dx)-1/(dx^2)+4/(dx^4))*u(N-1)   +  (2/(dx^2)-7/(dx^4))*u(N);
end

% Function to compute controlled KSE
function dudt = ks_solve1(t,u,dx,N,c,K)
         dudt = zeros(N,1);
         dudt(1) = (2/(dx^2)-7/(dx^4))*u(1)                   + (-(2*c+u(2))/(4*dx)-1/(dx^2)+4/(dx^4))*u(2) - (1/(dx^4))*u(3) + K(1,:)*u;
         dudt(2) = ((2*c+u(1))/(4*dx)-1/(dx^2)+4/(dx^4))*u(1) + (2/(dx^2)-6/(dx^4))*u(2) + (-(2*c+u(3))/(4*dx)-1/(dx^2)+4/(dx^4))*u(3) -(1/(dx^4))*u(4) + K(2,:)*u;
         for i = 3:N-2
         dudt(i) = (-1/(dx^4))*u(i-2) + ((2*c+u(i-1))/(4*dx)-1/(dx^2)+4/(dx^4))*u(i-1)  +  (2/(dx^2)-6/(dx^4))*u(i)  +  (-(2*c+u(i+1))/(4*dx)-1/(dx^2)+4/(dx^4))*u(i+1)  -  (1/(dx^4))*u(i+2) + K(i,:)*u; 
         end
         dudt(N-1) = -(1/(dx^4))*u(N-3) + ((2*c+u(N-2))/(4*dx)-1/(dx^2)+4/(dx^4))*u(N-2) +  (2/(dx^2)-6/(dx^4))*u(N-1)  +   (-(2*c+u(N))/(4*dx)-1/(dx^2)+4/(dx^4))*u(N) + K(N-1,:)*u;
         dudt(N) = -(1/(dx^4))*u(N-2)   + ((2*c+u(N-1))/(4*dx)-1/(dx^2)+4/(dx^4))*u(N-1)   +  (2/(dx^2)-7/(dx^4))*u(N) + K(N,:)*u;
end

% Function to compute tangent equations 
function dvdt = con_hom_solve(tc,v,t,u,dx,N,c,K)
         
         u    = interp1(t,u,tc);
         
         dvdt    = zeros(N,1);
         dvdt(1) = (2/(dx^2)-7/(dx^4))*v(1) + (-(c+u(2))/(2*dx)-1/(dx^2)+4/(dx^4))*v(2) - (1/(dx^4))*v(3) + K(1,:)*v;
         dvdt(2) = ((c+u(1))/(2*dx)-1/(dx^2)+4/(dx^4))*v(1) + (2/(dx^2)-6/(dx^4))*v(2) + (-(c+u(3))/(2*dx)-1/(dx^2)+4/(dx^4))*v(3) -(1/(dx^4))*v(4) + K(2,:)*v;
         for i = 3:N-2
         dvdt(i) = (-1/(dx^4))*v(i-2) + ((c+u(i-1))/(2*dx)-1/(dx^2)+4/(dx^4))*v(i-1)  +  (2/(dx^2)-6/(dx^4))*v(i)  +  (-(c+u(i+1))/(2*dx)-1/(dx^2)+4/(dx^4))*v(i+1)  -  (1/(dx^4))*v(i+2) + K(i,:)*v; 
         end
         dvdt(N-1) = -(1/(dx^4))*v(N-3) + ((c+u(N-2))/(2*dx)-1/(dx^2)+4/(dx^4))*v(N-2) +  (2/(dx^2)-6/(dx^4))*v(N-1)  +   (-(c+u(N))/(2*dx)-1/(dx^2)+4/(dx^4))*v(N) + K(N-1,:)*v;
         dvdt(N) = -(1/(dx^4))*v(N-2)   + ((c+u(N-1))/(2*dx)-1/(dx^2)+4/(dx^4))*v(N-1)   +  (2/(dx^2)-7/(dx^4))*v(N) + K(N,:)*v;
end

% Function to compute adjoint equations 
function dwdt = adj_solve(ta,w,t,u,dx,N,c,K)
         
         u     = interp1(t,u,ta);
         
         dwdt    = zeros(N,1);
         dwdt(1) = -(2/(dx^2)-7/(dx^4))*w(1) - ((c+u(1))/(2*dx) - 1/(dx^2)+4/(dx^4))*w(2) + (1/(dx^4))*w(3) - K(:,1)'*w - (2*dx/(L*T))*(u(1))^1;
         dwdt(2) = -(-(c+u(2))/(2*dx)-1/(dx^2)+4/(dx^4))*w(1) - (2/(dx^2)-6/(dx^4))*w(2) - ((c+u(2))/(2*dx)-1/(dx^2)+4/(dx^4))*w(3) + (1/(dx^4))*w(4) - K(:,2)'*w - (2*dx/(L*T))*(u(2))^1;
         for i   = 3:N-2
         dwdt(i) =  (1/(dx^4))*w(i-2) - (-(c+u(i))/(2*dx)-1/(dx^2)+4/(dx^4))*w(i-1) - (2/(dx^2)-6/(dx^4))*w(i) - ((c+u(i))/(2*dx)-1/(dx^2)+4/(dx^4))*w(i+1) + (1/(dx^4))*w(i+2) - K(:,i)'*w - (2*dx/(L*T))*(u(i))^1;
         end
         dwdt(N-1) = (1/(dx^4))*w(N-3) - (-(c+u(N-1))/(2*dx)-1/(dx^2)+4/(dx^4))*w(N-2) - (2/(dx^2)-6/(dx^4))*w(N-1) - ((c+u(N-1))/(2*dx)-1/(dx^2)+4/(dx^4))*w(N) - K(:,N-1)'*w - (2*dx/(L*T))*(u(N-1))^1;
         dwdt(N)   = (1/(dx^4))*w(N-2) - (-(c+u(N))/(2*dx)-1/(dx^2)+4/(dx^4))*w(N-1) - (2/(dx^2)-7/(dx^4))*w(N) - K(:,N)'*w - (2*dx/(L*T))*(u(N))^1;
end

% Function to compute homogeneous adjoint equations 
function dwdt = adj_hom_solve(ta,w,t,u,dx,N,c,K)
         
         u    = interp1(t,u,ta);
    
         dwdt    = zeros(N,1);
         dwdt(1) = -(2/(dx^2)-7/(dx^4))*w(1) - ((c+u(1))/(2*dx) - 1/(dx^2)+4/(dx^4))*w(2) + (1/(dx^4))*w(3) - K(:,1)'*w;
         dwdt(2) = -(-(c+u(2))/(2*dx)-1/(dx^2)+4/(dx^4))*w(1) - (2/(dx^2)-6/(dx^4))*w(2) - ((c+u(2))/(2*dx)-1/(dx^2)+4/(dx^4))*w(3) + (1/(dx^4))*w(4) - K(:,2)'*w;
         for i   = 3:N-2
         dwdt(i) =  (1/(dx^4))*w(i-2) - (-(c+u(i))/(2*dx)-1/(dx^2)+4/(dx^4))*w(i-1) - (2/(dx^2)-6/(dx^4))*w(i) - ((c+u(i))/(2*dx)-1/(dx^2)+4/(dx^4))*w(i+1) + (1/(dx^4))*w(i+2) - K(:,i)'*w;
         end
         dwdt(N-1) = (1/(dx^4))*w(N-3) - (-(c+u(N-1))/(2*dx)-1/(dx^2)+4/(dx^4))*w(N-2) - (2/(dx^2)-6/(dx^4))*w(N-1) - ((c+u(N-1))/(2*dx)-1/(dx^2)+4/(dx^4))*w(N) - K(:,N-1)'*w;
         dwdt(N)   = (1/(dx^4))*w(N-2) - (-(c+u(N))/(2*dx)-1/(dx^2)+4/(dx^4))*w(N-1) - (2/(dx^2)-7/(dx^4))*w(N) - K(:,N)'*w;
end

end