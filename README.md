This MATLAB code computes an optimal feedback controller for the Kuramoto Sivashinsky 
Equation with zero Dirichlet and Neumann boundary conditions, discretised
using finite differences. The objective is the space-time averaged energy of the system.

Gradient descent is used to find the optimum controller. The derivatives are 
provided by the function 'MSS', which uses adjoint Preconditioned Multiple 
Shooting Shadowing theory. This requires the iterative solution to a linear 
system of equations that is preconditioned and regularised for faster convergence using GMRES.
The time stepper ODE45 is used to integrate the non-linear and linearised
tangent and adjoint equations. 'svds' is used to compute the singular value 
decomposition of the state transition matrices (for computing the
preconditioner).

The code outputs the uncontrolled solution contour, and the objective on 
each iteration. Upon convergence of the algorithm, the controlled
solution contour is also plotted, as well as the feedback matrix.
