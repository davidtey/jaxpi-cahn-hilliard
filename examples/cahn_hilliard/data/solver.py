import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Define parameters
nn = 511
steps = 250
xmin, xmax = -40, 40
tmax = 140
tspan = np.linspace(0, tmax, steps + 1)
x = np.linspace(xmin, xmax, nn+1)[:-1]  # Exclude duplicate endpoint

def laplacian_matrix(n, dx):
    """Create a sparse matrix for the second derivative using finite differences."""
    diagonals = [-2*np.ones(n), np.ones(n-1), np.ones(n-1)]
    return sp.diags(diagonals, [0, -1, 1]) / dx**2

def biharmonic_matrix(n, dx):
    """Create a sparse matrix for the fourth derivative using finite differences."""
    L = laplacian_matrix(n, dx)
    return L @ L

# Define linear and nonlinear operators
dx = (xmax - xmin) / nn
L2 = laplacian_matrix(nn, dx)
L4 = biharmonic_matrix(nn, dx)
L = -L2 - L4 - 0.01 * sp.eye(nn)

def rhs(t, u):
    """Right-hand side of the PDE system."""
    u = u.reshape(-1, 1)
    nonlin = -L2 @ (u - u**3)
    return (L @ u + nonlin).flatten()

# Initial condition
u0 = 0.9 * np.sin(0.42 * np.pi * x) + 0.1 * np.sin(10 * np.pi * x)

# Solve PDE
sol = spi.solve_ivp(rhs, [0, tmax], u0, t_eval=tspan, method='RK45')
usol = sol.y.T  # shape = (steps+1, nn)

# Append periodic boundary condition
usol = np.hstack([usol, usol[:, 0].reshape(-1, 1)])

# Visualization
t, x = np.meshgrid(tspan, np.append(x, xmax))
plt.pcolormesh(t, x, usol, shading='auto', cmap='jet')
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Space")
plt.title("Extended Cahn-Hilliard Equation")
plt.show()

# Save data
np.savez('ch.npz', t=tspan, x=np.append(x, xmax), usol=usol)
