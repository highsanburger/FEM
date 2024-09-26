import numpy as np
from dolfinx import *

# Create mesh (2D domain)
nx, ny = 30, 30  # Number of elements in x and y directions
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

# Define function space for Electric field
V = FunctionSpace(mesh, "P", 1)  # 'P' is Lagrange basis functions

# Define constants
k0 = 2 * np.pi / 0.5  # Wavenumber in free space (for wavelength 0.5 units)
eps_r = Constant(
    ((2.0, 0.1), (0.1, 1.5))
)  # Relative permittivity tensor for anisotropic medium
mu_r = Constant(1.0)  # Relative permeability (assuming isotropic for simplicity)

# Define trial and test functions
E = TrialFunction(V)
v = TestFunction(V)

# Define source term (current or external excitation)
f = Expression("sin(2*pi*x[0]) * sin(2*pi*x[1])", degree=2)

# Define the weak form for Maxwell's equations (Helmholtz equation for E field)
a = dot(eps_r * grad(E), grad(v)) * dx
L = f * v * dx

# Boundary conditions (e.g., Dirichlet boundary conditions, E = 0 at the boundaries)
bc = DirichletBC(V, Constant(0), "on_boundary")

# Solve the linear system
E_sol = Function(V)
solve(a == L, E_sol, bc)

# Plot solution
import matplotlib.pyplot as plt

plot(E_sol)
plt.title("Electric Field Distribution in Anisotropic Medium")
plt.show()
