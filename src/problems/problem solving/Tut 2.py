import lib.fem_modules.utils.plotting as plotting
import src.fem_modules.elements2d.quads.Q1_P0 as bqs
import src.fem_modules.solver.solver as s
import src.fem_modules.utils.boundary_conditions as bcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

#Solution file for the Quadrilateral element mesh.

# Material properties
youngs_modulus = 250000  # Young's modulus (Pa)
poissons_ratio = 0.35  # Poisson's ratio
lambda_val = 216049.3827
mue = 92592.59259

# Plane stress condition
if False:
    factor = youngs_modulus / ((1 - poissons_ratio**2))
    def D(x):
        return factor * np.array([[1, poissons_ratio, 0],
                                     [poissons_ratio, 1, 0],
                                     [0, 0, (1 - poissons_ratio) / 2]])
    print(D(0))

# Plane strain condition
if True:
    def D(x):
        return mue * np.array([[2, 0, 0],
                               [0, 2, 0],
                               [0, 0, 1]])
    print(D(mue))

x_nodes = np.array([ #define global node positions
    [0, 0],
    [2, 0],
    [4, 0],
    [1, 1],
    [3, 1],
    [0, 2],
    [2, 2],
    [4, 2],
])

## Inter-connectivity array
ICA = np.array([ #define ICA
    [1, 2, 6, 7], #element global nodes
    [2, 3, 7, 8],
])
ICA = ICA-1
ICA = ICA.astype(int)
print(ICA)
# print(x_nodes)
print()
# Plotting the mesh to see if it was setup correctly
plotting.plot_mesh(plt, ICA, x_nodes)
plt.tight_layout()
# # plt.savefig('figures/quad_mesh.png')
# # Unncomment to see mesh
plt.show()

## A function defining the body force term
def S(x):
    return 0

## A function defining the dirichlet condition on the appropriate nodes, fixed boundaries
def d_f1(x):
    return 0

## The Dirichlet condition on the left in the x direction,nodes are according to global node numbering
d_fixed = bcs.DirichletBoundaryCondition(f=d_f1, nodes=np.array([0, 1, 10, 11]))

d_bcs = np.array([d_fixed]) #matrix containing all direchlet bcs functions and nodal positions

## A function defining the neumann condition (No neumann boundary condition)
def t1(x): #traction
    return np.array([[1000,0],
                    [0,0]])

# The Neumann condition
n_beam = bcs.NeumannBoundaryCondition(q=t1,
                                 edges_nodes=np.array([
                                     [2, 7],
                                     # [9, 10],
                                 ]),
                                 edges_normals=np.array([
                                     [1, 0],
                                     # [0, 1],
                                 ]))

n_bcs = np.array([n_beam])


## The value of the displacement solution at the nodes
d,p = s.solve_displacement_pressure(ICA=ICA,
            x_nodes=x_nodes,
            D=D,
            S=S,
            n_bcs=n_bcs,
            d_bcs=d_bcs,
            el=bqs,
            lambda_val=lambda_val)

print(d) #solution
print(p)
ux = d[::2]
uy = d[1::2]
print(ux)
print(uy)
plt.figure(1)
plotting.plot_quad_result(plt, ICA, x_nodes, ux, n_contours=200)
plotting.plot_mesh(plt, ICA, x_nodes, node_numbers=False, el_numbers=False)
plt.tight_layout()
plt.show()

plt.figure(2)
plotting.plot_quad_result(plt, ICA, x_nodes, uy, n_contours=200)
plotting.plot_mesh(plt, ICA, x_nodes, node_numbers=False, el_numbers=False)
plt.tight_layout()
plt.show()



# Generate LaTeX code for matrix solution
# Generate the LaTeX code for the table
# latex_code = r'''
# \begin{table}[ht]
#     \centering
#     \caption{Table of Solutions}
#     \begin{tabular}{|c|c|}
#         \hline
#         \rowcolor{gray!20}
#         \textbf{Global Node Number} & \textbf{Temperature} \\
# '''
#
# # Add rows to the LaTeX code
# for i, value in enumerate(d):
#     latex_code += f'        {i} & {value} \\\\\hline\n'
#
# # Complete the LaTeX code
# latex_code += r'''    \end{tabular}
# \end{table}
# '''
#
# # Print the LaTeX code
# print(latex_code)
