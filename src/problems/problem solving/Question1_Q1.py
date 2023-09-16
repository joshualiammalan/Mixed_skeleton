import lib.fem_modules.utils.plotting as plotting
import src.fem_modules.elements2d.quads.bilinear_vector as bqs
import src.fem_modules.solver.solver as s
import src.fem_modules.utils.boundary_conditions as bcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

#Solution file for the Quadrilateral element mesh.

# Material properties
youngs_modulus = 1500  # Young's modulus (Pa)
poissons_ratio = 0.45  # Poisson's ratio


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
    factor = youngs_modulus / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    def D(x):
        return factor * np.array([[1 - poissons_ratio, poissons_ratio, 0],
                                     [poissons_ratio, 1 - poissons_ratio, 0],
                                     [0, 0, (1 - 2 * poissons_ratio) / 2]])
    print(D(0))

x_nodes = np.array([ #define global node positions
    [0, 0],
    [2.5, 0],
    [5, 0],
    [7.5, 0],
    [10, 0],
    [0, 1],
    [2.5, 1],
    [5, 1],
    [7.5, 1],
    [10, 1],
    [0, 2],
    [2.5, 2],
    [5, 2],
    [7.5, 2],
    [10, 2]
])

## Inter-connectivity array
ICA = np.array([ #define ICA
    [1, 2, 7, 6], #element global nodes
    [2, 3, 8, 7],
    [3, 4, 9, 8],
    [4, 5, 10, 9],
    [6, 7, 12, 11],
    [7, 8, 13, 12],
    [8, 9, 14, 13],
    [9, 10, 15, 14]
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
    # Calculate the linearly varying force in the x-direction
    # The force varies linearly from 0 to f_max along the x-axis
    f_max = 3000  # Maximum force magnitude in N
    slope = f_max / 10.0  # Slope of the linear variation (beam length is 10)
    F_x = slope * x[0]  # Linear variation of force along x-axis

    # Return the force as a 2D numpy array (horizontal and vertical components)
    return np.array([F_x, 0])

## A function defining the dirichlet condition on the appropriate nodes, fixed boundaries
def d_f1(x):
    return 0

## The Dirichlet condition on the left in the x direction,nodes are according to global node numbering
d_fixed = bcs.DirichletBoundaryCondition(f=d_f1, nodes=np.array([0, 1, 10, 20]))

d_bcs = np.array([d_fixed]) #matrix containing all direchlet bcs functions and nodal positions

## A function defining the neumann condition (No neumann boundary condition)
def t1(x): #traction
    return np.array([[0,0],
                    [0,0]])

# The Neumann condition
n_beam = bcs.NeumannBoundaryCondition(q=t1,
                                 edges_nodes=np.array([
                                     [3, 0],
                                     # [9, 10],
                                 ]),
                                 edges_normals=np.array([
                                     [0, 1],
                                     # [0, 1],
                                 ]))

n_bcs = np.array([n_beam])


## The value of the displacement solution at the nodes
d = s.solve(ICA=ICA,
            x_nodes=x_nodes,
            D=D,
            S=S,
            n_bcs=n_bcs,
            d_bcs=d_bcs,
            el=bqs)

print(d) #solution
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