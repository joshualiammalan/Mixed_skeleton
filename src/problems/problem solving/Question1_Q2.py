import lib.fem_modules.utils.plotting as plotting
import src.fem_modules.elements2d.quads.biquadratic_vector as bqs
import src.fem_modules.solver.solver as s
import src.fem_modules.utils.boundary_conditions as bcs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Define mesh parameters
height = 2.0  # Height of the mesh
width = 10.0  # Width of the mesh
num_elements_x = 4  # Number of elements in each column
num_elements_y = 2 # Number of elements in each row

# Calculate the number of nodes in each direction
num_nodes_x = num_elements_x * 2 + 1  # There are two nodes between each element
num_nodes_y = num_elements_y * 2 + 1  # There are two nodes between each row of elements

# Calculate nodes per element
nodes_per_element = 9  # Q2 elements

# Create global node positions
x_nodes = np.zeros((num_nodes_x * num_nodes_y, 2))
dx = width / (num_elements_x * 2)
dy = height / (num_elements_y * 2)

for i in range(num_nodes_y):
    for j in range(num_nodes_x):
        node_number = i * num_nodes_x + j
        x_nodes[node_number, 0] = j * dx
        x_nodes[node_number, 1] = i * dy

# Create the Inter Connectivity Array (ICA)
ICA = np.zeros((num_elements_x * num_elements_y, nodes_per_element), dtype=int)

for i in range(num_elements_y):
    for j in range(num_elements_x):
        element_number = i * num_elements_x + j
        base_node = i * num_nodes_x * 2 + j * 2
        node_numbers = [
            base_node,
            base_node + 1,
            base_node + 2,
            base_node + num_nodes_x,
            base_node + num_nodes_x + 1,
            base_node + num_nodes_x + 2,
            base_node + 2 * num_nodes_x,
            base_node + 2 * num_nodes_x + 1,
            base_node + 2 * num_nodes_x + 2,
        ]
        ICA[element_number, :] = node_numbers

# Plot nodal positions and element numbers
plt.figure(figsize=(8, 6))
plt.scatter(x_nodes[:, 0], x_nodes[:, 1], c='blue', s=50, marker='o', label='Nodes')
for i, (x, y) in enumerate(x_nodes):
    plt.text(x, y, str(i), fontsize=12, ha='center', va='bottom')

for i in range(num_elements_x * num_elements_y):
    element_x = x_nodes[ICA[i], 0]
    element_y = x_nodes[ICA[i], 1]
    plt.text(np.mean(element_x), np.mean(element_y), str(i + 1), fontsize=12, ha='center', va='center')

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Nodal Positions and Element Numbers')
plt.legend()
plt.grid()
plt.show()

print(ICA)

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
d_fixed = bcs.DirichletBoundaryCondition(f=d_f1, nodes=np.array([0, 1, 36, 72]))

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
# plt.figure(1)
# plotting.plot_quad_result(plt, ICA, x_nodes, ux, n_contours=200)
# plotting.plot_mesh(plt, ICA, x_nodes, node_numbers=False, el_numbers=False)
# plt.tight_layout()
# plt.show()
#
# plt.figure(2)
# plotting.plot_quad_result(plt, ICA, x_nodes, uy, n_contours=200)
# plotting.plot_mesh(plt, ICA, x_nodes, node_numbers=False, el_numbers=False)
# plt.tight_layout()
# plt.show()



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

