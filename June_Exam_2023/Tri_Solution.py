import src.fem_modules.utils.plotting as plotting
import src.fem_modules.elements2d.triangles.linear_vector as lts
import src.fem_modules.solver.solver as s
import src.fem_modules.utils.boundary_conditions as bcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import math

#Solution file for the Quadrilateral element mesh.

# Material properties
youngs_modulus = 27*10**8  # Young's modulus (Pa)
poissons_ratio = 0.31   # Poisson's ratio


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


x_nodes = np.array([    #define global node positions
    [8.036, 19.4],
    [0, 19.4],
    [-8.036, 19.4],
    [-16.072, 8.036],
    [-16.072, 0],
    [-8.036, 0],
    [0, 0],
    [8.036, 0],
    [16.072, 0],
    [16.072, 8.036],
])

## Inter-connectivity array
ICA = np.array([ #define ICA
    [5, 6, 4], #element global nodes
    [4,6,3],
    [6,7,3],
    [7,2,3],
    [7,1,2],
    [7,8,1],
    [8,10,1],
    [8,9,10],
])
ICA = ICA-1
ICA = ICA.astype(int)
# print(ICA)
# print(x_nodes)
# print()
# # Plotting the mesh to see if it was setup correctly
# plotting.plot_mesh(plt, ICA, x_nodes)
# plt.tight_layout()
# # # plt.savefig('figures/quad_mesh.png')
# # # Unncomment to see mesh
# plt.show()


## A function defining the body force term
def S(x):
    return np.array([0,0]) #zero body force

## A function defining the dirichlet condition on the appropriate nodes, fixed boundaries
def d_f1(x):
    return 0

## The Dirichlet condition on the left due to the fixed edge. Nodes 4 and 5 are fixed in x and y directions
d_fixed = bcs.DirichletBoundaryCondition(f=d_f1, nodes=np.array([6,7,8,9]))

d_bcs = np.array([d_fixed]) #matrix containing all direchlet bcs functions and nodal positions


# # calculating normals for traction edge:
#
# def calculate_unit_normal(node1, node2):
#     dx = node2[0] - node1[0]
#     dy = node2[1] - node1[1]
#     magnitude = math.sqrt(dx**2 + dy**2)
#     unit_normal_x = dy / magnitude
#     unit_normal_y = -dx / magnitude  # Negative sign for positive y direction
#     return unit_normal_x, unit_normal_y
# nodes = [
#     [(8.036,19.4), (16.072,8.036)],  # Edge 1
# ]
#
# unit_normals = []
# for node1, node2 in nodes:
#     unit_normal = calculate_unit_normal(node1, node2)
#     unit_normals.append(unit_normal)
#
# # Print the unit normals
# for i, (x, y) in enumerate(unit_normals, start=1):
#     print(f"Edge {i} Unit Normal: [{x}, {y}]")


## A function defining the neumann condition
def t1(x): #traction
    return np.array([[-18.77*10**3,0],
                    [0,-10.838*10**3]])

# The Neumann condition on the top
n_top = bcs.NeumannBoundaryCondition(q=t1,
                                 edges_nodes=np.array([
                                     [0, 9],
                                 ]),
                                 edges_normals=np.array([
                                     [0.8165, 0.5774],
                                 ]))

n_bcs = np.array([n_top]) #matrix containing all neumann bcs functions, normals and nodal positions

## The value of the displacement solution at the nodes
d = s.solve(ICA=ICA,
            x_nodes=x_nodes,
            D=D,
            S=S,
            n_bcs=n_bcs,
            d_bcs=d_bcs,
            el=lts)

print(d) #solution
ux = d[::2]
uy = d[1::2]
print(ux)
plt.figure()
plotting.plot_tri_result(plt, ICA, x_nodes, ux, n_contours=200)
plotting.plot_mesh(plt, ICA, x_nodes, node_numbers=False, el_numbers=False)
plt.tight_layout()
plt.show()
