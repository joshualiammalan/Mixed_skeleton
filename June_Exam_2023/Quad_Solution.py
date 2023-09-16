import src.fem_modules.utils.plotting as plotting
import src.fem_modules.elements2d.quads.bilinear_vector as bqs
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
    [19.4, 0],
    [19.4, 8.036],
])

## Inter-connectivity array
ICA = np.array([ #define ICA
    [5, 6, 3, 4], #element global nodes
    [6, 7, 2, 3],
    [7, 8, 1, 2],
    [8, 9, 10, 1],
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


# calculating normals for traction edge:

def calculate_unit_normal(node1, node2):
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]
    magnitude = math.sqrt(dx**2 + dy**2)
    unit_normal_x = dy / magnitude
    unit_normal_y = -dx / magnitude  # Negative sign for positive y direction
    return unit_normal_x, unit_normal_y
nodes = [
    [(8.036,19.4), (19.4,8.036)],  # Edge 1
]

unit_normals = []
for node1, node2 in nodes:
    unit_normal = calculate_unit_normal(node1, node2)
    unit_normals.append(unit_normal)

# Print the unit normals
for i, (x, y) in enumerate(unit_normals, start=1):
    print(f"Edge {i} Unit Normal: [{x}, {y}]")


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
            el=bqs)

print(d) #solution
ux = d[::2]
uy = d[1::2]
print(ux)
plt.figure()
plotting.plot_quad_result(plt, ICA, x_nodes, uy, n_contours=200)
plotting.plot_mesh(plt, ICA, x_nodes, node_numbers=False, el_numbers=False)
plt.tight_layout()
plt.show()

strain_xx1 = np.zeros(4)
strain_yy1 = np.zeros(4)
strain_xy1 = np.zeros(4)
stress_xx1 = np.zeros(4)
stress_yy1 = np.zeros(4)
stress_xy1 = np.zeros(4)

xi = np.array([
    [-1,-1],
    [1,-1],
    [1,1],
    [-1,1]
])
x_nodes1 = np.array([
    [-16.072, 0],
    [-8.036, 0],
    [-8.036, 19.4],
    [-16.072, 8.036]
])
d1 = np.array([-1.18230957e-06, -2.53017994e-04,  3.21030543e-05, -1.50835342e-04, 4.35842710e-05, -6.27071960e-05,  0.00000000e+00,  0.00000000e+00])
for i in range(len(xi)):
    ref_node = xi[i]
    B_ref = bqs.B(x_nodes1, ref_node) #x_nodes element, hardcode
    strain = np.dot(B_ref, d1)
    strain_xx1[i] = strain[0]
    strain_yy1[i] = strain[1]
    strain_xy1[i] = strain[2]
    stress = np.dot(D(0), strain)
    stress_xx1[i] = stress[0]
    stress_yy1[i] = stress[1]
    stress_xy1[i] = stress[2]

strain_xx2 = np.zeros(4)
strain_yy2 = np.zeros(4)
strain_xy2 = np.zeros(4)
stress_xx2 = np.zeros(4)
stress_yy2 = np.zeros(4)
stress_xy2 = np.zeros(4)

xi = np.array([
    [-1,-1],
    [1,-1],
    [1,1],
    [-1,1]
])
x_nodes2 = np.array([
    [-8.036, 0],
    [0, 0],
    [0, 19.4],
    [-8.036, 19.4]
])
d2 = np.array([3.21030543e-05, -1.50835342e-04, 4.35842710e-05, -6.27071960e-05,-7.06018079e-05, -9.13430135e-05,-1.17567113e-04, -1.82530278e-04 ])
for i in range(len(xi)):
    ref_node = xi[i]
    B_ref = bqs.B(x_nodes2, ref_node) #x_nodes element, hardcode
    strain = np.dot(B_ref, d2)
    strain_xx2[i] = strain[0]
    strain_yy2[i] = strain[1]
    strain_xy2[i] = strain[2]
    stress = np.dot(D(0), strain)
    stress_xx2[i] = stress[0]
    stress_yy2[i] = stress[1]
    stress_xy2[i] = stress[2]

strain_xx3 = np.zeros(4)
strain_yy3 = np.zeros(4)
strain_xy3 = np.zeros(4)
stress_xx3 = np.zeros(4)
stress_yy3 = np.zeros(4)
stress_xy3 = np.zeros(4)

xi = np.array([
    [-1,-1],
    [1,-1],
    [1,1],
    [-1,1]
])
x_nodes3 = np.array([
    [0, 0],
    [8.036, 0],
    [8.036, 19.4],
    [0, 19.4]
])
d3 = np.array([-1.18230957e-06, -2.53017994e-04,  3.21030543e-05, -1.50835342e-04, -1.17567113e-04, -1.82530278e-04, -1.40579209e-04, -2.70106034e-04])
for i in range(len(xi)):
    ref_node = xi[i]
    B_ref = bqs.B(x_nodes3, ref_node) #x_nodes element, hardcode
    strain = np.dot(B_ref, d3)
    strain_xx3[i] = strain[0]
    strain_yy3[i] = strain[1]
    strain_xy3[i] = strain[2]
    stress = np.dot(D(0), strain)
    stress_xx3[i] = stress[0]
    stress_yy3[i] = stress[1]
    stress_xy3[i] = stress[2]

strain_xx4 = np.zeros(4)
strain_yy4 = np.zeros(4)
strain_xy4 = np.zeros(4)
stress_xx4 = np.zeros(4)
stress_yy4 = np.zeros(4)
stress_xy4 = np.zeros(4)

xi = np.array([
    [-1,-1],
    [1,-1],
    [1,1],
    [-1,1]
])
x_nodes4 = np.array([
    [8.036, 0],
    [16.072, 0],
    [16.072, 8.036],
    [8.036, 19.4]
])
d4 = np.array([-1.18230957e-06, -2.53017994e-04, 0.00000000e+00, -7.06018079e-05, -1.17567113e-04, -1.40579209e-04,-1.51263744e-04, -1.05475103e-04])
for i in range(len(xi)):
    ref_node = xi[i]
    B_ref = bqs.B(x_nodes4, ref_node) #x_nodes element, hardcode
    strain = np.dot(B_ref, d4)
    strain_xx4[i] = strain[0]
    strain_yy4[i] = strain[1]
    strain_xy4[i] = strain[2]
    stress = np.dot(D(0), strain)
    stress_xx4[i] = stress[0]
    stress_yy4[i] = stress[1]
    stress_xy4[i] = stress[2]


print(stress_xx1)
print(stress_xx2)
print(stress_xx3)
print(stress_xx4)

print(stress_xy1)
print(stress_xy2)
print(stress_xy3)
print(stress_xy4)

print(stress_yy1)
print(stress_yy2)
print(stress_yy3)
print(stress_yy4)