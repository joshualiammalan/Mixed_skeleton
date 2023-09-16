"""Solves the problem outlined in the Example page using 4 bilinear quadrilateral elements.
"""

import src.fem_modules.utils.plotting as plotting
import src.fem_modules.elements2d.quads.bilinear_vector as bqs
import src.fem_modules.solver.solver as s
import src.fem_modules.utils.boundary_conditions as bcs
import numpy as np
import matplotlib.pyplot as plt

#Solution file for the Quadrilateral element mesh.

x_nodes = np.array([ #define global node positions
    [0, 1],
    [0, 0],
    [2, 0.5],
    [2, 1],
])

## Inter-connectivity array
ICA = np.array([ #define ICA
    [1, 2, 3, 4], #element global nodes
    # [4, 8, 10, 7],
    # [2, 3, 6, 5],
    # [2, 5, 9, 8],
    # [8, 9, 11, 10],
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

## A function defining the conductivity matrix
def D(x):
    return np.array([
        [1,0.3,0],
        [0.3,1,0],
        [0,0,0.35]
    ])*3.3*10**7

## A function defining the body force term
def S(x):
    return np.array([0,0])

## A function defining the dirichlet condition on the appropriate nodes
def d_f1(x):
    return 0

## The Dirichlet condition on the left in the x direction,nodes are according to global node numbering
d_left = bcs.DirichletBoundaryCondition(f=d_f1, nodes=np.array([0, 1, 2, 3]))

d_bcs = np.array([d_left]) #matrix containing all direchlet bcs functions and nodal positions

## A function defining the neumann condition on the top of the plate
def t1(x): #traction
    return np.array([[0,0],
                    [0,-20]])

# The Neumann condition on the top
n_top = bcs.NeumannBoundaryCondition(q=t1,
                                 edges_nodes=np.array([
                                     [3, 0],
                                     # [9, 10],
                                 ]),
                                 edges_normals=np.array([
                                     [0, 1],
                                     # [0, 1],
                                 ]))

## A function defining the neumann condition on the top of the plate
# def q2(x):
#     return np.array([3*(x[1]**2),0])
#
# ## The Neumann condition on the left
# n_left = bcs.NeumannBoundaryCondition(q=q2,
#                              edges_nodes=np.array([
#                                  [0, 3],
#                                  [3, 6],
#                              ]),
#                              edges_normals=np.array([
#                                  [-1, 0],
#                                  [-1, 0],
#                              ]))
#
# ## A function defining the neumann condition on the right of the plate
# def q3(x):
#     return np.array([0,0])
#
# ## The Neumann condition on the right
# n_right = bcs.NeumannBoundaryCondition(q=q3,
#                                  edges_nodes=np.array([
#                                      [2, 5],
#                                  ]),
#                                  edges_normals=np.array([
#                                      [1, 0],
#                                  ]))
#
# ## A function defining the neumann condition on the circular edge of the plate
# def q4(x):
#     return np.array([0,0])

# calculating normals for circular edges:
# nodes = [
#     [(7, 4 - math.sqrt(3)), (8, 2)],  # Edge 1
#     [(8 - math.sqrt(3), 3), (7, 4 - math.sqrt(3))],  # Edge 2
#     [(6, 4), (8 - math.sqrt(3), 3)]  # Edge 3
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

## The Neumann edges and normals on the circular edge
# n_circle = bcs.NeumannBoundaryCondition(q=q4,
#                                  edges_nodes=np.array([
#                                      [5, 4],
#                                      [4, 8],
#                                      [8, 10],
#                                  ]),
#                                  edges_normals=np.array([
#                                      [0.2588190451025209, 0.9659258262890684],
#                                      [0.7071067811865478, 0.7071067811865474],
#                                      [0.9659258262890682, 0.25881904510252124],
#                                  ]))

n_bcs = np.array([n_top]) #matrix containing all neumann bcs functions, normals and nodal positions

## The value of the solution at the nodes
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
plt.figure()
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

#plotting of temperature and flux along an edge
# def plot_temperature_and_flux_along_line(plt, ICA, x_nodes, d, lts, x1, y1, x2, y2, x3, y3):
#     line_nodes = []
#     line_values_temperature = []
#     line_values_flux = []
#     for element_nodes in ICA:
#         for node in element_nodes:
#             x, y = x_nodes[node]
#             if x1 <= x <= x3 and y1 <= y <= y3:
#                 line_nodes.append(node)
#                 line_values_temperature.append(d[node])
#                 line_values_flux.append(q1(x))
#
#     plt.figure()
#     plt.subplot(2, 1, 1)
#     plt.plot(line_nodes, line_values_temperature, marker='o')
#     plt.xlabel('Node')
#     plt.ylabel('Temperature')
#     plt.title('Temperature Along the Line')
#
#     plt.subplot(2, 1, 2)
#     plt.plot(line_nodes, line_values_flux, marker='o')
#     plt.xlabel('Node')
#     plt.ylabel('Flux')
#     plt.title('Flux Along the Line')
#
#     plt.tight_layout()
#     plt.show()
#
# # Define the line endpoints nodal positions for the required line
# x1, y1 = 0, 0
# x2, y2 = 2, 0
# x3, y3 = 4, 0

# plot_temperature_and_flux_along_line(plt, ICA, x_nodes, d, lts, x1, y1, x2, y2, x3, y3)
