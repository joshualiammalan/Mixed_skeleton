import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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
