"""A module for plotting functions
"""
import numpy as np
import matplotlib.tri as mtri
import src.fem_modules.elements2d.quads.bilinear_vector as bsq
from matplotlib import cm, pyplot as plt


def plot_mesh(plt_obj, ICA, x_nodes, el_numbers=True, node_numbers=True):
    r"""!
    A function that adds the outline of a mesh to a plot
    @param plt_obj: A matplotlib.pyplot module
    @param ICA: The mesh inter-connectivity array
    @param x_nodes: An \f$N_{nodes}\times 2\f$ numpy array of node positions
    @param el_numbers: A boolean that activates element number plotting if true
    @param node_numbers: A boolean that activates node number plotting if true
    """
    n_els = ICA.shape[0]
    n_nodes = x_nodes.shape[0]

    for i_el in range(n_els):
        el_nodes = ICA[i_el, :]
        x_el_nodes = x_nodes[el_nodes, :].copy()
        cent = np.mean(x_el_nodes, axis=0)
        x_el_nodes = np.vstack((x_el_nodes, x_el_nodes[0, :]))
        plt_obj.plot(x_el_nodes[:, 0], x_el_nodes[:, 1], color='black')
        if el_numbers:
            plt_obj.text(cent[0], cent[1], "["+str(i_el)+"]", color='black')
        # plt.show()

    if node_numbers:
        for i_node in range(n_nodes):
            x = x_nodes[i_node, :]
            plt_obj.text(x[0], x[1], "("+str(i_node)+")", color='red')


def plot_tri_result(plt_obj, ICA, x_nodes, d, n_contours=200):
    r"""!
    Adds a contour plot of a solution on a triangular mesh to a plot
    @param plt_obj: A matplotlib.pyplot module
    @param ICA: The mesh inter-connectivity array
    @param x_nodes: An \f$N_{nodes}\times 2\f$ numpy array of node positions
    @param d: A size \f$N_{nodes}\f$ numpy array of the solution at the nodes
    @param n_contours: The number of contours to be used in the plot
    """
    d_max, d_min = np.max(d), np.min(d)

    triang = mtri.Triangulation(x_nodes[:, 0], x_nodes[:, 1], ICA)
    plot = plt_obj.tricontourf(triang, d, np.linspace(d_min, d_max, n_contours), cmap='jet')
    plt_obj.colorbar(plot)


def plot_quad_result(plt_obj, ICA, x_nodes, d, res=100, n_contours=500):
    r"""!
    Adds a contour plot of a solution on a quadrilateral mesh to a plot
    @param plt_obj: A matplotlib.pyplot module
    @param ICA: The mesh inter-connectivity array
    @param x_nodes: An \f$N_{nodes}\times 2\f$ numpy array of node positions
    @param d: A size \f$N_{nodes}\f$ numpy array of the solution at the nodes
    @param res: The resolution of the plot per element
    @param n_contours: The number of contours to be used in the plot
    """
    d_max, d_min = np.max(d), np.min(d)
    n_els = ICA.shape[0]
    xi = np.linspace(-1, 1, res)
    eta = np.linspace(-1, 1, res)
    xi, eta = np.meshgrid(xi, eta)
    x = np.zeros(shape=xi.shape)
    y = np.zeros(shape=xi.shape)
    T = np.zeros(shape=xi.shape)
    for i_el in range(n_els):
        el_nodes = ICA[i_el, :]
        x_nodes_el = x_nodes[el_nodes, :]
        d_el = d[el_nodes]
        for x_i in range(res):
            for y_i in range(res):
                xi_val = np.array([xi[x_i, y_i], eta[x_i, y_i]])
                pos = bsq.xi_to_x(x_nodes_el, xi_val)
                x[x_i, y_i] = pos[0]
                y[x_i, y_i] = pos[1]
                T[x_i, y_i] = np.dot(bsq.N(xi_val), d_el)

        plt_obj.contourf(x, y, T, np.linspace(d_min, d_max, n_contours), vmax=d_max, vmin=d_min, cmap='jet')

    plt_obj.colorbar()

def plot_tri_result_elementwise(plt_obj, ICA, x_nodes, d_el_array, n_contours=200):
    r"""!
    Adds a contour plot of a solution on a triangular mesh to a plot
    @param plt_obj: A matplotlib.pyplot module
    @param ICA: The mesh inter-connectivity array
    @param x_nodes: A [nr-of-nodes by 2]  numpy array of node positions
    @param d_el_array: A [nr-of-elements by 3] numpy array of the quantity at the element nodes
    @param n_contours: The number of contours to be used in the plot
    """
    # find min and max of 1st el
    d_max = np.max(d_el_array[0])
    d_min = np.min(d_el_array[0])

    # find global min and max
    for d_el in d_el_array:

        # element min and max
        el_d_max = np.max(d_el)
        el_d_min = np.min(d_el)

        # if element max bigger than global max, global max becomes element max
        if el_d_max > d_max:
            d_max = el_d_max

        # if element min smaller than global min, global min becomes element min
        if el_d_min < d_min:
            d_min = el_d_min

    # plot solution on each element
    for el_nr, d_el in enumerate(d_el_array):
        el_ICA = np.array([list(ICA[el_nr])])
        el_nodes = x_nodes[el_ICA, :][0]
        x = np.array(el_nodes[:, 0])
        y = np.array(el_nodes[:, 1])
        triang = mtri.Triangulation(x, y, [[0, 1, 2]])
        plot = plt_obj.tricontourf(triang, d_el, levels=n_contours, cmap='jet', vmin=d_min, vmax=d_max)

    plt_obj.colorbar(cm.ScalarMappable(norm=plt.Normalize(d_min, d_max), cmap='jet'))