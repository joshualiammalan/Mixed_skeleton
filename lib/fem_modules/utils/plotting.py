import lib.build.plotting as plting


def plot_mesh(plt_obj, ICA, x_nodes, el_numbers=True, node_numbers=True):
    plting.plot_mesh(plt_obj, ICA, x_nodes, el_numbers, node_numbers)


def plot_tri_result(plt_obj, ICA, x_nodes, d, el, res=10, el_numbers=True, node_numbers=True, n_contours=15):
    plting.plot_tri_result(plt_obj, ICA, x_nodes, d, n_contours)


def plot_quad_result(plt_obj, ICA, x_nodes, d, res=10, n_contours=15):
    plting.plot_quad_result(plt_obj, ICA, x_nodes, d, res, n_contours)
