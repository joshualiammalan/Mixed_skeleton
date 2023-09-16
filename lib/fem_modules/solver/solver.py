import numpy as np
import lib.build.solver as s


def solve(ICA: np.array,
          x_nodes: np.array,
          D: callable,
          S: callable,
          n_bcs: np.array,
          d_bcs: np.array,
          el: object):

    return np.asarray(s.solve(ICA, x_nodes, D, S, n_bcs, d_bcs, el))


def element_volume_integrations(ICA, x_nodes, n_els, n_nodes, n_nodes_per_el, el, D, S):
    return np.asarray(s.element_volume_integrations(ICA, x_nodes, n_els, n_nodes, n_nodes_per_el, el, D, S))


def neumann_bcs(x_nodes, f, n_bcs):
    return np.asarray(s.neumann_bcs(x_nodes, f, n_bcs))


def dirichlet_bcs(x_nodes, f, K, d_bcs):
    return s.dirichlet_bcs(x_nodes, f, K, d_bcs)

