import numpy as np
import lib.build.linear_scalar as lts


def N(xi: np.array):
    return np.asarray(lts.N(xi))


def GN(xi: np.array):
    return np.asarray(lts.GN(xi))


def J(x_nodes: np.array, xi: np.array):
    return np.asarray(lts.J(x_nodes, xi))


def B(x_nodes: np.array, xi: np.array):
    return np.asarray(lts.B(x_nodes, xi))


def xi_to_x(x_nodes: np.array, xi: np.array):
    return np.asarray(lts.xi_to_x(x_nodes, xi))


def gauss_quad(x_nodes: np.array, integrand: callable, n_gp: int):
    return np.asarray(lts.gauss_quad(x_nodes, integrand, n_gp))

