import lib.build.bilinear_scalar as bqs
import numpy as np


def N(xi: np.array):
    return np.asarray(bqs.N(xi))


def GN(xi: np.array):
    return np.asarray(bqs.GN(xi))


def J(x_nodes: np.array, xi: np.array):
    return np.asarray(bqs.J(x_nodes, xi))


def B(x_nodes: np.array, xi: np.array):
    return np.asarray(bqs.B(x_nodes, xi))


def xi_to_x(x_nodes: np.array, xi: np.array):
    return np.asarray(bqs.xi_to_x(x_nodes, xi))


def gauss_quad(x_nodes: np.array, integrand: callable, n_gp: int):
    return np.asarray(bqs.gauss_quad(x_nodes, integrand, n_gp))

