import lib.build.element_integrations as ei
import numpy as np


def K_el(x_nodes,
         el,
         D=lambda x: np.eye(3, 3),
         ngp=3):
    return np.asarray(ei.K_el(x_nodes,
         el,
         D,
         ngp))


def f_b(x_nodes,
        el,
        S=lambda x: 1,
        ngp=3):
    return np.asarray(ei.f_b(x_nodes,
        el,
        S,
        ngp))


def scalar_xi(x_nodes,
       el,
       S=lambda xi: 1,
       ngp=3):

    return np.asarray(ei.scalar_xi(x_nodes,
        el,
        S,
        ngp))


def scalar_x(x_nodes,
       el,
       S=lambda xi: 1,
       ngp=3):

    return np.asarray(ei.scalar_x(x_nodes,
        el,
        S,
        ngp))


def f_gamma(x_nodes, n, q, ngp=3):

    return np.asarray(ei.f_gamma(x_nodes, n, q, ngp))

