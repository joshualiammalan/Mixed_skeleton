import lib.build.quadrature as q
import numpy as np


def tri_gauss_quad(integrand, n_gp):
    return np.asarray(q.tri_gauss_quad(integrand, n_gp))


def quad_gauss_quad(integrand, n_gp):
    return np.asarray(q.quad_gauss_quad(integrand, n_gp))


def one_d_gauss_quad(integrand, n_gp):
    return np.asarray(q.one_d_gauss_quad(integrand, n_gp))
