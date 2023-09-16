"""A module containing functions for Gaussian Quadrature
"""
import numpy as np
#gauss quad constants dictionary
GQ_W_1D =  {
    1: np.array([2.]),
    2: np.array([1., 1.]),
    3: np.array([5./9, 8./9, 5./9]),
    5: np.array([0.568888888888889, 0.478628670499366, 0.478628670499366, 0.236926885056189, 0.236926885056189])
}

GQ_W_TRI = {
     1: np.array([1]),
     3: np.array([1., 1., 1.])/3.,
     4: np.array([-27., 25., 25., 25.])/48.
 }

GQ_XI_1D = {
     1: np.array([1.]),
     2: np.array([-1., 1.])/(3.**0.5),
     3: np.array([-3.**0.5, 0, 3.**0.5]) / (5**0.5),
     5: np.array([0.0, -0.538469310105683, 0.538469310105683, -0.906179845938664, 0.906179845938664])
 }

GQ_XI_TRI = {
     1: np.array([
         [1., 1.]
                  ])/3.,
     3: np.array([
         [1./6., 1./6.],
         [2./3., 1./6.],
         [1./6., 2./3.],
     ]),
     4: np.array([
         [1./3., 1./3.],
         [2./10., 2./10.],
         [6./10., 2./10.],
         [2./10., 6./10.],
     ])
 }

def tri_gauss_quad(integrand, n_gp):
    result = 0 #initialise result
    xi, w = GQ_XI_TRI[n_gp], GQ_W_TRI[n_gp] #define xi and w using dictionary
    for i in range(n_gp): #loop over ngp
        result += w[i] * integrand(xi[i]) #multiply weighting by integrand ito xi and eta to perform traingular gauss quad
    return result
    r"""!
    Performs a Gauss quadrature over a triangular domain. Note that the integrand must be defined in terms of \f$\xi\f$
    and \f$\eta\f$ and it must include the Jacobian.
    @param integrand: A function including a Jocbian to be integrated taking a size 2 numpy array representing [\f$xi,\eta\f$]
    @param n_gp: The number of Gauss points to be used
    @return The integrated function
    """

def quad_gauss_quad(integrand, n_gp):
    r"""!
    Performs a Gauss quadrature over a quadrilateral domain. Note that the integrand must be defined in terms of \f$\xi\f$
    and \f$\eta\f$ and it must include the Jacobian.
    @param integrand: A function including a Jocbian to be integrated taking a size 2 numpy array representing [\f$xi,\eta\f$]
    @param n_gp: The number of Gauss points to be used
    @return The integrated function
    """
    # Initialize result
    result = 0
    # Get Gauss points and weights
    xi, w = GQ_XI_1D[n_gp], GQ_W_1D[n_gp]
    # Perform Gauss quadrature
    for i in range(n_gp):
        for j in range(n_gp):
            # Update result
            result += w[i] * w[j] * integrand([xi[i], xi[j]])
    return result

def one_d_gauss_quad(integrand, n_gp):
    r"""!
    Performs a Gauss quadrature over a quadrilateral domain. Note that the integrand must be defined in terms of \f$\xi\f$
    and it must include the Jacobian.
    @param integrand: A function including a Jocbian to be integrated taking a float representing \f$xi\f$
    @param n_gp: The number of Gauss points to be used
    @return The integrated function
    """
    # Get Gauss points and weights
    xi = GQ_XI_1D[n_gp]
    w = GQ_W_1D[n_gp]
    # Initialize result
    result = 0
    # Perform Gauss quadrature
    for i in range(n_gp):
        # Update result
        result += w[i] * integrand(xi[i])
    return result



