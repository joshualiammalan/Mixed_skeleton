"""A module containing functions determining commonly used FEM values.
"""
import numpy as np
import src.fem_modules.utils.quadrature as quad



def K_el(x_nodes, el, D=lambda x: np.eye(2, 2), ngp=3):
    def integrand(xi): #define an integrand for the calculation of K_el ito the B matrix calculated in el and D ito xi and eta
        B = el.B(x_nodes, xi)
        D_xi_eta = D(el.xi_to_x(x_nodes, xi)) #use el.xi to x to convert D to the isoparametric domain
        return np.dot(B.T, np.dot(D_xi_eta, B)) #compute the integrand for the calculation of K_el
    return el.gauss_quad(x_nodes, integrand, ngp)
    #r"""!
    #Performs the integration over the element:
    #\f[
    #    \boldsymbol{K}_{el} = \int_{\Omega^{el}}\boldsymbol{B}^{T}\boldsymbol{DB}\text{d}\Omega
    #\f]
    #@param x_nodes: An \f$N_{en}\times 2\f$ numpy of node positions for the element where \f$ N_{en} \f$ is the number of elemental nodes.
    #@param el: An element module providing the functions el.B() and el.gauss_quad().
    #@param D: A function that takes in a size 2 numpy array representing a position [ \f$ x,y\f$ ] and returns a \f$2\times2\f$ numpy array.
    #@param ngp: The number of Gauss Points to be used for the quadrature.
    #@return \f$ \int_{\Omega^{el}}\boldsymbol{B}^{T}\boldsymbol{DB}\text{d}\Omega\f$, an \f$N_{en} \times N_{en}\f$ numpy array.
    #"""

def f_b(x_nodes,el,S=lambda x: np.array([0, 0]), ngp=3):
    def integrand(xi):#define an integrand for the calculation of f_b ito the N matrix calculated in el and the source term ito xi and eta
        N = el.N_extended(xi)
        S_xi_eta = S(el.xi_to_x(x_nodes, xi))
        return np.dot(N.T, S_xi_eta) #compute the integrand for the calculation of f_b

    return el.gauss_quad(x_nodes, integrand, ngp)
    r"""!
    Performs the integration over the element:
    \f[
        \boldsymbol{f}_b = \int_{\Omega^{el}}\boldsymbol{N}^{T}S\text{d}\Omega
    \f]
    @param x_nodes: An \f$N_{en}\times 2\f$ numpy array of node positions for the element where \f$ N_{en} \f$ is the number of elemental nodes.
    @param el: An element module providing the functions el.N() and el.gauss_quad().
    @param S: A function that takes in a size 2 numpy array representing a position [ \f$ x,y\f$ ] and returns a scalar value.
    @param ngp: The number of Gauss Points to be used for the quadrature.
    @return \f$ \int_{\Omega^{el}}\boldsymbol{N}^{T}S\text{d}\Omega \f$, a numpy array of size \f$N_{en}\f$
    """

def f_gamma(x_nodes, n, t, ngp=3):
    diff = (x_nodes[1]-x_nodes[0])/2 #calculate the length of an edge
    mid = (x_nodes[0]+x_nodes[1])/2 #calculate the midpoint of an edge
    J = np.linalg.norm(diff) #calculate the Jacobian as the magnitude\length of the edge
    def integrand(xi): #compute a new integrand in terms of a 1D shape function of a line and an x position ito the length and midpoint
        N = np.array([[(1-xi)/2,0,(1+xi)/2,0],
                      [0,(1-xi)/2,0,(1+xi)/2]
                     ])
        x = diff*xi + mid
        return N.T.dot(np.dot(t(x),n.T))*J
    return (quad.one_d_gauss_quad(integrand,ngp)).reshape(-1,1)
    r"""!
    Performs the integration over the element:
    \f[
        \boldsymbol{f}_\Gamma = \int_{\Gamma}\boldsymbol{N}^{T}\boldsymbol{n}\cdot\boldsymbol{q}\text{d}\Gamma
    \f]
    @param x_nodes: A \f$2 \times 2\f$ numpy array of node positions at either end of the edge.
    @param n: A size 2 numpy array of unit length representing the outward facing normal of the edge
    @param q: A function returning a size 2 numpy array representing the flux on the edge. The function takes in a size 2 numpy array representing position in terms of [\f$x,y\f$]
    @param ngp: The number of Gauss Points to be used for the quadrature.
    @return \f$ \int_{\Gamma}\boldsymbol{N}^{T}\boldsymbol{n}\cdot\boldsymbol{q}\text{d}\Gamma \f$, a numpy array of size \f$ 2 \f$
    """


def G(x_nodes, el, ngp=3):
    def integrand(xi):
        L = el.L(x_nodes, xi)
        Pressure_N = el.Pressure_N(xi)
        return -np.outer(L, Pressure_N)  # Compute the integrand for the G matrix

    return el.gauss_quad(x_nodes, integrand, ngp)

def M(x_nodes, el, lambda_val, ngp=3):
    def integrand(xi):
        Pressure_N = el.Pressure_N(xi)
        return -1/lambda_val * np.outer(Pressure_N, Pressure_N)  # Compute the integrand for the M matrix

    return el.gauss_quad(x_nodes, integrand, ngp)


