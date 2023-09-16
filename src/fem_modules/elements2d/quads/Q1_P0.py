import numpy as np
import src.fem_modules.utils.quadrature as q

def N(xi):
    xi1, xi2 = xi
    return np.array([  #computation of the isoparametric shape functions for a bilinear scalar element in terms of xi and eta
        (1 - xi1) * (1 - xi2) / 4,
        (1 + xi1) * (1 - xi2) / 4,
        (1 + xi1) * (1 + xi2) / 4,
        (1 - xi1) * (1 + xi2) / 4
    ])
    r"""!
    Defines the shape functions in terms of \f$ \xi \f$ and \f$ \eta \f$.
    @param xi: A numpy array of size 2 with values [\f$ \xi \f$, \f$ \eta \f$].
    @return A numpy array of size 4 containing the value of the shape functions for each node, \f$ \boldsymbol{N} \f$.
    """

def N_extended(xi):
    xi1, xi2 = xi
    return np.array([[(1 - xi1) * (1 - xi2) / 4, 0, (1 + xi1) * (1 - xi2) / 4, 0, (1 + xi1) * (1 + xi2) / 4, 0, (1 - xi1) * (1 + xi2) / 4, 0],
                    [0, (1 - xi1) * (1 - xi2) / 4, 0, (1 + xi1) * (1 - xi2) / 4, 0, (1 + xi1) * (1 + xi2) / 4, 0, (1 - xi1) * (1 + xi2) / 4]])  # computation of the isoparametric shape functions for a linear scalar element in terms of xi and eta


def GN(xi):
    xi1, xi2 = xi
    return np.array([  #computation of the derivative isoparametric shape functions for a bilinear scalar element in terms of xi and eta
        [(-1 + xi2) / 4, (1 - xi2) / 4, (1 + xi2) / 4, (-1 - xi2) / 4],
        [(-1 + xi1) / 4, (-1 - xi1) / 4, (1 + xi1) / 4, (1 - xi1) / 4]
    ])
    r"""!
    Defines the gradient of the shape functions in terms of \f$ \xi \f$ and \f$ \eta \f$:
    \f[
    \boldsymbol{GN} = \left[ \begin{matrix} \frac{\partial \boldsymbol{N}}{\partial \xi} \\ \frac{\partial \boldsymbol{N}}{\partial \eta} \end{matrix} \right]
    \f]
    @param xi: A numpy array of size 2 with values [\f$ \xi \f$, \f$ \eta \f$ ].
    @return A numpy array of size 2x4 containing the value of the shape function derivatives.
    """

def J(x_nodes, xi):
    return np.dot(GN(xi), x_nodes)  #computation of the Jacobian matrix
    r"""!
    Determines the Jacobian matrix for the element
    \f[
        \boldsymbol{J} = \begin{bmatrix} \frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\  \frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta}  \end{bmatrix}
    \f]
    @param x_nodes: A 4x2 numpy of node positions for the element:
    | Local \n Node num | x | y |
    |-|-|-|
    |1|x1|y1|
    |2|x2|y2|
    |3|x3|y3|
    |4|x4|y4|
    @param xi: A numpy array of size 2 with values [\f$ \xi \f$, \f$ \eta \f$ ].
    @return \f$ \begin{bmatrix} \frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\  \frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta}  \end{bmatrix} \f$
    """

def B(x_nodes, xi):
    inv_J = np.linalg.inv(J(x_nodes, xi))  #calculating the inverse of the Jacobian matrix
    P =  np.dot(inv_J, GN(xi))  #calculating the B matrix
    B = np.array([
        [P[0,0], 0, P[0,1], 0, P[0,2], 0, P[0,3], 0],
        [0, P[1,0], 0, P[1,1], 0, P[1,2], 0, P[1,3]],
        [P[1,0], P[0,0], P[1,1], P[0,1], P[1,2], P[0,2], P[1,3], P[0,3]]
    ])
    return B
    r"""!
    Defines the gradient of the shape functions in terms of \f$ x \f$ and \f$ y \f$:
    \f[
    \boldsymbol{B} = \left[ \begin{matrix} \frac{\partial \boldsymbol{N}}{\partial x} \\ \frac{\partial \boldsymbol{N}}{\partial y} \end{matrix} \right]
    \f]
    Hint: You'll need #J and #GN.
    @param x_nodes: A 4x2 numpy of node positions for the element:
    | Local \n Node num | x | y |
    |-|-|-|
    |1|x1|y1|
    |2|x2|y2|
    |3|x3|y3|
    |4|x4|y4|
    @param xi: A numpy array of size 2 with values [\f$ \xi \f$, \f$ \eta \f$ ].
    @return A numpy array of size 2x4 containing the value of the shape function derivatives.
    """

def xi_to_x(x_nodes, xi):
    return np.dot(N(xi), x_nodes)  #converting x and y to xi and eta by multiplying isoparametric shape functions into nodal positions
    r"""!
    Determines the [\f$x,y\f$] position in an element for a given [\f$\xi,\eta\f$] position.
    @param x_nodes: A 4x2 numpy of node positions for the element:
    | Local \n Node num | x | y |
    |-|-|-|
    |1|x1|y1|
    |2|x2|y2|
    |3|x3|y3|
    |4|x4|y4|
    @param xi: A numpy array of size 2 with values [\f$ \xi \f$, \f$ \eta \f$ ].
    @return A numpy array of size 2 containing the positions [\f$x,y\f$]
    """


def Pressure_N(xi):
    return 1

def L(x_nodes, xi):
    one = np.array([1,1,0])
    L = np.dot(B(x_nodes,xi),one)
    return L

def gauss_quad(x_nodes, integrand, n_gp):
    r"""!
    A function that multiplies an integrand by the Jacobian and applies a Gauss Quadrature for a quadrilateral.
    @param x_nodes: A 4x2 numpy of node positions for the element:
    | Local \n Node num | x | y |
    |-|-|-|
    |1|x1|y1|
    |2|x2|y2|
    |3|x3|y3|
    |4|x4|y4|
    @param integrand: Function in terms of \f$\xi\f$ and \f$\eta\f$ to be integrated
    @param n_gp: Number of Gauss points
    @return: The integrated function
    """
    def Jacobian(xi): #define a new function for the Jacobian ito xi
        J_xi = J(x_nodes, xi)
        det_J_xi = np.linalg.det(J_xi) #determinant of J ito xi
        return det_J_xi
    def integrand_with_J(xi): #Create new integrand function ito xi which includes Jacobian
        return Jacobian(xi) * integrand(xi)
    return q.quad_gauss_quad(integrand_with_J, n_gp) #perform gauss quadrature