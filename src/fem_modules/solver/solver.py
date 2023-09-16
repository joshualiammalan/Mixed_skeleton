"""A module containing functions solving a given FEM problem.
"""
import numpy as np
import src.fem_modules.fe_values.element_integrations as ei


def solve(ICA: np.array, x_nodes: np.array, D: callable, S:callable , n_bcs: np.array, d_bcs: np.array, el: object): #def a function which will solve the FEM problem
    n_els = len(ICA) #compute the number of elements in the mesh as the number of rows in our ICA
    n_nodes = len(x_nodes) #compute the number of nodes in the mesh as the number of rows in our x_nodes matrix
    n_nodes_per_el = len(ICA[0]) #compute the number of elements per node as the number of columns in our ICA
    K, F = element_volume_integrations(ICA, x_nodes, n_els, n_nodes, n_nodes_per_el, el, D, S) #set K and F to the values computed by the element integrations function
    F = neumann_bcs(x_nodes, F, n_bcs) #Set global F to the values recieved after neumann BCS are applied in the function neumann_bcs
    K, F = dirichlet_bcs(x_nodes, F, K, d_bcs) #Set global K and F to the values recieved after direchlet BCS are applied in the function direchlet_bcs
    d = np.linalg.solve(K,F) #compute the values of d by solving the equation Kd = F

    return d.flatten() #return the d matrix
    r"""!
    Finds \f$d\f$ that satisfies
    \f[
        \underbrace{\int_{\Omega}\boldsymbol{B}^{T}\boldsymbol{DB}\text{d}\Omega}_{\boldsymbol{K}} \boldsymbol{d} = \underbrace{\underbrace{\int_{\Omega}\boldsymbol{N}^{T}S\text{d}\Omega}_{\boldsymbol{F_b}} - \underbrace{ \int_{\Gamma}\boldsymbol{N}^{T}\boldsymbol{n}\cdot\boldsymbol{q}\text{d}\Gamma}_{\boldsymbol{F}_N}}_{\boldsymbol{F}}
    \f]
    with given Dirichlet boundary conditions imposed.
    @param ICA: Inter-connectivity array
    @param x_nodes: A \f$N_{nodes}\times 2\f$ numpy array of node positions
    @param D: A function defining the conductivity matrix as a function of position
    @param S: A function defining the source term as a function of position
    @param n_bcs: A numpy array of #boundary_conditions.NeumannBoundaryCondition objects
    @param d_bcs: A numpy array of #boundary_conditions.DirichletBoundaryCondition objects
    @param el: An element module providing el.N(), el.B(), etc functions
    @return \f$\boldsymbol{d}\f$ a numpy array of the value of the solution at the nodes
    """

def element_volume_integrations(ICA, x_nodes, n_els, n_nodes, n_nodes_per_el, el, D, S):
    r"""!
    Performs the required volume integrals over the domain, i.e
    \f[
    \int_{\Omega}\boldsymbol{B}^{T}\boldsymbol{DB}\text{d}\Omega\qquad \text{and} \qquad \int_{\Omega}\boldsymbol{N}^{T}S\text{d}\Omega
    \f]
    and returns the resulting global stiffness matrix and force vector
    @param ICA: Inter-connectivity array
    @param x_nodes: A \f$N_{nodes}\times 2\f$ numpy array of node positions
    @param n_els: Number of elements
    @param n_nodes: Number of nodes
    @param n_nodes_per_el: Number of nodes per element
    @param D: A function defining the conductivity matrix as a function of position
    @param S: A function defining the source term as a function of position
    @param el: An element module providing el.N(), el.B(), etc functions
    @return K, f
    """
    K = np.zeros((n_nodes*2, n_nodes*2)) #initialise K and F
    F = np.zeros(n_nodes*2)

    # Perform volume integrals over the domain
    for e in range(len(ICA)): #start a for loop which loops over the elements
        I = ICA[e] # Get element connectivity
        U = np.zeros(I.size * 2, dtype=int)
        U[::2] = I*2
        U[1::2] =I*2 + 1
        x_e = x_nodes[I] # Get element nodal positions
        # Compute element stiffness matrix and force vector using the function K_el and f_b in the element_integrations file
        Ke = ei.K_el(x_e, el, D, ngp=3)
        fe = ei.f_b(x_e,el,S,ngp=3)
        K[U[:, None], U] += Ke #assemble Ke to global K
        F[U] += fe.reshape(-1) #assemble fe to global f
    return K, F

def neumann_bcs(x_nodes, f, n_bcs):
    r"""!
    Adds force terms due to a Neumann boundary condition to a global force vector. It should be called as to update the
    force vector, i.e.
    f = neumann_bcs(x_nodes, f, n_bcs)
    @param x_nodes: A \f$N_{nodes}\times 2\f$ numpy array of node positions
    @param f: Global force array to add Neumann boundary condition force contributions to
    @param n_bcs: A numpy array of #boundary_conditions.NeumannBoundaryCondition objects
    @return f, the adjusted global force vector
    """
    for bc in n_bcs:
        edges_nodes = bc.edges_nodes #obtain the nodal values of the neumann bcs from n_bcs
        edges_normals = bc.edges_normals #obtain the normals of the neumann bcs from n_bcs
        for i in range(len(edges_nodes)):
            n1 = edges_nodes[i][0] #obtain the first global node corresponding to the edge
            n2 = edges_nodes[i][1] #obtain the second global node corresponding to the edge
            U = np.zeros(4, dtype=int)
            U[::2] = n1*2,n2*2
            U[1::2] = n1*2+1,n2*2+1
            normal = edges_normals[i] #obtain the normal for the edge
            edge_nodes = x_nodes[[n1, n2]] #retrieve the corresponding nodal positions from x_nodes array
            q = bc.q #assign q to the value specified in n_bcs
            force = ei.f_gamma(edge_nodes, normal, q) #calculate the force contribution from neumann bc using f_gamma function
            f[U[0]] += force[0] #subtract the force contributions from neumann bc to the corresponding node indices
            f[U[1]] += force[1]
            f[U[2]] += force[2]
            f[U[3]] += force[3]
    return f

def dirichlet_bcs(x_nodes, f, K, d_bcs):
    for bc in d_bcs:
        nodes = bc.nodes #obtain the nodal values of the neumann bcs from n_bcs
        extended_nodes = np.repeat(x_nodes,2,axis=0)
        for node in nodes:
            positions = extended_nodes[node] #obtain the nodal coordinate positions
            d = bc.f(positions) #assign the value of d to the funtion value assigned in the bc.f
            f -= d* K[:, node] # Adjust force vector
            f[node] = d  # Adjust force vector for nodes in Dirichlet BC
            K[node, :] = 0  # Set row of K to zero
            K[:, node] = 0  # Set column of K to zero
            K[node, node] = 1  # Set diagonal entry of K to 1
    return K, f.reshape(-1,1) #return adjusted K and f
    r"""!
    A function that adjusts the global stiffness matrix and force vector in order to apply the Dirichlet boundary conditions.
    It loops over each Dirichlet BC object and loops each node in each Dirichlet BC object. For each node \f$i\f$ it
    applies the following:
    \f[
        f = f - d\_bc.f(x\_node)\boldsymbol{K}_i\,, \qquad
        \boldsymbol{K} = \begin{bmatrix} k_{11} & ... & 0 & ... & k_{1n} \\ \vdots & \ddots & 0 & \ddots & \vdots \\ 0 & ... & 1 & ... & 0  \\ \vdots & \ddots & 0 & \ddots & \vdots \\  k_{n1} & ... & 0 & ... & k_{nn} \end{bmatrix}
    \f]
    where \f$\boldsymbol{K}_i\f$ is the \f$i^{th}\f$ column of the global stiffness matrix, \f$d\_bc.f\f$ is the function
    provided by the Dirichlet BC object that defines the value at the boundary as a function of position and \f$x\_node\f$
    is the position of the $i^{th}$ node. It then loops over each Dirichlet BC object and loops each node in each Dirichlet BC object
    again and adjusts the force array as follows
    \f[
    f_i = d\_bc.f(x\_node)
    \f]
    Finally, it returns the adjusted global stiffness matrix and force array.
    It should be called as to update these values, i.e. K, f = dirichlet_bcs(x_nodes, f, K, d_bcs)
    @param x_nodes: A \f$N_{nodes}\times 2\f$ numpy array of node positions
    @param f: Global force array to adjust
    @param K: Global stiffness matrix to adjust
    @param d_bcs: A numpy array of #boundary_conditions.DirichletBoundaryCondition objects
    @return Adjusted K and f
    """

def solve_displacement_pressure(ICA: np.array, x_nodes: np.array, D: callable, S: callable, n_bcs: np.array, d_bcs: np.array, el: object, lambda_val: float):
    n_els = len(ICA)  # Number of elements in the mesh
    n_nodes = len(x_nodes)  # Number of nodes in the mesh
    n_nodes_per_el = len(ICA[0])  # Number of nodes per element

    # Calculate K and F matrices as before
    K, F = element_volume_integrations(ICA, x_nodes, n_els, n_nodes, n_nodes_per_el, el, D, S)

    # Apply Neumann boundary conditions to F
    F = neumann_bcs(x_nodes, F, n_bcs)

    # Apply Dirichlet boundary conditions to K and F
    K, F = dirichlet_bcs(x_nodes, F, K, d_bcs)

    # Calculate the global G matrix
    G = ei.G(x_nodes, el, ngp=3)

    # Calculate the global M matrix
    M = ei.M(x_nodes, el, lambda_val, ngp=3)

    # Calculate K_hat using the formula: K - G * M^-1 * G^T
    K_hat = K - np.dot(G, np.dot(np.linalg.inv(M), G.T))

    # Solve for displacement d
    d = np.linalg.solve(K_hat, F)

    # Calculate pressure p using p = -M^-1 * G^T * d
    p = -np.dot(np.dot(np.linalg.inv(M), G.T), d)

    return d.flatten(), p.flatten()


