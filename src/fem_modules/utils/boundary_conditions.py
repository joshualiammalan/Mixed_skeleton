"""A module containing boundary condition data types.
"""
import numpy as np


class DirichletBoundaryCondition:
    r"""!
    A class for storing information about a Dirichlet boundary condition.
    """

    def __init__(self, f, nodes):
        r"""!
        A constructor.

        @param f: A function taking a size 2 numpy array representing [\f$x,y\f$] that returns the prescribed value at that position.
        @param nodes: A numpy array of global node numbers to which the Dirichlet BC applies.
        """

        ## A function taking a size 2 numpy array representing [\f$x,y\f$] that returns the prescribed value at that position
        self.f = f
        ## A numpy array of global node numbers to which the Dirichlet BC applies
        self.nodes = nodes


class NeumannBoundaryCondition:
    r"""!
    A class for storing information about a Neumann boundary condition.
    """

    def __init__(self, q, edges_nodes, edges_normals):
        r"""!
        A constructor.
        @param q: A function taking a size 2 numpy array representing [\f$x,y\f$] that returns the prescribed flux at that position as a size 2 numpy array
        @param edges_nodes: An \f$n\times 2\f$ numpy array containing the global node numbers of the nodes on each side of the edge
        @param edges_normals: An \f$n\times 2\f$ numpy array representing the outward facing unit normal of each edge
        """
        ## A function taking a size 2 numpy array representing [\f$x,y\f$] that returns the prescribed flux at that position as a size 2 numpy array
        self.q = q
        ## An \f$n\times 2\f$ numpy array containing the global node numbers of the nodes on each side of the edge
        self.edges_nodes = edges_nodes
        ## An \f$n\times 2\f$ numpy array representing the outward facing unit normal of each edge
        self.edges_normals = edges_normals


