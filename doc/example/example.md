Example {#example}
=======

In order to illustrate how this 2D scalar FEM library should be used an example 
problem is given here. Consider the diagram below

<img src="..\example\project-example.png" alt="drawing" width="300"/>

The diagram illustrates a temperature conduction problem with the governing 
equation
\f[
    \nabla (-\boldsymbol{D}\nabla T) - S = 0\,,
\f] 
where \f$T\f$ is the temperature, \f$\boldsymbol{D}\f$ is the conductivity matrix and \f$S\f$ is a source term.
In this problem the conductivity matrix and the source term are given by
\f[
    \boldsymbol{D}=\boldsymbol{I}\,,\qquad S=-5(x^2 + y^2)\,,
\f]
respectively. Additionally, the following boundary conditions hold
\f[
\overline{T}_1 = 0
\f]
\f[
\overline{T}_2 = 5y^2
\f]
\f[
\overline{q}_1 = 2\begin{bmatrix}x & y\end{bmatrix}^T
\f]
\f[
\overline{q}_2 = \begin{bmatrix}10 & 0\end{bmatrix}^T
\f]
The problem was solved on a 4 element bilinear quadrilateral mesh in #example_quads and 
an 8 element linear triangle mesh in #example_tris. The resulting solutions are displayed below

<img src="..\example\quad_solution.png" alt="drawing" width="300"/><img src="..\example\tri_solution.png" alt="drawing" width="300"/>