# Mathematical Formalism

This document gives the mathematical formulation of the GWM-Pilot
architecture. It is self-contained for a reader familiar with Lie algebras
and Clifford algebras. For a descriptive overview, see
[`overview.md`](overview.md). For the pilot's central finding — that the
gauge update is strictly local — see
[`../findings/architectural_limit.md`](../findings/architectural_limit.md).

## 1. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| $v, v'$ | Voxel indices | — |
| $\mu \in \{x, y, z\}$ | Spatial direction | — |
| $M_v$ | Multivector at voxel $v$ | $8$ |
| $h_v$ | Semantic fiber at voxel $v$ | $512$ |
| $A_{v,\mu}$ | Gauge field at voxel $v$, direction $\mu$ | $10 \times 10$ |
| $p_v$ | Momentum conjugate to $M_v$ | $8$ |
| $V(M, h)$ | Potential function | scalar |
| $\Delta t$ | Layer time step | scalar |
| $L$ | Number of layers | integer |

Bold symbols denote vectors in $\mathbb{R}^3$. Italic lowercase denotes
scalars. Capital italic denotes matrices or multivectors, depending on
context.

## 2. Clifford algebra $\mathrm{Cl}(3,0)$

### 2.1 Basis

We work in the Clifford algebra $\mathrm{Cl}(3,0)$ with generators
$e_1, e_2, e_3$ satisfying

$$
e_i e_j + e_j e_i \;=\; 2 \delta_{ij} I.
$$

The algebra has dimension $2^3 = 8$. A multivector $M \in \mathrm{Cl}(3,0)$
decomposes into grades:

$$
M \;=\; \underbrace{s}_{\text{grade 0}}
    \;+\; \underbrace{v_1 e_1 + v_2 e_2 + v_3 e_3}_{\text{grade 1}}
    \;+\; \underbrace{b_{12} e_1 e_2 + b_{23} e_2 e_3 + b_{31} e_3 e_1}_{\text{grade 2}}
    \;+\; \underbrace{t \, e_1 e_2 e_3}_{\text{grade 3}}
$$

In the implementation, these are stored as an 8-vector

$$
M \;=\; [s, \; v_1, v_2, v_3, \; b_{12}, b_{23}, b_{31}, \; t].
$$

### 2.2 Geometric product

The product of two multivectors is the standard Clifford product, computed
via the structure constants

$$
(e_a e_b) \;=\; \sum_{c=0}^{7} C_{ab}^c \, e_c,
$$

where $C_{abc} \in \mathbb{R}^{8 \times 8 \times 8}$ is a fixed tensor.
Implementation: `gwm_mul(a, b) = einsum('bi,bj,ijk->bk', a, b, C)`.

### 2.3 Reversion

Reversion is the anti-automorphism defined by reversing the order of basis
vectors. On a grade-$k$ element it acts as a sign $(-1)^{k(k-1)/2}$:

$$
\widetilde{M} \;=\; \sum_{k=0}^{3} (-1)^{k(k-1)/2} \, \langle M \rangle_k
$$

where $\langle M \rangle_k$ denotes grade-projection. In the 8-component
representation:

$$
\widetilde{M} \;=\; [s, \; v_1, v_2, v_3, \; -b_{12}, -b_{23}, -b_{31}, \; -t].
$$

### 2.4 Grade projection

Projection onto grade $k$ is a linear operator $\langle \cdot \rangle_k$
that zeroes all components except those in grade $k$:

$$
\langle M \rangle_0 = [s, 0, 0, 0, 0, 0, 0, 0], \quad
\langle M \rangle_1 = [0, v_1, v_2, v_3, 0, 0, 0, 0], \quad \text{etc.}
$$

### 2.5 Rotors

A rotor $R \in \mathrm{Spin}(3) \subset \mathrm{Cl}(3,0)$ is an even
multivector ($R = \widetilde{R}^{-1}$) that acts on multivectors by
sandwiching:

$$
M \;\mapsto\; R \, M \, \widetilde{R}.
$$

This is the double-cover action of $\mathrm{SO}(3)$ on multivectors.
The group homomorphism $\mathrm{Spin}(3) \to \mathrm{SO}(3)$ is
2-to-1: $R$ and $-R$ implement the same rotation.

## 3. Multivector field

The encoder constructs $M_v$ for each occupied voxel $v$ from the point
cloud:

$$
\begin{aligned}
M_v[0] &= \frac{\log(1 + c_v)}{\log(1 + c_{\max})} && \text{(log-normalized density)} \\
M_v[1:4] &= \frac{\bar{x}_v - \bar{x}_{\text{scene}}}{r_{\max}} && \text{(centroid-radius position)} \\
M_v[4:7] &= \text{dual}(n_v) && \text{(PCA normal as bivector)} \\
M_v[7] &= \text{sign}\!\left(n_v \cdot (\bar{x}_v - \bar{x}_{\text{scene}})\right) && \text{(outward orientation)}
\end{aligned}
$$

where $c_v$ is the point count in voxel $v$, $\bar{x}_v$ is the voxel
centroid, $\bar{x}_{\text{scene}}$ is the scene centroid, $r_{\max}$ is
the max radius, and $n_v$ is the smallest singular vector of the
within-voxel point spread.

The dual map takes a normal vector $\mathbf{n}$ to its Hodge-dual bivector:

$$
\text{dual}(\mathbf{n}) \;=\; [-n_3, \; -n_1, \; -n_2].
$$

## 4. Gauge field $\mathfrak{so}(10)$

### 4.1 Algebra

The gauge field takes values in $\mathfrak{so}(10)$ — the Lie algebra of
$10 \times 10$ skew-symmetric real matrices with the commutator bracket

$$
[X, Y] \;=\; X Y - Y X.
$$

The dimension is $\dim \mathfrak{so}(10) = 45$.

### 4.2 Embedding of $\mathfrak{so}(3)$

We embed $\mathfrak{so}(3)$ into $\mathfrak{so}(10)$ via three fixed
generators acting on the first three coordinates:

$$
T_1 = \begin{pmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad
T_2 = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix}, \quad
T_3 = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{pmatrix},
$$

each padded to $10 \times 10$ with zeros. They satisfy

$$
[T_1, T_2] = T_3, \quad [T_2, T_3] = T_1, \quad [T_3, T_1] = T_2.
$$

### 4.3 Lifting a grade-1 multivector

Given a grade-1 multivector $J \in \mathrm{Cl}(3,0)$, we lift to
$\mathfrak{so}(10)$ by

$$
\text{lift}(J) \;=\; J[1] \, T_1 + J[2] \, T_2 + J[3] \, T_3.
$$

The result is a skew-symmetric $10 \times 10$ matrix. The map is
$\mathbb{R}$-linear and injective.

## 5. Gauge current $J_\mu$

For each spatial direction $\mu$, the current is

$$
J_\mu(M, h) \;=\; \text{lift}\!\left(\left\langle \widetilde{M} \, e_\mu \, M \right\rangle_1\right)
$$

where $e_\mu$ is the Clifford generator for direction $\mu$, and
$\langle \cdot \rangle_1$ extracts the grade-1 part.

**Structural properties:**

- $J_\mu(0, h) = 0$ (no source when $M = 0$).
- $J_\mu(M, 0) = 0$ (no source when $h = 0$).
- $J_\mu(M, h)$ is skew-symmetric, hence in $\mathfrak{so}(10)$.
- $J_\mu$ is quadratic in $M$.

**Note.** The current does not depend on $h$ explicitly in the current
implementation — the semantic fiber enters through the gauge-fixed
$h_K$ in the transport term, not through the current. This is a design
choice, not a theorem.

## 6. Covariant derivative

The stack uses a finite-difference covariant derivative. The update to
$M$ in one layer is

$$
M'_{v} \;=\; M_v \;-\; \Delta t \sum_{\mu} A_{v,\mu} \cdot \partial_\mu M_v,
$$

where $\partial_\mu M_v$ is the central finite-difference derivative of
$M$ on the voxel grid:

$$
\partial_\mu M_v \;\approx\; \frac{M_{v + \hat{\mu}} - M_{v - \hat{\mu}}}{2 \, \Delta x},
$$

with $\Delta x = 1 / R$ the physical grid spacing and $R$ the resolution.

The action $A_{v,\mu} \cdot \partial_\mu M_v$ is a $\mathfrak{so}(10)$
action on the 8-component multivector — that is, we promote $A_{v,\mu}$
to a $10 \times 10$ matrix, act on $M_v$ padded to 10 dimensions with
zeros, and project back to the first 8 components.

**Locality.** Every element of $M'_v$ depends only on $M$ and $A$ at the
same voxel and its immediate neighbors. There is no long-range
aggregation. This is the mathematical origin of the architectural limit
(see [`../findings/architectural_limit.md`](../findings/architectural_limit.md)).

## 7. Gauge update

The gauge field evolves as

$$
A^{(l+1)}_{v,\mu} \;=\; \text{clamp}\!\left(
    A^{(l)}_{v,\mu} + \Delta t \cdot J_\mu(M^{(l+1)}_v, h_v), \; \pm a_{\text{cap}}
\right),
$$

where the clamp is applied elementwise with a scalar cap $a_{\text{cap}}$.

**Locality.** The right-hand side depends only on quantities at the same
voxel $v$. No term couples $A_{v}$ to $A_{v'}$ for $v' \neq v$. This is
the strict locality that the pilot establishes as the architecture's
defining constraint.

**$\mathfrak{so}(10)$-preservation.** Because $J_\mu$ outputs a
skew-symmetric matrix and the clamp is elementwise-symmetric
($\text{clamp}(-x) = -\text{clamp}(x)$), the skew-symmetry of $A$ is
preserved at every step.

## 8. Gauge fixing

The semantic fiber's $k$-subspace component must remain orthonormal. Given
a matrix $X \in \mathbb{R}^{N \times K}$, the polar factor $Q$ (with
$Q^\top Q = I$) is the closest orthonormal matrix to $X$ in Frobenius
norm. It is computed by the Newton-Schulz iteration:

$$
\begin{aligned}
X_0 &= X / \|X\|_F, \\
X_{k+1} &= \tfrac{1}{2} X_k \left(3 I - X_k^\top X_k\right).
\end{aligned}
$$

**Convergence.** For full-rank $X$, the iteration converges quadratically
to the polar factor:

$$
\|X_{k+1}^\top X_{k+1} - I\| \;\le\; \tfrac{1}{2} \|X_k^\top X_k - I\|^2.
$$

**Iteration count.** In the implementation, $n_{\text{iter}} = 20$ is
sufficient for inputs up to condition number $\sim 10^2$. The pilot
verified orthogonality error $1.19 \times 10^{-7}$ on inputs with
condition number 113.

**Differentiability.** The iteration is a composition of matrix products
and additions, all differentiable. No SVD or eigendecomposition is used.

## 9. Symplectic dynamics

The $(M, p)$ pair follows a Hamiltonian system with total energy

$$
H(M, p) \;=\; \tfrac{1}{2} \|p\|^2 + V(M, h),
$$

where $V(M, h)$ is a learned potential (a 3-layer MLP in the
implementation). The equations of motion are

$$
\dot{M} \;=\; \frac{\partial H}{\partial p} \;=\; p, \qquad
\dot{p} \;=\; -\frac{\partial H}{\partial M} \;=\; -\nabla_M V(M, h).
$$

The symplectic leapfrog integrator advances one layer:

$$
\begin{aligned}
p_{1/2} &= p^{(l)} - \tfrac{\Delta t}{2} \nabla_M V(M^{(l)}, h), \\
M^{(l+1)} &= M^{(l)} + \Delta t \cdot p_{1/2}, \\
p^{(l+1)} &= p_{1/2} - \tfrac{\Delta t}{2} \nabla_M V(M^{(l+1)}, h).
\end{aligned}
$$

**Properties.**

- Second-order accurate in $\Delta t$.
- Symplectic: preserves the symplectic 2-form $dM \wedge dp$.
- Reversible: integrating backward with $-\Delta t$ returns the initial state.
- Energy drift $< 0.01\%$ over 200 steps (verified in
  [`../verification/cell_3_single_layer.md`](../verification/cell_3_single_layer.md)).

## 10. Full layer update

Combining Sections 6, 7, 8, 9, one layer executes the following sequence:

$$
\begin{aligned}
h^{(l)}_v &\leftarrow \text{NS}\!\left(h^{(l)}_v\right) && \text{(gauge-fix)} \\
M^{(\text{half})}_v &\leftarrow M^{(l)}_v - \Delta t \sum_\mu A^{(l)}_{v,\mu} \cdot \partial_\mu M^{(l)}_v && \text{(transport)} \\
(M^{(l+1)}, p^{(l+1)}) &\leftarrow \text{Leapfrog}(M^{(\text{half})}, p^{(l)}, h^{(l)}, \Delta t) && \text{(symplectic step)} \\
A^{(l+1)}_{v,\mu} &\leftarrow A^{(l)}_{v,\mu} + \Delta t \cdot J_\mu(M^{(l+1)}_v, h^{(l)}_v) && \text{(gauge update)}
\end{aligned}
$$

The stack is the composition of $L$ such layers:

$$
(M^{(L)}, p^{(L)}, A^{(L)}, h^{(L)}) \;=\; \text{Layer}_L \circ \cdots \circ \text{Layer}_1(M^{(0)}, p^{(0)}, A^{(0)}, h^{(0)}).
$$

## 11. Reversibility

Each layer is exactly reversible. Given $(M^{(l+1)}, p^{(l+1)}, A^{(l+1)}, h^{(l)})$,
the inverse computes $(M^{(l)}, p^{(l)}, A^{(l)}, h^{(l)})$ by reversing
the four operations in the opposite order:

$$
\begin{aligned}
A^{(l)}_{v,\mu} &\leftarrow A^{(l+1)}_{v,\mu} - \Delta t \cdot J_\mu(M^{(l+1)}_v, h^{(l)}_v) \\
p^{(\text{half})} &\leftarrow p^{(l+1)} + \tfrac{\Delta t}{2} \nabla_M V(M^{(l+1)}, h^{(l)}) \\
M^{(\text{half})} &\leftarrow M^{(l+1)} - \Delta t \cdot p^{(\text{half})} \\
p^{(l)} &\leftarrow p^{(\text{half})} + \tfrac{\Delta t}{2} \nabla_M V(M^{(\text{half})}, h^{(l)}) \\
M^{(l)}_v &\leftarrow M^{(\text{half})}_v + \Delta t \sum_\mu A^{(l)}_{v,\mu} \cdot \partial_\mu M^{(l)}_v
\end{aligned}
$$

**Reconstruction error** at float32 precision: $< 10^{-7}$ relative
across the full 8-layer stack (see [`../results/cell_4_results.json`](../results/cell_4_results.json)).

**Memory consequence.** Because each layer is exactly reversible, the
stack only needs to store the final state $(M^{(L)}, p^{(L)}, A^{(L)}, h^{(L)})$,
not the intermediate states. Backward pass reconstructs inputs layer by
layer from the output side. Peak memory is $\mathcal{O}(1)$ in depth.

## 12. The holonomy

The connection $A$ defines a holonomy — the path-ordered exponential along
a path $\gamma$:

$$
H_\gamma \;=\; \mathcal{P} \exp\!\left(\int_\gamma A_\mu \, dx^\mu\right),
$$

where $\mathcal{P}$ is the path-ordering operator. Discretized along $S$
sample points:

$$
H_\gamma \;\approx\; \prod_{s=1}^{S} \exp\!\left(\Delta t \cdot A_{\mu(s)} \, dx^\mu_s\right),
$$

with the product taken right-to-left (later samples multiplied on the left).

**A linearized variant** replaces the path-ordered product with a single
exponential:

$$
H_\gamma \;\approx\; \exp\!\left(\sum_{s=1}^{S} \Delta t \cdot A_{\mu(s)} \, dx^\mu_s\right).
$$

This is exact when the $A_{\mu(s)}$ commute along the path. In the
ribbon task, the linearized form is used because the path-ordered product
has 20 `matrix_exp` calls per decoder forward; the linearized form has one.

**Gauge covariance.** Under a constant gauge transformation $A \to G A G^\top$
with $G \in \mathrm{SO}(10)$:

$$
H_\gamma \;\to\; G \, H_\gamma \, G^\top.
$$

Verified in [`../verification/cell_2_holonomy.md`](../verification/cell_2_holonomy.md)
at residual $< 10^{-7}$.

## 13. Invariants and verification

| Quantity | Theoretical value | Verified value | Reference |
|---|---|---|---|
| Clifford associativity | $0$ | $< 10^{-5}$ | `cell_2_results.json` |
| Reversion anti-automorphism | $0$ | $< 10^{-5}$ | `cell_2_results.json` |
| Grade orthogonality | $0$ | $0$ | `cell_2_results.json` |
| Commutator antisymmetry | $0$ | $< 10^{-6}$ | `cell_2_results.json` |
| Symplectic energy drift (200 steps) | $< 0.01\%$ | $0.002\%$ | `cell_3_results.json` |
| Reversibility (8 layers, fp32) | $< 10^{-6}$ | $1.3 \times 10^{-8}$ | `cell_4_results.json` |
| Gauge covariance (Wu-Yang) | $< 10^{-6}$ | $< 10^{-7}$ | `cell_2_results.json` |
| Holonomy $\in \mathrm{SO}(10)$ | $\det = +1$ | $+1 \pm 10^{-6}$ | `cell_4_results.json` |
| Newton-Schulz orthogonality | $0$ | $1.2 \times 10^{-7}$ | `cell_2_results.json` |

## 14. What's missing — the site/link distinction

In lattice gauge theory, the connection lives on **links** between adjacent
sites:

$$
U_\mu(v) \;=\; \exp\!\left(i \, g \, a \, A_\mu(v)\right) \;\in\; G,
$$

and the curvature is a plaquette product:

$$
P_{\mu\nu}(v) \;=\; U_\mu(v) \, U_\nu(v + \hat{\mu}) \, U_\mu(v + \hat{\nu})^{-1} \, U_\nu(v)^{-1}.
$$

The GWM-Pilot implementation places $A$ on **sites**, not links. This is
why the gauge update is strictly local: there is no link structure to
carry information between sites. See
[`../findings/architectural_limit.md`](../findings/architectural_limit.md)
for the full development of this observation and its consequences.

**What a link formulation would change:**

- The tensor layout: $A$ would be indexed by $(v, \mu)$ where $v + \hat{\mu}$
  is a neighbor, requiring link-to-voxel association.
- The transport: $M'_{v} = U_\mu(v) M_{v + \hat{\mu}} - M_v$, a
  shift-and-rotate operation that genuinely transports.
- The gauge update: would need to preserve the link structure and would
  naturally couple $A_\mu(v)$ to $A_\mu(v + \hat{\mu})$ through the
  Wilson action.

The pilot's negative results for the ribbon and torque tasks are direct
consequences of the site-based formulation. This is not a bug; it's the
defining constraint of the pilot's architecture, and the pilot's finding
is a precise statement of its consequences.

## 15. References to the implementation

| Section | Code artifact |
|---|---|
| 2 | `core/clifford/table.py`, `core/clifford/ops.py` |
| 3 | `encoders/voxelizer.py`, `encoders/mv_builder.py` |
| 4 | `core/layer/gauge_field.py` (see `architecture/components/gauge_field.md`) |
| 5 | `core/layer/learnable_j.py` |
| 6 | `core/layer/symplectic_layer.py` |
| 7 | `core/layer/first_order.py`, `core/layer/capped.py` |
| 8 | `core/lie/gauge_fix.py` |
| 9 | `core/autograd/symplectic.py` |
| 10 | `core/stack.py` |
| 11 | `core/autograd/reversible.py` |
| 12 | `core/lie/holonomy.py` (see `architecture/components/gauge_field.md`) |
| 14 | — (this is the open problem for the next version) |

Directories marked `core/`, `encoders/`, etc., are the intended structure
of the reference implementation. They are documented here for
specification; the pilot release ships artifacts (weights + results),
not code.