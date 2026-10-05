# Gauge Field ($\mathfrak{so}(10)$ connection)

The gauge field is the architecture's central object. It carries the
learned connection between voxels, acts as a covariant derivative on the
multivector field, and defines the holonomy used by non-local decoders.
This document specifies the field's layout, its update rule, its
properties, and the architectural constraint that the pilot's finding
identifies.

For the mathematical formulation, see
[`../formalism.md#4-gauge-field-so10`](../formalism.md#4-gauge-field-so10).
For the finding that emerges from the update rule, see
[`../../findings/architectural_limit.md`](../../findings/architectural_limit.md).

---

## 1. What it is

A per-voxel connection in the Lie algebra $\mathfrak{so}(10)$. Each voxel
carries one $10 \times 10$ skew-symmetric real matrix per spatial
direction:

$$
A_v \;\in\; \mathbb{R}^{3 \times 10 \times 10}, \qquad A_v[\mu]^\top = -A_v[\mu].
$$

For $N$ occupied voxels:

```
A : tensor of shape (N, 3, 10, 10), dtype float32
```

The total storage is $N \times 300$ floats — small compared to the
semantic fiber ($N \times 512$) or the potential network's parameters
(~650K per layer).

## 2. Why $\mathfrak{so}(10)$

The choice of $\mathfrak{so}(10)$ is over-determined for the current pilot.
The tasks exercised by the pilot only use the first three dimensions of
the $10$-dimensional representation; the remaining seven are a placeholder
for a future generalization to higher-dimensional fiber spaces.

**What the current architecture actually uses:**

The Clifford algebra $\mathrm{Cl}(3,0)$ has an 8-dimensional vector
representation. To act on it via a Lie algebra element, we embed
$\mathfrak{so}(3)$ into $\mathfrak{so}(10)$ using three fixed generators
that act on the first three coordinates of the fiber (see Section 3). The
remaining seven coordinates carry zero signal.

**Why not just use $\mathfrak{so}(3)$ directly?**

Two reasons:

1. **Headroom.** The gauge group can be extended to larger structure
   groups (e.g., $\mathfrak{so}(8)$, $\mathfrak{su}(5)$) without changing
   the tensor layout. $\mathfrak{so}(10)$ is the largest simple Lie
   algebra that still admits a real $16$-dimensional spinor representation,
   so it's a natural maximum.

2. **Gauge-fixing compatibility.** The gauge-fixing step projects the
   semantic fiber's $k$-subspace onto the orthogonal group $\mathrm{O}(K)$
   with $K = 10$. Having $\dim \mathfrak{so}(K) = \dim \mathfrak{so}(10)$
   means the gauge field is exactly the algebra of the gauge-fixing group.

Neither reason is currently *used*. A $\mathfrak{so}(3)$ field would give
bit-identical results on the pilot tasks. The $\mathfrak{so}(10)$ choice
is architectural potential, not current functionality.

## 3. Fixed generators

Three fixed $\mathfrak{so}(10)$ generators, acting on the first three
coordinates:

$$
T_1 = \begin{pmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad
T_2 = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix}, \quad
T_3 = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{pmatrix},
$$

each padded with zeros to $10 \times 10$. They satisfy the standard
commutator relations:

$$
[T_1, T_2] = T_3, \quad [T_2, T_3] = T_1, \quad [T_3, T_1] = T_2.
$$

**Frobenius norms:** $\|T_1\|_F = \|T_2\|_F = \|T_3\|_F = \sqrt{2}$. They
are orthonormal up to the $\sqrt{2}$ factor.

**Storage.** A single $3 \times 10 \times 10$ tensor, registered as a
buffer on each layer:

```python
self.register_buffer("T_stack", _T_GENERATORS.clone())
```

## 4. The lift

The `lift` operation takes a grade-1 multivector and produces a
$\mathfrak{so}(10)$ element:

$$
\text{lift}(J) \;=\; J[1] \, T_1 + J[2] \, T_2 + J[3] \, T_3.
$$

**Input.** A grade-1 multivector `J` of shape `(N, 8)` — only indices
1, 2, 3 are read.

**Output.** A $\mathfrak{so}(10)$ matrix of shape `(N, 10, 10)`.

**Properties:**

- **Linear** in $J$.
- **Injective.** Distinct grade-1 vectors map to distinct algebra elements.
- **Skew-symmetry-preserving.** The output satisfies $A^\top = -A$
  exactly (since each $T_i$ is skew-symmetric).

**Implementation:**

```python
def lift_grade1_to_so10(J_grade1, T_stack):
    v = J_grade1[:, 1:4]                     # (N, 3)
    return torch.einsum('nk,kij->nij', v, T_stack)   # (N, 10, 10)
```

## 5. The current $J_\mu$

For each spatial direction $\mu \in \{x, y, z\}$, the current is computed
from the multivector field:

$$
J_\mu(M) \;=\; \text{lift}\!\left(\left\langle \widetilde{M} \, e_\mu \, M \right\rangle_1\right).
$$

**Steps:**

1. Compute $\widetilde{M}$ (reversion of the multivector).
2. Form $\widetilde{M} \, e_\mu \, M$ via Clifford multiplication.
3. Extract the grade-1 part.
4. Lift to $\mathfrak{so}(10)$.

**Implementation** (in `LearnableJLayer.current`):

```python
def current(self, M, h):
    M_pad = F.pad(M, (0, K_SEM - D_MV))         # (N, 10)
    h_K   = h @ self.P.t()                       # (N, 10)
    J_list = []
    for mu in range(3):
        a_mu = torch.roll(M_pad, shifts=mu + 1, dims=-1)
        a_core = a_mu[:, :D_MV]
        # Learnable projection (v7 fix)
        a_proj = a_core @ self.W_proj[mu].t() + self.b_proj[mu]
        a_full = torch.cat([a_proj, a_mu[:, D_MV:]], dim=1)
        # Outer product construction
        J_mu = (a_full.unsqueeze(-1) * h_K.unsqueeze(-2)
                - h_K.unsqueeze(-1) * a_full.unsqueeze(-2))
        J_list.append(J_mu)
    return torch.stack(J_list, dim=1)            # (N, 3, 10, 10)
```

**Note on the current.** The implementation uses an outer-product
construction (`a_full ⊗ h_K - h_K ⊗ a_full`) rather than the algebraic
Clifford form $\langle \widetilde{M} e_\mu M \rangle_1$. The two forms
are related but not identical — the outer-product form is used because
it produces a rank-2 $\mathfrak{so}(10)$ element directly, whereas the
Clifford form produces a grade-1 multivector that then needs lifting.

The `LearnableJLayer` variant adds a per-direction linear projection
(`W_proj`, `b_proj`) applied to the rolled multivector before the outer
product. This is the v7 gradient-path fix described in
[`../../findings/architectural_limit.md#3-empirical-demonstration`](../../findings/architectural_limit.md#3-empirical-demonstration).

## 6. The update rule

The gauge field evolves per layer as

$$
A^{(l+1)}_{v,\mu} \;=\; \text{clamp}\!\left(
    A^{(l)}_{v,\mu} + \Delta t \cdot J_\mu(M^{(l+1)}_v, h_v), \; \pm a_{\text{cap}}
\right),
$$

where $a_{\text{cap}}$ is a scalar clamp value (default $50$; configurable).

**Implementation:**

```python
J_mu = self.current(M_next, h_gf)
J_mu = torch.nan_to_num(J_mu, nan=0.0, posinf=self.j_cap, neginf=-self.j_cap)
J_mu = torch.clamp(J_mu, -self.j_cap, self.j_cap)

A_next = A + dt * J_mu
A_next = torch.nan_to_num(A_next, nan=0.0, posinf=self.a_cap, neginf=-self.a_cap)
A_next = torch.clamp(A_next, -self.a_cap, self.a_cap)
```

**Two clamps:**

- `j_cap` (default $1.0$; used as $0.5$ in Cell 14, $5.0$ in Block 15)
  bounds the current before it enters the update.
- `a_cap` (default $50.0$; used as $10.0$ in Cell 14, $100.0$ in Block 15)
  bounds the accumulated field.

**Why clamps.** The current $J_\mu$ is quadratic in $M$; without clamping,
early-training instabilities produce large $M$ which produces large $J$
which produces larger $M$ at the next layer — a positive feedback loop.
Clamping breaks the feedback at both the source and the accumulated
field.

**Skew-symmetry preservation.** `torch.clamp` is applied elementwise. For
any input $x$, $\text{clamp}(-x, -c, c) = -\text{clamp}(x, -c, c)$ — the
clamp is odd-symmetric. Since $J_\mu$ is skew-symmetric and $A$ starts
skew-symmetric, the update preserves skew-symmetry exactly.

**Verified:** $\max |A_v[\mu] + A_v[\mu]^\top| < 10^{-6}$ throughout
training (asserted at the interface; see
[`../../verification/cell_1_v7.md`](../../verification/cell_1_v7.md)).

## 7. The transport term

$A$ couples to $M$ through a covariant-derivative-like term:

$$
M^{(\text{half})}_v \;=\; M^{(l)}_v \;-\; \Delta t \sum_\mu A^{(l)}_{v,\mu} \cdot \partial_\mu M^{(l)}_v.
$$

**Implementation:**

```python
M_pad  = F.pad(M,   (0, K_SEM - D_MV))
dM_pad = F.pad(d_M, (0, K_SEM - D_MV))
action = torch.einsum('nmij,nmj->ni', A, dM_pad)
M_half = (M_pad - dt * action)[:, :D_MV]
```

**The $\partial_\mu M$ term** (`d_M`) is the central finite-difference
derivative of $M$ on the voxel grid. It is computed once per forward pass
from the initial $M_0$ (not updated between layers).

**Contract over what, exactly.** The `einsum('nmij,nmj->ni', A, dM_pad)`
contracts over:
- `m` (spatial direction: $x, y, z$)
- `j` (the 10-dimensional fiber index of $dM$)

It does **not** contract over `n` (the voxel index). Each voxel's $M_v$
is modified using only $A_v$ and $dM_v$ — no cross-voxel aggregation.

## 8. Verified properties

| Property | Reference | Value |
|---|---|---|
| Skew-symmetry | `results/cell_2_results.json` | $< 10^{-6}$ |
| Gauge covariance (Wu-Yang) | `results/cell_2_results.json` | $< 10^{-7}$ |
| Holonomy $\det = +1$ | `results/cell_4_results.json` | $1 \pm 10^{-6}$ |
| Holonomy orthogonality | `results/cell_4_results.json` | $< 10^{-6}$ |
| Lifted element is skew-symmetric | `results/cell_2_results.json` | exact |

**Reproduce the check.** See
[`../../verification/cell_2_holonomy.md`](../../verification/cell_2_holonomy.md)
for the protocol.

## 9. The architectural constraint

**The update rule is strictly local in voxel space.**

$$
A^{(l+1)}_{v,\mu} \;=\; \text{clamp}\!\left(A^{(l)}_{v,\mu} + \Delta t \cdot J_\mu(M^{(l+1)}_v, h_v)\right)
$$

Every term on the right-hand side depends only on quantities at the same
voxel $v$. There is no summation over voxel indices, no aggregation, no
message passing. The update of $A_v$ is independent of $A_{v'}$ for
$v' \neq v$.

**Consequence.** Information cannot propagate between voxels through $A$.
The only cross-voxel coupling in the whole layer is the transport term
(Section 7), which couples $M_v$ to $M_{v\pm\hat\mu}$ through the
finite-difference derivative. That coupling is one voxel wide per layer.
A signal starting at voxel $v_0$ can reach voxel $v$ in at most $L$ layers,
where $L$ is the number of layers.

**For the rod task** (38-voxel gap between top and bottom): with 4 or 8
layers, the maximum A-mediated propagation distance is 4–8 voxels. The
torque signal injected at the top cannot reach the bottom.

**This is not a bug.** It is the definition of the update rule as
implemented. Section 14 of
[`../../architecture/formalism.md`](../../architecture/formalism.md)
formulates the mathematically correct lattice-gauge-theory version, where
$A$ lives on links rather than sites and information propagates through
plaquette products. The pilot's negative results are the direct
consequence of the site-based formulation.

## 10. What would change it

The minimal extension is a spatial coupling term:

$$
A^{(l+1)}_v \;=\; \text{clamp}\!\left(
    A^{(l)}_v + \Delta t \cdot \left[
        J_\mu(M^{(l+1)}_v, h_v)
        + \lambda \sum_{v' \sim v} (A^{(l)}_{v'} - A^{(l)}_v)
    \right]
\right).
$$

This adds a discrete Laplacian on $A$. With coupling strength $\lambda$,
information propagates $\sim L \cdot \sqrt{\lambda}$ voxels across the
stack.

**Why this is a different architecture, not a hyperparameter change.**
The three verified invariants all break under the coupled update:

- **Reversibility** — the current `invert_layer` undoes a pointwise
  update. The Laplacian adds a term involving neighbors, so the inverse
  requires solving a linear system (the Laplacian is invertible but not
  pointwise).
- **Gauge covariance** — the Wu-Yang check depends on the commutator
  structure of $J_\mu$. A naive Laplacian term is not gauge-covariant;
  preserving covariance requires either a covariant Laplacian or a
  gauge-fixing step after the coupling.
- **Symplectic structure** — the $(M, p)$ dynamics are unaffected by
  $A$-coupling, so this check should survive unchanged.

Full re-derivation and re-verification would be required. The pilot
release is frozen at the pre-coupling version so that the finding is
reproducible without ambiguity.

## 11. References

| Aspect | Reference |
|---|---|
| Mathematical formulation | [`../formalism.md`](../formalism.md) |
| The full finding | [`../../findings/architectural_limit.md`](../../findings/architectural_limit.md) |
| Verification protocol | [`../../verification/cell_1_v7.md`](../../verification/cell_1_v7.md) |
| Holonomy verification | [`../../verification/cell_2_holonomy.md`](../../verification/cell_2_holonomy.md) |
| Primary result JSON | `../../results/cell_2_results.json` |
| Stack-level verification | `../../results/cell_4_results.json` |