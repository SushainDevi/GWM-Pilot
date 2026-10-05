# Architecture Overview

**GWM-Pilot implements a reversible Yang-Mills stack** — a geometric neural
architecture where information flows through a learned non-abelian gauge
field. This document describes the five components and how they compose.
For the mathematical formulation, see [`formalism.md`](formalism.md). For
per-component details, see [`components/`](components/).

## The one-paragraph version

A point cloud is voxelized to a sparse 3D grid. Each occupied voxel carries
an 8-component Clifford multivector $M$ (density, position, PCA normal,
pseudoscalar), a semantic fiber $h \in \mathbb{R}^{512}$, and a
$\mathfrak{so}(10)$ connection $A$ with one $10 \times 10$ skew-symmetric
matrix per spatial direction. The stack alternates between a symplectic
leapfrog step on $(M, p)$ and a gauge update on $A$. Both steps are wrapped
in custom `autograd.Function`s so the stack is reversible (O(1) memory in
depth) and the integrator is symplectic (energy-conserving). A decoder reads
either $M_8$ or $A_8$ at the output — the choice determines whether the
task exercises the gauge field.

## The five components

### 1. Clifford multivector fields

**What it is.** Each voxel's geometric content is stored as a multivector
in the Clifford algebra $\mathrm{Cl}(3,0)$, which has $2^3 = 8$ components:

| Index | Grade | Content |
|---|---|---|
| 0 | scalar | density (log-count normalized) |
| 1–3 | vector | centroid-radius-normalized position |
| 4–6 | bivector | dual of the PCA surface normal |
| 7 | pseudoscalar | outward-orientation sign |

**Why it matters.** The geometric product (implemented as an 8×8×8 structure
constant tensor) supports grade-preserving operations that a plain vector
field cannot. Rotors, reversion, and the covariant derivative all act on
this algebra.

**Reference.** [`components/clifford_field.md`](components/clifford_field.md)

### 2. Yang-Mills gauge field

**What it is.** A per-voxel connection in the Lie algebra $\mathfrak{so}(10)$,
one $10 \times 10$ skew-symmetric matrix per spatial direction:

$$
A_v \;\in\; \mathbb{R}^{3 \times 10 \times 10}, \qquad A_v[\mu]^\top = -A_v[\mu]
$$

**How it evolves.** The current $J_\mu$ is computed from the multivector
and the semantic fiber, lifted into $\mathfrak{so}(10)$ via three fixed
generators:

$$
J_\mu(M, h) \;=\; \mathrm{lift}\!\left(\langle \tilde{M} \, e_\mu \, M \rangle_1\right),
\qquad
A^{(l+1)}_v \;=\; A^{(l)}_v + \Delta t \cdot J_\mu
$$

**What it does.** $A$ couples to $M$'s spatial derivative, acting as a
covariant derivative on the multivector field. It also defines a
holonomy — the path-ordered exponential $H = \mathcal{P} \exp(\oint A \,
dx)$ — which is a gauge-covariant observable.

**The finding.** This update is strictly local in voxel space. See
[`../findings/architectural_limit.md`](../findings/architectural_limit.md).

**Reference.** [`components/gauge_field.md`](components/gauge_field.md)

### 3. Symplectic leapfrog integrator

**What it is.** The $(M, p)$ dynamics follow a Hamiltonian system. The
integrator is a custom `torch.autograd.Function` that runs the standard
symplectic leapfrog:

$$
\begin{aligned}
p_{1/2} &= p - \tfrac{\Delta t}{2} \nabla_M V(M, h) \\
M' &= M + \Delta t \, p_{1/2} \\
p' &= p_{1/2} - \tfrac{\Delta t}{2} \nabla_M V(M', h)
\end{aligned}
$$

**Why a custom `Function`.** The reference-aligned backward is defined by
replaying the forward pass under `torch.enable_grad()`, which lets the
outer autograd graph treat the whole step as a single primitive. This
preserves second-order accuracy and energy conservation.

**Verified.** Energy drift $< 0.01\%$ over 200 steps on a harmonic potential
(see [`../verification/cell_3_single_layer.md`](../verification/cell_3_single_layer.md)).

**Reference.** [`components/symplectic_layer.md`](components/symplectic_layer.md)

### 4. Reversible autograd stack

**What it is.** The 8-layer stack (configurable) uses a custom
`ReversibleStackFn` that saves only the final state during forward and
reconstructs intermediate states during backward:

```python
class ReversibleStackFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stack, M, p, A, h):
        with torch.no_grad():
            M_o, p_o, A_o, h_o = stack._run_layers(M, p, A, h)
        ctx.save_for_backward(M_o, p_o, A_o, h_o)
        return M_o, p_o, A_o, h_o
```

**What it buys.** O(1) memory in depth. Peak memory grows 0.7% from 2→8
layers (see [`../results/cell_4_results.json`](../results/cell_4_results.json)).
A naive stack would grow linearly.

**One constraint.** The backward is only called when at least one input
requires grad. Detaching all of `M, p, A, h` at the stack's boundary
silently produces a stack with no parameter gradients. This was discovered
during Block 15 v1 and is documented in
[`../findings/architectural_limit.md`](../findings/architectural_limit.md).

**Reference.** [`components/reversible_stack.md`](components/reversible_stack.md)

### 5. Newton-Schulz gauge fixing

**What it is.** The gauge constraint on $h$ — the requirement that the
$k$-subspace projection be orthonormal — is imposed via SVD-free polar
iteration:

$$
X_{k+1} \;=\; \tfrac{1}{2} X_k \left(3 I - X_k^\top X_k\right)
$$

**Why SVD-free.** SVD gradients are unstable in the reversed stack and
expensive to compute per voxel. Newton-Schulz converges quadratically to
the polar factor and is differentiable throughout.

**Verified.** 20 iterations converge to orthogonality error $1.2 \times
10^{-7}$ on inputs with condition number 113 (see
[`../results/cell_2_results.json`](../results/cell_2_results.json)).

**Reference.** [`components/gauge_fixing.md`](components/gauge_fixing.md)

## How they compose

```
point cloud (N, 7)
        │
        ▼
┌──────────────────────────┐
│  Equivariant voxelizer   │   centroid-radius normalization
│  PCA surface normals     │   sparse grid (grid_res)
└──────────────────────────┘
        │
        ▼  M0 (N_vox, 8),  h0 (N_vox, 512),  A0 (N_vox, 3, 10, 10)
        │
┌──────────────────────────┐
│  Reversible YM stack     │   for l in range(n_layers):
│  ├─ gauge-fix h          │     h ← Newton-Schulz(h)
│  ├─ transport            │     M ← M − Δt · A · ∂M
│  ├─ symplectic step      │     (M, p) ← Leapfrog(M, p, V_net)
│  └─ gauge update         │     A ← A + Δt · J_μ(M, h)
└──────────────────────────┘
        │
        ▼  M8 (N_vox, 8),  p8,  A8 (N_vox, 3, 10, 10),  h8
        │
        ▼
┌──────────────────────────┐
│  Task-specific decoder   │   — direction (M8)
│                          │   — per-voxel field (A8)
│                          │   — Wilson line (A8 along path)
│                          │   — bottom-A (A8 at subset)
└──────────────────────────┘
```

**The decoder choice determines the task class.** A decoder that reads
$A_8$ per-voxel is "A-local" and can succeed. A decoder that integrates
$A_8$ along paths or reads it at spatially distant locations is
"A-non-local" and will fail (see Section 6 of the finding).

## What the pilot demonstrated

| Decoder type | Task | Result |
|---|---|---|
| $A_8$ per-voxel | FDTD field prediction | ✅ 5.27× over trivial |
| $M_8$ only | Rotation direction | ⚪ Neutral, 8.14° |
| $A_8$ along path | Kinematic ribbon | ❌ 95° held-out |
| $A_8$ at distant site | Active torque | ❌ Transport = 0.0000 |

The finding is that these four outcomes are the same outcome, viewed from
four angles. Full derivation: [`../findings/architectural_limit.md`](../findings/architectural_limit.md).

## Parameter count

The full pipeline has **3,585,024 parameters** (v5 encoder, 4-layer stack):

| Component | Parameters |
|---|---|
| Semantic encoder (MLP) | ~400K |
| Label embedding | ~10K |
| Stack layers (4×) | ~3.1M |
| — $V_\text{net}$ per layer | ~650K |
| — $W_\text{proj}$, $b_\text{proj}$ per layer | ~1.9K |
| Decoder | task-dependent |

The bulk is in $V_\text{net}$, the potential MLP. The gauge-coupling
parameters ($W_\text{proj}$, $b_\text{proj}$) are deliberately small.

## What this architecture is not

- **Not a Transformer.** There is no attention, no token-mixing, no
  global aggregation. Information propagates only through the local
  coupling between adjacent layers.
- **Not a CNN.** The stack is not translation-invariant. Each voxel's
  state evolves under its own local update plus the finite-difference
  coupling.
- **Not a graph neural network.** The voxel grid is regular; there is
  no learned message-passing topology.

It is closest to a **lattice gauge theory simulation** where the connection
is learned rather than fixed.

## Where to go next

- **To understand the math:** [`formalism.md`](formalism.md)
- **To understand a specific component:** [`components/`](components/)
- **To understand the finding:** [`../findings/architectural_limit.md`](../findings/architectural_limit.md)
- **To see the numbers:** [`../results/`](../results/)
- **To load a trained model:** [`../README.md`](../README.md#quick-start)