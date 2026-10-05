# The Architectural Limit of GWM-Pilot v1

**The A-update rule is strictly local in voxel space.**

This document establishes that claim formally, demonstrates it empirically
under three independent control regimes, and shows that it explains every
downstream task result in the pilot. It closes with the minimal
architectural change that would lift the limit, and with an honest
accounting of what remains untested.

---

## 1. Statement of the finding

The gauge field in this architecture evolves through the update

$$
A^{(l+1)}_v \;=\; A^{(l)}_v \;+\; \Delta t \cdot J_\mu\!\left(M^{(l+1)}_v,\; h_v\right)
$$

for each voxel $v$ and each spatial direction $\mu \in \{x, y, z\}$. The
current $J_\mu$ is computed entirely from the multivector $M$ and the
semantic fiber $h$ **at the same voxel**. There is no term in this update
through which the field $A_{v'}$ at a neighboring voxel $v'$ can influence
$A_v$.

Consequently, information cannot propagate between voxels through $A$.
A perturbation to $A$ at voxel $v'$ produces zero change to $A$ at voxel
$v$ if $v \neq v'$, at every layer, at every step. This is a structural
property of the update rule, not a numerical property of any particular
parameter setting.

The architecture still uses $A$ to influence $M$. The transport term

$$
M^{(l+1/2)}_v \;=\; M^{(l)}_v \;-\; \Delta t \sum_{\mu} A^{(l)}_{v,\mu} \cdot \partial_\mu M^{(l)}_v
$$

couples $A$ to the *spatial derivative* of $M$ at the same voxel. This is
a covariant-derivative-like coupling, but it is pointwise: $A_v$ acts on
$M_v$ and its immediate finite-difference neighbors, not on distant voxels.
One layer of the stack transports information by at most one voxel through
this term.

With $L$ layers, the maximum distance over which a signal can travel
through the layer stack by $A$-mediated transport is $\mathcal{O}(L)$ voxels.
For $L = 4$ or $L = 8$, this is far shorter than the $\sim 38$-voxel gap
that separates the top and bottom of the rod in the Block 15 test, or the
$\sim 30$-voxel path length integrated by the Wilson line in the ribbon
task.

---

## 2. The mechanism, in code

The layer's forward pass (see `architecture/components/symplectic_layer.md`)
executes three operations in sequence:

```python
# Step 1: transport
action = einsum('nmij,nmj->ni', A, dM_pad)      # (N, D_MV)
M_half = (M_pad - dt * action)[:, :D_MV]

# Step 2: symplectic leapfrog on (M, p)
M_sym, p_next = SymplecticLeapfrogFn.apply(M_half, p, h_gf, V_net, dt)

# Step 3: gauge update
J_mu = self.current(M_next, h_gf)               # (N, 3, K, K)
A_next = clamp(A + dt * J_mu, -a_cap, a_cap)
```

Step 1 uses $A$ at voxel $v$ to modify $M$ at voxel $v$. The `einsum`
contracts over the spatial direction $\mu$; it does not contract over
voxel indices. The `dM_pad` tensor is the finite-difference derivative of
$M$ computed on the sparse voxel grid. Every element of `action[v]` depends
only on `A[v]` and `dM[v]`.

Step 3 computes $J_\mu$ from `M_next[v]` and `h_gf[v]` alone. `A_next[v]`
depends only on `A[v]`, `M_next[v]`, and `h_gf[v]`.

**There is no `roll`, no `conv`, no `gather`, no message-passing step
between voxels in either the A-update or the transport term.** The sparse
grid is used only for indexing the finite-difference derivative; it does
not introduce any spatial coupling in the A-evolution itself.

This is why we call the update *strictly local*: not approximately local,
not local to some radius, but dependent only on the same-voxel quantities.

---

## 3. Empirical demonstration

The Block 15 "Active Torque" task isolates the transport question. Its
design is specifically constructed to defeat every alternative failure
mode the pilot had previously encountered.

### 3.1 Task design

A single straight rod is generated with fixed geometry — the same point
cloud in every scene. Only the *input* varies: a random torque
$\omega \sim \mathcal{N}(0, I)$ is injected into the momentum $p_0$ at
the top 20% of the rod. The target is

$$
R_{\text{true}} \;=\; \exp_{\mathrm{SO}(3)}(\omega).
$$

The decoder reads the field $A$ **only at the bottom 20% of the rod**.

The target distribution has mean $\mathbb{E}[R_{\text{true}}] \approx I$,
so the trivial "predict identity" solution is actively bad (mean error
$\approx 92°$). The task cannot be solved by learning a fixed output.
The only path to loss below the null baseline is:

1. Top torque $\to$ $M$ perturbation at top voxels
2. $M$ perturbation $\to$ $J_\mu$ at top voxels
3. $J_\mu$ $\to$ $A$ at top voxels
4. $A$ at top voxels $\to$ $A$ at bottom voxels (transport)
5. $A$ at bottom voxels $\to$ decoder output

Step 4 is where the architecture fails.

### 3.2 Three control regimes

We ran three configurations to eliminate every confound.

#### v1 — naive

The layer forward ran through `ReversibleStackFn`, which runs its forward
pass under `torch.no_grad()` and requires at least one grad-requiring
input to trigger backward. Because the rod is encoded once and reused
across all scenes, all stack inputs were detached, and `ReversibleStackFn`
returned zero gradient to the stack parameters:

$$
\frac{\partial L}{\partial \text{encoder}} = 0, \quad
\frac{\partial L}{\partial W_{\text{proj}}} = 0, \quad
\frac{\partial L}{\partial b_{\text{proj}}} = 0, \quad
\frac{\partial L}{\partial V_{\text{net}}} = 0
$$

The stack was frozen. Transport ratio $= 0.0000$, but this result is
uninformative — the stack was not training.

#### v2 — patched

We bypassed `ReversibleStackFn` by calling `stack._run_layers` directly.
We also computed the spatial derivative `d_M` from `M0` using sparse
finite differences and passed it to the stack, since `d_M` had previously
defaulted to zero. All gradients became healthy:

$$
\frac{\partial L}{\partial \text{encoder}} = 1.7770 \times 10^{2}, \quad
\frac{\partial L}{\partial W_{\text{proj}}} = 9.8159, \quad
\frac{\partial L}{\partial b_{\text{proj}}} = 5.0403, \quad
\frac{\partial L}{\partial V_{\text{net}}} = 4.4965 \times 10^{-2}
$$

Training proceeded for 30 epochs. Transport ratio at every epoch:

```
epoch  0: 0.0000
epoch  5: 0.0000
epoch 15: 0.0000
epoch 29: 0.0000
```

Bit-identical $A_8^{\text{bot}}$ under two independently-sampled top torques.

#### v3 — frozen decoder

The remaining possible confound was that the decoder might be overfitting
to specific $A_8^{\text{bot}}$ patterns during training, giving the
appearance of transport failure when the truth was a bad readout. We
removed this channel by freezing the decoder at initialization
($\text{log\_scale} = 100$, $\mu\text{\_weights} = [1, 1, 1]$). With the
decoder fixed, the loss can only decrease by changing $A_8^{\text{bot}}$.
If the stack transports, the training loss will drop below the null
baseline of $3.8022$. If not, the loss must return to that baseline.

Training loss trajectory:

```
epoch  0: 4.7589   (transient from first encoder forward)
epoch 11: 6.0309   (worst)
epoch 24: 3.8280
epoch 29: 3.7916   (final — 0.28% from null)
```

Final loss is **within 0.28% of the null baseline**. Transport ratio
across all 30 epochs: $0.0000$. The frozen-decoder configuration is the
definitive test: no overfitting channel exists, gradients are healthy,
`d_M` is nonzero, and yet the loss cannot drop below null because
$A_8^{\text{bot}}$ is torque-independent.

### 3.3 What the numbers mean

The transport ratio is defined as

$$
\text{transport} \;=\; \frac{\left\| A_8^{\text{bot}}(\omega_a) - A_8^{\text{bot}}(\omega_b) \right\|}{\tfrac{1}{2}\left(\left\| A_8^{\text{bot}}(\omega_a) \right\| + \left\| A_8^{\text{bot}}(\omega_b) \right\|\right)}
$$

for two independently-sampled torques $\omega_a, \omega_b$. A value of
zero means the bottom field is *exactly* the same regardless of what
torque was applied at the top.

The reported value is not "small." It is `0.0000` in the JSON to four
decimal places, and it does not move across 30 epochs of training. The
underlying `float32` difference

$$
\left\| A_a - A_b \right\| \;=\; 0.0000 \times 10^{0}
$$

at initialization. Every distinct top torque produces the same $A$ at
every bottom voxel, bit-for-bit.

---

## 4. Cross-task confirmation

The locality of the A-update explains every downstream task result in
the pilot. The table below classifies each task by whether it requires
$A$ to carry information across space.

| Task | Decoder reads | Requires non-local $A$? | Result | Mechanism |
|---|---|---|---|---|
| FDTD (Cell 12) | $A$ at each voxel, independently | No — pointwise | ✅ $5.27\times$ over trivial | Per-voxel prediction from per-voxel $A$ is a local-to-local mapping |
| Rotation (Cell 13) | $M_8$ only; $A$ unused | No | ⚪ Neutral ($8.14°$) | Task doesn't use $A$; locality irrelevant |
| Ribbon (Cell 14) | Wilson line along a path | Yes | ❌ $95°$ held-out | Path requires $A$ to carry information along its length; $A$ can't |
| Torque (Block 15) | $A$ at rod's bottom only | Yes | ❌ Transport $= 0.0000$ | Same as ribbon but more isolated |

**FDTD** works because the FDTD decoder performs a
$\text{Linear}(A_8.\text{reshape}(N, -1))$ mapping — each voxel's
predicted field value depends only on that voxel's $A$. This is exactly
the operation the local A-update can support.

**Rotation** is neutral because the decoder reads $M_8$, not $A_8$. The
architecture can solve the task using the $M$-path, but the task doesn't
exercise the $A$-path at all. The stack's parameters feeding $A$ receive
$\partial L / \partial W_{\text{proj}} = 0.0$ exactly — a signature of a
decoder with no $A$-dependency.

**Ribbon** requires $A$ to vary along a path that spans the full length
of the twisted tube. The Wilson line integrates $A$ along $\sim 20$
sampled points covering a path of $\sim 40$ voxels. For the integral to
be nontrivial, $A$ must differ along the path. Since $A$ is strictly
local and the input $M$ is smoothly varying along the tube, $A$ too
varies smoothly — but its values at different path points are determined
by their own local $M$, not by information propagating from the
endpoints. The Wilson line reads the *scene's local field texture*, not
the *accumulated twist of the tube*. The model converges to predicting
the mean twist, which is the best a local-$A$-integral can do.

**Torque** is the isolation of the ribbon failure mechanism. It removes
every other dependency: the rod geometry is fixed, the input varies only
at the top, and the target depends only on the input torque. If $A$
transported, the bottom field would carry the torque signal. It doesn't.

---

## 5. What the finding rules out

The three-control-regime demonstration (v1, v2, v3) eliminates the
following alternative explanations for the transport failure:

| Hypothesis | Eliminated by | Evidence |
|---|---|---|
| "Stack frozen — no gradient" | v2 | $\partial L / \partial W_{\text{proj}} = 9.82$, $\partial L / \partial \text{encoder} = 1.78 \times 10^{2}$ |
| "`d_M` missing or zero" | v2 | Sparse finite-difference `dM0` computed and passed |
| "`ReversibleStackFn` bug" | v2 | Bypassed via `_run_layers`; identical result |
| "Caps too tight — $A$ can't grow" | v2, v3 | `a_cap = 100`, $A_8$ per-voxel norm $\sim 6 \times 10^{-2}$ at end of training |
| "Signal too weak" | v2, v3 | `torque_scale = 5.0`; top $A_8$ per-voxel norm $5.8 \times 10^{-1}$ |
| "Optimization needs more time" | v2, v3 | 30 epochs; transport ratio never moves from zero |
| "Decoder overfits to noise" | v3 | Decoder frozen at init; loss returns to null baseline |
| "Wrong `dt_holo`" | v2, v3 | `dt_holo = 100`; a torque-induced $A$ of even $10^{-3}$ at the bottom would produce a visible rotation |

None of the above. The only remaining explanation is the A-update's
structural locality.

---

## 6. The architecture's envelope

The finding gives the pilot a well-defined operating envelope. The
following table classifies a candidate task by its A-dependency, and
predicts the outcome with high confidence.

| Classification | Task type | Predicted result |
|---|---|---|
| **A-independent** | Decoder reads only $M$ | Neutral — architecture can solve via M-path but doesn't exercise $A$ |
| **A-local** | Decoder reads $A$ at each voxel independently | Positive — succeeds at the level the FDTD task demonstrates |
| **A-non-local** | Decoder integrates $A$ along paths, or reads $A$ at a spatially distant location | Negative — fails; the failure mode is the mean-attractor or a torque-independent readout |

The classification is decidable from the decoder's API alone. A decoder
that takes $A_8[v]$ and produces a per-voxel output is A-local. A decoder
that takes $A_8$ and produces a scene-level output via path integration,
mean, sum, or any aggregation that requires coupling distant voxels is
A-non-local.

**This is the pilot's most useful contribution.** It is a design rule
that says, before writing a line of training code, whether the
architecture can solve a proposed task.

---

## 7. What would fix it

The minimal architectural change that would lift the limit is a spatial
coupling term in the A-update:

$$
A^{(l+1)}_v \;=\; A^{(l)}_v \;+\; \Delta t \cdot \left[
    J_\mu\!\left(M^{(l+1)}_v, h_v\right)
    \;+\; \lambda \sum_{v' \sim v} \left(A^{(l)}_{v'} - A^{(l)}_v\right)
\right]
$$

where $v' \sim v$ ranges over the finite-difference neighbors of $v$ on
the voxel grid, and $\lambda$ is a scalar coupling coefficient. This is
a discrete Laplacian on the gauge field, added to the source term.

Under this rule, a perturbation to $A$ at voxel $v'$ propagates to
neighboring voxels over successive layers. With $L$ layers and coupling
strength $\lambda$, the effective propagation radius is approximately
$L \cdot \sqrt{\lambda}$ voxels. For the rod task ($38$-voxel gap) at
$L = 8$ layers, $\lambda \approx 0.5$ would give a propagation radius
of $\sim 5\text{–}6$ voxels per layer, sufficient to bridge the gap.

**This is a different architecture.** The current verification suite
does not automatically extend. Specifically:

- **Reversibility** — the current `invert_layer` undoes a pointwise
  update. The coupled update requires undoing a Laplacian, which is
  invertible but requires a different inverse method.
- **Gauge covariance** — the Wu-Yang residual check depends on the
  commutator structure of $J_\mu$. The Laplacian term must itself be
  gauge-covariant or the covariance property is lost.
- **Symplectic structure** — the current energy-conservation check
  measures the $(M, p)$ dynamics. The $A$-coupling does not directly
  affect $M$'s dynamics, so this check should pass unchanged.

Every primitive would need to be re-verified. Every downstream task
would need to be re-run. The v1 release is frozen at the pre-coupling
version so that the finding reported here is reproducible without
ambiguity about which version was being tested.

---

## 8. Limitations of the pilot

The finding is strong but the pilot has open questions.

**What we did not test:**

1. **Coupling magnitude scaling.** We did not test whether increasing
   `n_layers` to 32 or 64 (with a spatial coupling of zero) would
   eventually produce transport. Theoretically, with $L$ layers and
   zero coupling, the propagation radius is $\mathcal{O}(L)$ voxels
   through the $A \cdot \partial M$ term alone. With $L = 64$ it might
   just barely bridge the rod's $38$-voxel gap, but this would be an
   inefficiency, not a fix. We did not run this control.

2. **Site vs. link formulation.** The current architecture places $A$
   on *sites* (voxels). The mathematically correct formulation of
   lattice gauge theory places the connection on *links* between
   sites, with the curvature given by a plaquette product. Adopting a
   link formulation would give $A$ genuine transport by construction,
   at the cost of a different tensor layout and a different decoder
   API. We did not implement this.

3. **Long-range tasks with intermediate aggregation.** We tested two
   non-local tasks (ribbon, torque). Both use the failure mode
   directly. A task with a different non-local structure — e.g., a
   decoder that reads $A$ at several *distant but fixed* voxels — was
   not tested. The classification rule (Section 6) predicts these
   would also fail, but the empirical basis is two tasks, not a
   systematic sweep.

4. **Non-abelian structure at large amplitude.** The clamps
   `j_cap = 5.0`, `a_cap = 100.0` were chosen empirically. Whether
   the non-abelian structure of $\mathfrak{so}(10)$ plays any role in
   the transport failure — or whether the same result would obtain
   with an abelian $\mathfrak{u}(1)$ field — was not tested. A
   $\mathfrak{u}(1)$ ablation would clarify whether the failure is
   generic or specific to the non-abelian case.

5. **Alternative loss formulations.** The torque task uses pose loss,
   which has a mean-attractor for symmetric distributions. A
   contrastive or von Mises-Fisher loss might change the observed
   behavior by removing the mean-attractor. This would not defeat the
   transport limitation but could change the intermediate numbers
   reported here (e.g., the held-out error ratio of $1.16$ in Block
   15 v2 might improve to $1.0$ or below without solving the task).

**What these limitations do not affect:**

The core claim of Section 1 stands. The A-update's locality is a
property of the code, not of any loss or optimizer choice. No loss
formulation, no hyperparameter, and no training schedule can change
whether $A^{(l+1)}[v]$ depends on $A^{(l)}[v']$ for $v' \neq v$. The
claim is formal and the empirical demonstrations are consistent with
it under every control regime we ran.

---

## 9. References to artifacts

Every number in this document corresponds to a JSON in `results/`.

| Claim | Artifact |
|---|---|
| A-update locality (code) | `architecture/components/symplectic_layer.md`, `architecture/components/gauge_field.md` |
| Block 15 v1 frozen stack | `results/block_15_results.json` |
| Block 15 v2 healthy gradients, transport $= 0$ | `results/block_15_v2_results.json` |
| Block 15 v3 frozen decoder, loss returns to null | `results/block_15_v3_results.json` |
| FDTD positive result | `results/block_9_v2_summary.json` |
| Rotation neutral result | `results/cell_13_results.json`, `results/cell_13_v3_sweep.json` |
| Ribbon partial result | `results/cell_14_results.json`, `results/cell_14_iterations.json` |
| Encoder iterations (v1–v5) | `results/cell_6_results.json` through `results/cell_6_v5_results.json` |
| Primitive verification | `results/cell_2_results.json`, `results/cell_3_results.json`, `results/cell_4_results.json` |

Trained models are in `weights/`. Loading instructions are in the
`weights/README.md` (in progress).

---

## Summary

The A-update rule

$$
A^{(l+1)}_v \;=\; A^{(l)}_v \;+\; \Delta t \cdot J_\mu\!\left(M^{(l+1)}_v, h_v\right)
$$

is strictly local. This was demonstrated empirically under three control
regimes that eliminate every alternative explanation, and it explains
every task result in the pilot. The limit is structural and cannot be
worked around by hyperparameter tuning or loss reformulation. Lifting it
requires a spatial coupling term in the A-update — a different
architecture that will be the subject of the next version.