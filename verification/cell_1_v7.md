# Verification — Cell 1 v7 (Primitives)

This document specifies the verification protocol for the primitive
operations that the GWM-Pilot architecture is built on. It is a checklist
for a reader who wants to independently reproduce the pilot's primitive
claims.

**Scope.** Clifford algebra, Lie algebra, sparse grid, gauge fixing,
symplectic integration, reversible autograd. This is the verification
suite for Cell 1 v7, which established the primitives used by all
downstream tasks.

**Prerequisites.** `numpy`, `torch`, `scipy`. No trained weights needed.
Runtime: ~60 seconds on CPU.

**Reference results.** Every check below reports a reference value. The
full set of measured values is in
[`../results/cell_2_results.json`](../results/cell_2_results.json),
[`../results/cell_3_results.json`](../results/cell_3_results.json), and
[`../results/cell_4_results.json`](../results/cell_4_results.json).

---

## How to read this document

Each section lists:
- **What to verify** — the mathematical property
- **How to verify** — the procedure, with a code sketch
- **Expected values** — the reference values from the pilot
- **Tolerance** — the acceptance threshold
- **If it fails** — the most likely causes of a mismatch

If your values match the reference within tolerance, you've reproduced
the pilot's primitive behavior. If they don't, the "If it fails" section
points to the usual culprits.

---

## 1. Clifford algebra $\mathrm{Cl}(3,0)$

### 1.1 Associativity of the geometric product

**What to verify.** For random multivectors $a, b, c$:

$$
(a b) c \;=\; a (b c).
$$

**How to verify.**

```python
a = torch.randn(200, 8)
b = torch.randn(200, 8)
c = torch.randn(200, 8)

lhs = gwm_mul(gwm_mul(a, b), c)
rhs = gwm_mul(a, gwm_mul(b, c))
err = (lhs - rhs).abs().max().item()
```

**Expected.** `err < 1e-5` on float32. The pilot measured `7.63e-6`.

**If it fails.** The Clifford structure constant tensor is likely wrong.
Re-derive from the canonical relations $e_i e_j + e_j e_i = 2\delta_{ij}$
and check the sign of the bivector components.

### 1.2 Reversion is an anti-automorphism

**What to verify.** Reversion reverses multiplication order:

$$
\widetilde{(a b)} \;=\; \widetilde{b} \, \widetilde{a}.
$$

**How to verify.**

```python
lhs = clifford_reverse(gwm_mul(a, b))
rhs = gwm_mul(clifford_reverse(b), clifford_reverse(a))
err = (lhs - rhs).abs().max().item()
```

**Expected.** `err < 1e-5`. Pilot measured `9.54e-7`.

**If it fails.** Check the reversion sign convention. For $\mathrm{Cl}(3,0)$
the reversion signs are $(+, +, +, +, -, -, -, -)$ — identity on grades
0 and 1, negative on grades 2 and 3.

### 1.3 Grade orthogonality

**What to verify.** The scalar part of $a \widetilde{b}$ vanishes when
grades are mismatched:

$$
\langle \langle a \rangle_i \cdot \widetilde{\langle b \rangle_j} \rangle_0 \;=\; 0
\quad \text{for } i \neq j.
$$

**How to verify.**

```python
max_cross = 0.0
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        ai = grade_k(a, i)
        bj = grade_k(b, j)
        s = gwm_mul(ai, clifford_reverse(bj))[:, 0]
        max_cross = max(max_cross, s.abs().max().item())
```

**Expected.** `max_cross < 1e-6`. Pilot measured exactly `0.0` — this is
a structural property of the grade decomposition and should hold to
machine precision.

**If it fails.** Your grade-indexing is wrong. In the 8-component
representation: grade 0 = index 0; grade 1 = indices 1, 2, 3; grade 2 =
indices 4, 5, 6; grade 3 = index 7.

### 1.4 Unit rotor norm preservation

**What to verify.** A rotor $R$ (even, unit-norm) preserves multivector
norm under sandwiching:

$$
\|R \, M \, \widetilde{R}\| \;=\; \|M\|.
$$

**How to verify.** Construct a rotor from a random unit bivector and
angle:

```python
theta = torch.rand(200, 1) * math.pi
biv = torch.randn(200, 3)
biv = biv / biv.norm(dim=1, keepdim=True)

R = torch.zeros(200, 8)
R[:, 0] = torch.cos(theta.squeeze(-1) / 2)
R[:, 4:7] = torch.sin(theta.squeeze(-1) / 2).unsqueeze(-1) * biv

M = torch.randn(200, 8)
err = (rotor_sandwich(R, M).norm(dim=1) - M.norm(dim=1)).abs().max().item()
```

**Expected.** `err < 1e-5`. Pilot measured `9.54e-7`.

**If it fails.** Either the rotor construction is wrong, or the sandwich
uses the wrong reversion (must use `R_rev` with negative signs on
grades 2 and 3).

---

## 2. Lie algebra

### 2.1 Jacobi identity

**What to verify.** For skew-symmetric matrices $A, B, C$:

$$
[A, [B, C]] + [B, [C, A]] + [C, [A, B]] \;=\; 0.
$$

**How to verify.**

```python
A = random_skew(128, 10)
B = random_skew(128, 10)
C = random_skew(128, 10)

jac = (commutator(A, commutator(B, C))
       + commutator(B, commutator(C, A))
       + commutator(C, commutator(A, B)))
err = jac.abs().max().item()
```

**Expected.** `err < 1e-5`. Pilot measured `1.12e-8`.

**If it fails.** The commutator is not `AB - BA`. Check implementation.

### 2.2 Adjoint action preserves skew-symmetry

**What to verify.** For $X \in \mathfrak{so}(K)$ and $A \in \mathfrak{so}(K)$:

$$
e^A X e^{-A} \;\in\; \mathfrak{so}(K).
$$

**How to verify.**

```python
A_small = random_skew(128, 10) * 0.1
X_in = random_skew(128, 10)
X_adj = torch.matrix_exp(A_small) @ X_in @ torch.matrix_exp(-A_small)
err = (X_adj + X_adj.transpose(-1, -2)).abs().max().item()
```

**Expected.** `err < 1e-5`. Pilot measured `1.79e-7`.

### 2.3 BCH at leading order

**What to verify.** For $\varepsilon = 10^{-3}$ and small $\mathfrak{so}(K)$
elements $A, B$:

$$
\log(\exp(\varepsilon A) \exp(\varepsilon B)) \;\approx\; \varepsilon(A + B) + \tfrac{\varepsilon^2}{2}[A, B].
$$

**How to verify.** Use `scipy_expm` for forward, `schur_logm_so` for the
log. Compare against the RHS. Expected error is $\mathcal{O}(\varepsilon^3)$.

**Expected.** `err < 1e-6`. Pilot measured `1.28e-7`.

**If it fails.** Either the exponential is off (unlikely for `scipy_expm`)
or `schur_logm_so` is returning the wrong branch. Check that the log
of a rotation by angle $\theta \in (-\pi, \pi)$ returns the principal
branch.

### 2.4 Schur log of SO(K)

**What to verify.** For $R = \exp(A)$ with $A$ skew-symmetric:

$$
\exp(\text{schur\_logm\_so}(R)) \;=\; R.
$$

**How to verify.**

```python
A_so = random_skew(20, 10)  # scale ~1
R_so = scipy_expm(A_so)
err = np.linalg.norm(scipy_expm(schur_logm_so(R_so)) - R_so, 'fro')
```

**Expected.** `err < 1e-9`. Pilot measured `3.62e-15`.

**If it fails.** The Schur decomposition must be done with
`output='real'` for this to work on real matrices with complex
eigenvalues.

---

## 3. Sparse grid

### 3.1 Self-lookup exactness

**What to verify.** After building a sparse grid from coordinates,
looking up the same coordinates returns the stored values.

**How to verify.**

```python
coords = torch.randint(0, 64, (5000, 3))
coords, _ = torch.unique(coords, dim=0, return_inverse=True)
values = torch.randn(coords.shape[0], 4)

grid = SortedKeySparseGrid(resolution=64)
grid.build_from_points(coords, values)

found, vals = grid.lookup(coords)
assert found.all()
assert (vals - values).abs().max().item() < 1e-7
```

**Expected.** All found; values match to machine precision.

### 3.2 Miss detection

**What to verify.** Querying coordinates that weren't in the grid
returns `found = False`.

**How to verify.**

```python
far = coords.clone()
far[:, 0] = (far[:, 0] + 32) % 64
found_far, _ = grid.lookup(far)
miss_rate = (~found_far).float().mean().item()
assert miss_rate > 0.5
```

**Expected.** Miss rate > 0.5 (typically ~0.98 for random offsets).

---

## 4. Gauge fixing (Newton-Schulz polar)

### 4.1 Convergence on well-conditioned input

**What to verify.** For a random full-rank $N \times K$ matrix $X$ with
condition number $\sim 4$, the Newton-Schulz iteration converges to the
polar factor:

$$
\text{NS}_{20}(X)^\top \text{NS}_{20}(X) \;\approx\; I_K.
$$

**How to verify.**

```python
Q_true, _ = torch.linalg.qr(torch.randn(500, 10))
S = torch.linspace(0.5, 2.0, 10)
V, _ = torch.linalg.qr(torch.randn(10, 10))
X = Q_true @ torch.diag(S) @ V.t()

Q_polar = newton_schulz_polar(X, n_iter=20)
err = (Q_polar.T @ Q_polar - torch.eye(10)).abs().max().item()
```

**Expected.** `err < 1e-5`. Pilot measured `1.19e-7`.

### 4.2 Convergence on ill-conditioned input

**What to verify.** Same as 4.1 but with condition number $\sim 10^2$.
The Newton-Schulz iteration should still converge with `n_iter=20`.

**How to verify.** Same as 4.1 with $S$ scaled to give condition number
$10^2$ (e.g., `torch.linspace(0.1, 10.0, 10)`).

**Expected.** `err < 1e-5`. Pilot measured `1.19e-7`.

**If it fails.** `n_iter` is too small. The pilot found `n_iter=8` gives
`err ≈ 4.9e-1` on this input (insufficient). `n_iter=20` is required.

### 4.3 Non-convergence with n_iter=8 (negative control)

**What to verify.** On ill-conditioned input, `n_iter=8` is
**insufficient**. This confirms the pilot's choice of `n_iter=20` was
necessary.

**How to verify.**

```python
Q_polar_8 = newton_schulz_polar(X, n_iter=8)
err_8 = (Q_polar_8.T @ Q_polar_8 - torch.eye(10)).abs().max().item()
assert err_8 > 1e-3   # expected to fail
```

**Expected.** `err_8 > 1e-3`. Pilot measured `4.86e-1`.

**This is a negative control.** A reader who finds `err_8 < 1e-3` has
either well-conditioned input or a different implementation.

---

## 5. Symplectic integration

### 5.1 Energy conservation on harmonic potential

**What to verify.** The symplectic leapfrog conserves total energy
$H(M, p) = \tfrac{1}{2}(\|p\|^2 + \|M\|^2)$ over 100–200 steps.

**How to verify.**

```python
class HarmonicV(nn.Module):
    def forward(self, M, h):
        return 0.5 * (M ** 2).sum(dim=-1)

M = torch.randn(20, 8) * 0.5
p = torch.zeros(20, 8)
h = torch.randn(20, 512) * 0.01

def energy(M, p):
    return 0.5 * ((p ** 2).sum() + (M ** 2).sum())

E0 = energy(M, p).item()
for _ in range(100):
    M, p = SymplecticLeapfrogFn.apply(M, p, h, HarmonicV(), 0.01)
drift_pct = abs(energy(M, p).item() - E0) / E0 * 100
```

**Expected.** `drift_pct < 0.01`. Pilot measured `0.002`.

**If it fails.** The integrator is not symplectic. Check that the update
is exactly the leapfrog (half-step on $p$, full-step on $M$, half-step
on $p$).

### 5.2 Invertibility

**What to verify.** Forward + backward through the integrator
reconstructs the initial state.

**How to verify.**

```python
M_in = torch.randn(16, 8)
p_in = torch.zeros(16, 8)
M_fwd, p_fwd = SymplecticLeapfrogFn.apply(M_in, p_in, h, V, 0.01)

# Algebraic reverse (see verification/cell_3_single_layer.md)
M_rec, p_rec = algebraic_leapfrog_inverse(M_fwd, p_fwd, h, V, 0.01)

assert (M_rec - M_in).abs().max().item() < 1e-5
assert (p_rec - p_in).abs().max().item() < 1e-5
```

**Expected.** Both errors < 1e-5. Pilot measured `0.0` and `9.3e-10`.

### 5.3 Gradcheck at float64

**What to verify.** For a single leapfrog step at double precision, the
custom `backward` matches numerical differentiation.

**How to verify.**

```python
layer = YMSLayer(P).to('cpu').double()
M = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
p = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
A = torch.randn(8, 3, 10, 10, dtype=torch.float64).requires_grad_(True)

torch.autograd.gradcheck(
    lambda M, p, A: layer(M, p, A, h),
    (M, p, A),
    eps=1e-6, atol=1e-4, rtol=1e-3,
)
```

**Expected.** `gradcheck = True`. Pilot confirmed.

**If it fails.** Numerical tolerance is too tight, or the `backward`
isn't reproducing the forward under `create_graph=True`. Try loosening
`atol` to 1e-3 as a first diagnostic.

---

## 6. Reversible stack

### 6.1 Reconstruction error at 8 layers

**What to verify.** Forward + inverse through 8 layers reconstructs the
input states to float32 precision.

**How to verify.**

```python
stack = GWMStack(P, n_layers=8, dt=0.01)
M8, p8, A8, h8 = stack.forward_no_grad(M, p, A, h)
M0_r, p0_r, A0_r, h0_r = stack.invert(M8, p8, A8, h8)

M_err = (M0_r - M).norm() / M.norm()
```

**Expected.** `M_err < 1e-6`. Pilot measured `1.33e-8`.

**If it fails.** An `invert_layer` is missing a term. Check that it
exactly reverses the order of operations.

### 6.2 Memory is O(1) in depth

**What to verify.** Peak GPU memory grows sub-linearly from 2 layers
to 8 layers.

**How to verify.**

```python
peaks = {}
for n in [2, 4, 8]:
    stack = GWMStack(P, n_layers=n).to('cuda')
    torch.cuda.reset_peak_memory_stats()
    M8, _, A8, _ = stack(M, p, A, h)
    torch.cuda.synchronize()
    peaks[n] = torch.cuda.max_memory_allocated() / 1024**2

growth = (peaks[8] - peaks[2]) / peaks[2]
assert growth < 1.0   # less than 100% growth
```

**Expected.** `growth < 1.0`. Pilot measured `0.7%` (2→8 layers).

**Note.** The forward alone is not O(1); the backward is where the
memory saving kicks in. To see the full effect, measure peak memory
including `backward()`.

### 6.3 Full-stack gradcheck at float64

**What to verify.** The reversible stack's backward produces correct
Jacobians at double precision.

**How to verify.**

```python
stack = GWMStack(P, n_layers=8).to('cpu').double()
torch.autograd.gradcheck(
    lambda M, p, A: stack(M, p, A, h),
    (M.requires_grad_(), p.requires_grad_(), A.requires_grad_()),
    eps=1e-6, atol=1e-4, rtol=1e-3,
)
```

**Expected.** `True`. Pilot confirmed. Runtime: ~5–10 minutes on CPU.

---

## 7. What this verification does not cover

The following are verified in separate documents:

- **Holonomy gauge covariance** — see
  [`cell_2_holonomy.md`](cell_2_holonomy.md)
- **Single-layer sub-properties (transport identity, YM source zero
  conditions, J_μ antisymmetry)** — see
  [`cell_3_single_layer.md`](cell_3_single_layer.md)
- **Stack-level properties (holonomy parity, gradient scaling across
  layers, throughput)** — see
  [`cell_4_stack.md`](cell_4_stack.md)
- **Encoder iteration behavior (v1–v5)** — see
  [`cell_6_v5_encoder.md`](cell_6_v5_encoder.md)

## 8. Full summary of reference values

| Check | Reference | Tolerance | Artifact |
|---|---|---|---|
| Clifford associativity | $7.63 \times 10^{-6}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Reversion anti-automorphism | $9.54 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Grade orthogonality | $0$ | $< 10^{-6}$ | `cell_2_results.json` |
| Unit-rotor norm preservation | $9.54 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Jacobi identity | $1.12 \times 10^{-8}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Adjoint skew-preservation | $1.79 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| BCH leading order | $1.28 \times 10^{-7}$ | $< 10^{-6}$ | `cell_2_results.json` |
| Schur log round-trip | $3.62 \times 10^{-15}$ | $< 10^{-9}$ | `cell_2_results.json` |
| Sparse grid self-lookup | exact | $< 10^{-7}$ | `cell_2_results.json` |
| Newton-Schulz (well-cond) | $1.19 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Newton-Schulz (ill-cond, 20 iter) | $1.19 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Newton-Schulz (ill-cond, 8 iter) | $4.86 \times 10^{-1}$ | $> 10^{-3}$ | `cell_2_results.json` |
| Leapfrog energy drift | $0.002\%$ | $< 0.01\%$ | `cell_3_results.json` |
| Leapfrog invertibility (M) | $0$ | $< 10^{-5}$ | `cell_3_results.json` |
| Leapfrog invertibility (p) | $9.3 \times 10^{-10}$ | $< 10^{-5}$ | `cell_3_results.json` |
| Single-layer gradcheck | `True` | — | `cell_3_results.json` |
| Reversibility (8 layers, fp32) | $1.33 \times 10^{-8}$ | $< 10^{-6}$ | `cell_4_results.json` |
| Memory growth (2→8 layers) | $0.7\%$ | $< 100\%$ | `cell_4_results.json` |
| Full-stack gradcheck | `True` | — | `cell_4_results.json` |

## 9. If your run disagrees

The most common sources of mismatch are:

1. **Sign conventions in the Clifford structure constants.** A different
   ordering of $e_1, e_2, e_3$ produces a different tensor, but all
   internal products should still satisfy the same identities. If
   associativity fails, the tensor is wrong; if only rotor behavior
   differs, the sign of the bivector basis might be flipped.

2. **The `output='real'` flag in Schur.** Without it, `schur` returns a
   complex decomposition and the log extraction is different.

3. **`n_iter` in Newton-Schulz.** If you use fewer iterations, the
   residual will be larger; if you use more, it will be smaller but
   slower. The pilot uses 20.

4. **The order of the two half-steps in leapfrog.** Some implementations
   do $M$, then $p$; the pilot does $p$, $M$, $p$. The energy-conservation
   property holds for both, but the invertibility algebra differs.

5. **Memory measurement granularity.** `reset_peak_memory_stats` measures
   the peak within the measurement window. If you measure forward-only
   vs forward+backward, the numbers differ; the pilot's O(1) claim is
   measured on the full forward+backward cycle.

If your values are within an order of magnitude of the reference but
not exact, the most likely cause is a different random seed. All
checks are seeded in the pilot; if you draw new random inputs the
specific numbers will differ but the tolerances should hold.