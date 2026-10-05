# Verification — Cell 2 (Holonomy & Curvature)

This document specifies the verification protocol for the gauge-covariance
properties of the architecture: path-ordered holonomy, Yang-Mills
curvature, and the deep Clifford/Lie properties that Cell 1's primitives
rely on.

**Scope.** Sparse finite-difference convergence, Yang-Mills curvature on
sparse grids, path-ordered holonomy in $\mathrm{SO}(10)$, and gauge
covariance under constant transformations.

**Prerequisites.** `numpy`, `torch`, `scipy`. Runtime: ~30 seconds.

**Reference.** [`../results/cell_2_results.json`](../results/cell_2_results.json).

---

## How to read this document

Each section lists: what to verify, how, expected values, and common
failure modes. See [`cell_1_v7.md`](cell_1_v7.md) for the same format
applied to primitives.

---

## 1. Clifford deep properties

Cell 1's verification (see [`cell_1_v7.md`](cell_1_v7.md) §1) tests basic
identities. Cell 2 tests the deeper algebraic properties that the gauge
construction relies on.

### 1.1 Grade-orthogonality of the Clifford inner product

**What to verify.** The Clifford inner product
$\langle a, b \rangle = \langle a \widetilde{b} \rangle_0$ vanishes on
mismatched grades:

$$
\langle \langle a \rangle_i, \langle b \rangle_j \rangle \;=\; 0
\quad\text{for } i \neq j.
$$

**How to verify.**

```python
a = torch.randn(512, 8)
b = torch.randn(512, 8)

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

**Expected.** `max_cross < 1e-6`. Pilot measured exactly `0.0`.

### 1.2 Rotor grade-preservation

**What to verify.** A rotor commutes with grade projection:

$$
\langle R M \widetilde{R} \rangle_k \;=\; R \langle M \rangle_k \widetilde{R}.
$$

**How to verify.** Construct a unit rotor, then check for each grade $k$:

```python
for k in range(4):
    lhs = grade_k(rotor_sandwich(R, M), k)
    rhs = rotor_sandwich(R, grade_k(M, k))
    assert (lhs - rhs).abs().max().item() < 1e-5
```

**Expected.** All grades pass. Pilot measured max $4.77 \times 10^{-7}$.

**If it fails.** The rotor construction is likely off. A valid rotor
satisfies $R \widetilde{R} = 1$ (check first).

### 1.3 Unit-rotor norm preservation

Already checked in [`cell_1_v7.md`](cell_1_v7.md) §1.4. Included in Cell 2
for completeness.

---

## 2. Sparse finite-difference convergence

### 2.1 Central FD on $\sin(2\pi x)$

**What to verify.** The central finite difference on a dense periodic
grid converges at $\mathcal{O}(\Delta x^2)$:

$$
\left| \partial_x^{\text{FD}} \sin(2\pi x) - 2\pi \cos(2\pi x) \right| \;\to\; 0
$$

with rate $\Delta x^2$.

**How to verify.**

```python
for R in [16, 32, 64, 128]:
    coords = build_dense_grid(R)          # (R^3, 3) int32
    x_phys = coords[:, 0].float() / R
    psi = torch.sin(2*math.pi * x_phys).unsqueeze(-1).expand(-1, 4)

    grid = SortedKeySparseGrid(R)
    grid.build_from_points(coords, psi)

    dpsi = sparse_finite_difference(grid, 'x')
    analytic = 2*math.pi * torch.cos(2*math.pi * x_phys).unsqueeze(-1).expand(-1, 4)

    ix = coords[:, 0]
    interior = (ix >= 1) & (ix <= R - 2)
    err = (dpsi[interior] - analytic[interior]).abs().mean().item()

    print(f"R = {R:4d}  dx = {1/R:.5f}  err = {err:.4e}")
```

**Expected (from pilot):**

| $R$ | $\Delta x$ | Error |
|---|---|---|
| 16 | 0.0625 | $9.31 \times 10^{-2}$ |
| 32 | 0.03125 | $2.46 \times 10^{-2}$ |
| 64 | 0.015625 | $6.30 \times 10^{-3}$ |
| 128 | 0.0078125 | $1.59 \times 10^{-3}$ |

**Convergence rate.** $\log$-$\log$ slope $\approx 2.0$. Pilot measured
$1.958$.

**If it fails.** Either the grid construction is wrong, or the FD stencil
is not central. Check the boundary handling: the pilot uses one-sided
differences at the grid edges, excluded from the error measurement.

---

## 3. Yang-Mills curvature

### 3.1 $F = 0$ when $A = 0$

**What to verify.** For a flat connection $A = 0$, the curvature

$$
F_{\mu\nu} \;=\; \partial_\mu A_\nu - \partial_\nu A_\mu + [A_\mu, A_\nu]
$$

is exactly zero.

**How to verify.**

```python
A_zero = torch.zeros(N_vox, 3, 10, 10)
F = sparse_ym_curvature(A_zero, grid)
assert F.abs().max().item() == 0.0
```

**Expected.** Exactly $0.0$.

### 3.2 Commutator antisymmetry

**What to verify.** $[A_\mu, A_\nu] = -[A_\nu, A_\mu]$ for skew-symmetric
$A_\mu, A_\nu$.

**How to verify.**

```python
A_rand = random_so_like(N_vox, 3, 10)
comm_01 = A_rand[:, 0] @ A_rand[:, 1] - A_rand[:, 1] @ A_rand[:, 0]
comm_10 = A_rand[:, 1] @ A_rand[:, 0] - A_rand[:, 0] @ A_rand[:, 1]
err = (comm_01 + comm_10).abs().max().item()
```

**Expected.** `err < 1e-5`. Pilot measured exactly `0.0`.

### 3.3 Constant-gauge covariance

**What to verify.** Under a constant gauge transformation
$A_\mu \to G A_\mu G^\top$, the curvature transforms as
$F \to G F G^\top$.

**How to verify.**

```python
G, _ = torch.linalg.qr(torch.randn(10, 10))     # random orthogonal
if torch.linalg.det(G) < 0:
    G[:, 0] *= -1                                # ensure SO(10)

A_gauge = apply_constant_gauge(A_rand, G)
F_before = sparse_ym_curvature(A_rand, grid)
F_after  = sparse_ym_curvature(A_gauge, grid)
F_expected = G @ F_before @ G.transpose(-1, -2)

err = (F_after - F_expected).abs().max().item()
```

**Expected.** `err < 1e-4` relative to $\|F\|$. Pilot measured
$1.79 \times 10^{-7}$ against $\|F\| = 0.76$.

**If it fails.** The FD term in `sparse_ym_curvature` is not
gauge-covariant under the chosen transformation. Check that the
transformation is applied to all $\mu$ components consistently.

---

## 4. Path-ordered holonomy

### 4.1 Holonomy of the identity path

**What to verify.** $H = P \exp(\oint A \, dx) = I$ when $A \equiv 0$.

**How to verify.**

```python
A_zero = torch.zeros(L, 10, 10)     # path of length L, all zero
H = path_ordered_holonomy(A_zero, dt=1.0)
err = (H - torch.eye(10)).abs().max().item()
```

**Expected.** `err < 1e-6`.

### 4.2 Rectangular loop → commutator

**What to verify.** For a small rectangular loop with side lengths
$\Delta x, \Delta y$, the path-ordered holonomy to leading order is

$$
\log H \;\approx\; -\Delta x \cdot \Delta y \cdot [A_\mu, A_\nu].
$$

**How to verify.**

```python
A_mu = random_so(1, 10)[0]
A_nu = random_so(1, 10)[0]

dx = dy = 1e-2
loop = torch.stack([A_mu * dx, A_nu * dy, -A_mu * dx, -A_nu * dy])
H = path_ordered_holonomy(loop, dt=1.0)

log_H_np = schur_logm_so(H.cpu().numpy())
log_H = torch.tensor(log_H_np)

comm = A_mu @ A_nu - A_nu @ A_mu
expected = -dx * dy * comm
err = (log_H - expected).abs().max().item()
```

**Expected.** `err < 1e-5`. Pilot measured $6.93 \times 10^{-8}$.

**Sign convention.** The negative sign matches the pilot's loop
orientation ($+x$, $+y$, $-x$, $-y$) with the right-to-left path-ordered
product. A different convention gives the opposite sign but otherwise
identical structure.

**If it fails.** Check both the loop orientation and the product order in
`path_ordered_holonomy`. Both contribute to the sign.

### 4.3 Holonomy is in $\mathrm{SO}(10)$

**What to verify.** For any so(10)-valued path,
$H \in \mathrm{SO}(10)$:

$$
H^\top H = I, \qquad \det H = +1.
$$

**How to verify.**

```python
# Random so(10) path
A_path = random_so(L, 10)
H = path_ordered_holonomy(A_path, dt=1.0)

orth_err = (H.T @ H - torch.eye(10)).abs().max().item()
det_H = torch.linalg.det(H).item()

assert orth_err < 1e-6
assert abs(det_H - 1.0) < 1e-5
```

**Expected.** Both hold. Pilot measured orthogonality $< 10^{-6}$ and
$\det H \in [1 - 10^{-6}, 1 + 10^{-6}]$.

**If it fails.** The `matrix_exp` step is producing an element outside
$\mathrm{SO}(10)$. Check that the input to `matrix_exp` is skew-symmetric
at each step.

### 4.4 Gauge covariance of the holonomy

**What to verify.** Under a constant gauge transformation
$A_\mu \to G A_\mu G^\top$:

$$
H \;\to\; G H G^\top.
$$

**How to verify.**

```python
A_path = random_so(L, 10)
G, _ = torch.linalg.qr(torch.randn(10, 10))
if torch.linalg.det(G) < 0:
    G[:, 0] *= -1

H_orig = path_ordered_holonomy(A_path, dt=1.0)
A_path_gauge = G @ A_path @ G.T
H_gauge = path_ordered_holonomy(A_path_gauge, dt=1.0)
H_expected = G @ H_orig @ G.T

err = (H_gauge - H_expected).abs().max().item()
```

**Expected.** `err < 1e-5`. Pilot measured residual $< 10^{-7}$.

**This is the key gauge-covariance property.** It is what makes the
holonomy a physically meaningful observable.

---

## 5. Summary of reference values

| Check | Reference | Tolerance | Artifact |
|---|---|---|---|
| Clifford associativity | $7.6 \times 10^{-6}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Reversion anti-automorphism | $9.5 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Grade orthogonality | $0$ | $< 10^{-6}$ | `cell_2_results.json` |
| Rotor grade-preservation | $4.8 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Rotor norm preservation | $9.5 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Jacobi identity | $1.1 \times 10^{-8}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Adjoint skew-preservation | $1.8 \times 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |
| BCH leading order | $1.3 \times 10^{-7}$ | $< 10^{-6}$ | `cell_2_results.json` |
| FD convergence slope | $1.958$ | $2.0 \pm 0.3$ | `cell_2_results.json` |
| $F(A=0) = 0$ | exact | — | `cell_2_results.json` |
| Commutator antisymmetry | $0$ | $< 10^{-5}$ | `cell_2_results.json` |
| Gauge covariance of $F$ | $1.8 \times 10^{-7}$ | $< 10^{-4}$ rel. | `cell_2_results.json` |
| Holonomy $\det = +1$ | $+1$ | $\pm 10^{-6}$ | `cell_2_results.json` |
| Holonomy orthogonality | $< 10^{-6}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Loop → commutator | $6.9 \times 10^{-8}$ | $< 10^{-5}$ | `cell_2_results.json` |
| Holonomy gauge covariance | $< 10^{-7}$ | $< 10^{-5}$ | `cell_2_results.json` |

---

## 6. Common failure modes

1. **Sign conventions in the Clifford table.** Affects reversion and rotor
   tests. If associativity passes but reversion fails, check the reversion
   signs (should be `+ + + + - - - -`).

2. **Boundary handling in sparse FD.** If the FD convergence rate is
   wrong, check that boundary points use one-sided differences and are
   excluded from the error measurement.

3. **$G \notin \mathrm{SO}(10)$.** If $G$ has $\det = -1$ (which
   `torch.linalg.qr` can produce), the gauge covariance test fails because
   the transformation is in $\mathrm{O}(10)$ not $\mathrm{SO}(10)$. Always
   flip the first column when $\det G < 0$.

4. **Sign of the loop-commutator relation.** The pilot's sign is negative.
   If yours is positive, you have a different loop orientation or a
   different product order. Both are valid; just be consistent.

5. **`schur_logm_so` branch.** The Schur log returns the principal branch.
   For rotations by angle near $\pm\pi$, the result can jump. Inputs in
   the pilot stay in the small-angle regime.

---

## 7. References

| Aspect | Reference |
|---|---|
| Mathematical formulation | [`../architecture/formalism.md`](../architecture/formalism.md) |
| Gauge field component | [`../architecture/components/gauge_field.md`](../architecture/components/gauge_field.md) |
| Primitive verification | [`cell_1_v7.md`](cell_1_v7.md) |
| Primary artifact | [`../results/cell_2_results.json`](../results/cell_2_results.json) |