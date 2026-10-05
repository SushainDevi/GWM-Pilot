# Clifford Multivector Field ($\mathrm{Cl}(3,0)$)

The multivector field is the architecture's geometric content per voxel.
Each voxel carries an 8-component element of the Clifford algebra
$\mathrm{Cl}(3,0)$, encoding density, position, surface normal, and
orientation. This document specifies the field's layout, its algebra,
and its role in the transport and gauge terms.

For the mathematical formulation, see
[`../formalism.md#2-clifford-algebra-cl30`](../formalism.md#2-clifford-algebra-cl30).
For the gauge field it couples to, see
[`gauge_field.md`](gauge_field.md).

---

## 1. What it is

Per voxel $v$, a multivector

$$
M_v \;=\; [s_v, \; v_{v,1}, v_{v,2}, v_{v,3}, \; b_{v,12}, b_{v,23}, b_{v,31}, \; t_v] \;\in\; \mathbb{R}^8
$$

with components assigned to grades:

| Index | Grade | Content |
|---|---|---|
| 0 | scalar | log-normalized density |
| 1, 2, 3 | vector | centroid-radius-normalized position |
| 4, 5, 6 | bivector | dual of the PCA surface normal |
| 7 | pseudoscalar | outward orientation sign |

For $N$ occupied voxels:

```
M : tensor of shape (N, 8), dtype float32
```

## 2. The algebra

### 2.1 Generators

$\mathrm{Cl}(3,0)$ has three generators $e_1, e_2, e_3$ satisfying

$$
e_i e_j + e_j e_i \;=\; 2 \delta_{ij}.
$$

The 8-dimensional basis is $\{1, e_1, e_2, e_3, e_1 e_2, e_2 e_3, e_3 e_1, e_1 e_2 e_3\}$
in the index order used above.

### 2.2 The geometric product

The product of two multivectors is computed via structure constants:

$$
(e_a \, e_b) \;=\; \sum_c C_{abc} \, e_c,
$$

with $C \in \mathbb{R}^{8 \times 8 \times 8}$ a fixed tensor.

**Implementation:**

```python
def gwm_mul(a, b):
    orig = a.shape
    a2 = a.reshape(-1, 8)
    b2 = b.reshape(-1, 8)
    return torch.einsum('bi,bj,ijk->bk', a2, b2, _C_CLIFFORD).view(orig)
```

### 2.3 Reversion

Reversion flips the sign of grades 2 and 3:

$$
\widetilde{M} \;=\; [s, v_1, v_2, v_3, -b_{12}, -b_{23}, -b_{31}, -t].
$$

### 2.4 Grade projection

$\langle M \rangle_k$ zeroes all components except grade $k$:

```python
def grade_k(m, k):
    out = torch.zeros_like(m)
    slices = {0: [0], 1: [1, 2, 3], 2: [4, 5, 6], 3: [7]}
    for idx in slices.get(k, []):
        out[..., idx] = m[..., idx]
    return out
```

### 2.5 Rotors

A rotor $R \in \mathrm{Spin}(3)$ (even multivector, $R \widetilde{R} = 1$)
acts on multivectors by sandwiching:

$$
M \;\mapsto\; R \, M \, \widetilde{R}.
$$

**Implementation:**

```python
def rotor_sandwich(R, M):
    R_rev = R.clone()
    R_rev[..., 1:4] *= -1
    R_rev[..., 4:7] *= -1
    return gwm_mul(gwm_mul(R, M), R_rev)
```

## 3. Encoder construction

The encoder builds $M_v$ per voxel from the point cloud:

$$
\begin{aligned}
M_v[0] &= \frac{\log(1 + c_v)}{\log(1 + c_{\max})} && \text{(log-normalized density)} \\
M_v[1:4] &= \frac{\bar{x}_v - \bar{x}_{\text{scene}}}{r_{\max}} && \text{(centroid-radius position)} \\
M_v[4:7] &= \text{dual}(n_v) && \text{(PCA normal as bivector)} \\
M_v[7] &= \text{sign}\!\left(n_v \cdot (\bar{x}_v - \bar{x}_{\text{scene}})\right) && \text{(outward orientation)}
\end{aligned}
$$

where:
- $c_v$ = point count in voxel $v$
- $\bar{x}_v$ = voxel centroid
- $\bar{x}_{\text{scene}}$ = scene centroid
- $r_{\max}$ = max radius over all voxels
- $n_v$ = smallest singular vector of the within-voxel point spread (PCA surface normal)

**Dual map** from normal $\mathbf{n}$ to bivector components:

$$
\text{dual}(\mathbf{n}) \;=\; [-n_3, \; -n_1, \; -n_2].
$$

## 4. Role in the layer

$M$ appears in two places in the layer's forward pass.

### 4.1 Transport (covariant derivative)

$M$ is modified using the gauge field $A$ and its own spatial derivative:

$$
M^{(\text{half})}_v \;=\; M_v \;-\; \Delta t \sum_\mu A_{v,\mu} \cdot \partial_\mu M_v.
$$

See [`gauge_field.md`](gauge_field.md) §7 for the implementation.

### 4.2 Gauge source

$M$ generates the current that updates $A$:

$$
J_\mu(M) \;=\; \text{lift}\!\left(\langle \widetilde{M} e_\mu M \rangle_1 \right).
$$

**In the pilot implementation**, the current uses a rolled-and-projected
form rather than the algebraic Clifford form; see
[`gauge_field.md`](gauge_field.md) §5 for the disclosure.

## 5. Verified properties

| Property | Reference | Value |
|---|---|---|
| Associativity | `results/cell_2_results.json` | $< 10^{-5}$ |
| Reversion anti-automorphism | `results/cell_2_results.json` | $< 10^{-5}$ |
| Grade orthogonality | `results/cell_2_results.json` | exact ($0$) |
| Unit-rotor norm preservation | `results/cell_2_results.json` | $< 10^{-5}$ |

**Reproduce.** See [`../../verification/cell_1_v7.md`](../../verification/cell_1_v7.md) §1.

## 6. Why 8 components

$\mathrm{Cl}(3,0)$ is the smallest Clifford algebra that supports
rotors of $\mathrm{SO}(3)$. Alternatives:

- **3-vector field:** loses the bivector channel. Cannot represent
  surface normals as algebra elements, only as separate vectors.
- **4-vector (quaternion) field:** supports rotations but not the full
  grade decomposition. The pseudoscalar channel is lost.
- **$\mathrm{Cl}(3,0)$ (current):** 8 components. Full grade structure.
  Supports rotors, reversion, and the covariant derivative.

The pilot does not claim 8D is *necessary* — a 4D or 3D ablation was not
run. The choice is standard for $\mathrm{SO}(3)$-equivariant networks.

## 7. Storage and cost

- $N$ voxels × 8 floats = $8N$ floats total.
- For $N \sim 2000$ (the pilot's rod), this is 16 KB.
- Dominated by the semantic fiber $h$ ($512 N$ floats) in the same layer.

## 8. References

| Aspect | Reference |
|---|---|
| Mathematical formulation | [`../formalism.md`](../formalism.md) |
| Gauge field coupling | [`gauge_field.md`](gauge_field.md) |
| Encoder construction | [`encoder.md`](encoder.md) |
| Verification protocol | [`../../verification/cell_1_v7.md`](../../verification/cell_1_v7.md) §1 |