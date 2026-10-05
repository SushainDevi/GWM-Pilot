# Newton-Schulz Gauge Fixing

The semantic fiber's $k$-subspace component must remain orthonormal. This
document specifies the SVD-free polar iteration used to enforce it, its
convergence properties, and its role in the layer.

For the mathematical formulation, see
[`../formalism.md#8-gauge-fixing`](../formalism.md#8-gauge-fixing).

---

## 1. What it is

A function that projects a matrix $X \in \mathbb{R}^{N \times K}$ onto
the closest orthonormal matrix:

$$
\text{NS}(X) \;=\; \arg\min_{Q^\top Q = I_K} \|X - Q\|_F.
$$

**Implementation:**

```python
def newton_schulz_polar(A, n_iter=20, eps=1e-8):
    K = A.shape[-1]
    norm = A.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    X = A / norm
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    for _ in range(n_iter):
        XtX = X.transpose(-2, -1) @ X
        X = 0.5 * X @ (3.0 * I - XtX)
    return X
```

## 2. The iteration

$$
X_0 = X / \|X\|_F, \qquad
X_{k+1} = \tfrac{1}{2} X_k (3 I - X_k^\top X_k).
$$

**Convergence.** For full-rank $X$, the iteration converges quadratically
to the polar factor:

$$
\|X_{k+1}^\top X_{k+1} - I\|_F \;\le\; \tfrac{1}{2} \|X_k^\top X_k - I\|_F^2.
$$

Each iteration squares the residual, so the number of correct digits
roughly doubles per step.

## 3. Why not SVD

The standard way to compute the polar factor is SVD:

$$
X = U \Sigma V^\top \;\Rightarrow\; Q = U V^\top.
$$

SVD is exact and well-conditioned, but has two problems:

1. **Gradient issues.** The backward through SVD is unstable when singular
   values are close or repeated. This is common in the reversed stack.
2. **Cost.** SVD has $\mathcal{O}(NK^2)$ cost per call. For $N \sim 2000$
   voxels and one call per layer per forward, this is significant.

Newton-Schulz avoids both:
- Gradients through the iteration are well-defined and stable.
- Each iteration is two matrix products, $\mathcal{O}(NK^2)$ total across
  `n_iter`.

## 4. Where it's used

**In the layer**, gauge fixing is applied to `h_K = h @ P.t()` at the start
of each layer's forward:

```python
h_K = h @ self.P.t()                # (N, K)
h_K_fixed = gauge_fix_k_subspace(h_K, n_iter=20)
h_gf = h + (h_K_fixed - h_K) @ self.P   # write back into h
```

This projects the $k$-subspace component of $h$ onto $\mathrm{O}(K)$ while
leaving the orthogonal complement unchanged.

**Idempotency.** $\text{NS}(\text{NS}(X)) = \text{NS}(X)$ up to numerical
precision. After the first layer, subsequent layers see an already-fixed
$h$ and produce no additional change. The pilot verified this:

$$
\|\text{NS}(\text{NS}(X)) - \text{NS}(X)\| \;<\; 10^{-5}.
$$

**Consequence.** $h$ is frozen after the first layer. Only the first
layer's gauge fix has an effect. This is by design — the gauge constraint
is a projection, and its fixed point is reached in one step.

## 5. Iteration count

The pilot uses `n_iter = 20`.

**Justification.** For a random $N \times K$ matrix with condition number
$\sim 10^2$, the initial residual $\|X_0^\top X_0 - I\|$ is
$\mathcal{O}(10)$. Quadratically:
- After 4 iterations: residual $\sim 10^{-2}$
- After 8 iterations: residual $\sim 10^{-5}$
- After 12 iterations: residual $\sim 10^{-10}$ (below float32)
- After 20 iterations: saturates at float32 machine precision

**The pilot verified that `n_iter = 8` is insufficient** — the residual
was $4.9 \times 10^{-1}$ on ill-conditioned input. `n_iter = 20` reaches
$1.2 \times 10^{-7}$.

**Cost.** Linear in `n_iter`. 20 iterations is safe but not minimal.
12–15 iterations would suffice for most inputs.

## 6. Verified properties

| Property | Reference | Value |
|---|---|---|
| Orthogonality (well-conditioned) | `results/cell_2_results.json` | $1.19 \times 10^{-7}$ |
| Orthogonality (ill-conditioned, cond $\sim 10^2$) | `results/cell_2_results.json` | $1.19 \times 10^{-7}$ |
| Insufficiency of `n_iter = 8` | `results/cell_2_results.json` | $4.86 \times 10^{-1}$ |
| Idempotency | `results/cell_3_results.json` | $< 10^{-5}$ |

**Reproduce.** See [`../../verification/cell_1_v7.md`](../../verification/cell_1_v7.md) §4.

## 7. Numerical notes

**Normalization.** The iteration normalizes by $\|X\|_F$ before starting.
Without normalization, the iteration is unstable when $\|X\|$ is far from 1.

**Low-rank inputs.** If $X$ has rank less than $K$, the polar factor is
not unique. The Newton-Schulz iteration converges to one valid choice.
In practice, inputs are full-rank because they come from an MLP.

**Float32 saturation.** After ~12 iterations on float32, the residual
saturates at $\sim 10^{-7}$. Additional iterations do not improve accuracy.

## 8. References

| Aspect | Reference |
|---|---|
| Mathematical formulation | [`../formalism.md`](../formalism.md) |
| Verification protocol | [`../../verification/cell_1_v7.md`](../../verification/cell_1_v7.md) §4 |
| Where it's used in the layer | [`gauge_field.md`](gauge_field.md) |