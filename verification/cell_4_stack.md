# Verification — Cell 4 (Reversible Stack)

This document specifies the verification protocol for the reversible
stack: reconstruction, memory scaling, full-stack gradcheck, holonomy
parity, and gradient flow.

**Scope.** Everything that operates at the stack level — the reversible
autograd `Function`, the memory profile of depth, and the gradient
pathway through all $L$ layers.

**Prerequisites.** `numpy`, `torch`. Runtime: ~10–15 minutes (gradcheck
at 8 layers is the bottleneck).

**Reference.** [`../results/cell_4_results.json`](../results/cell_4_results.json).

---

## 1. Stack reversibility at depth

### 1.1 Reconstruction error

**What to verify.** Forward + inverse through 8 layers reconstructs the
input state to float32 precision.

**How to verify.**

```python
torch.manual_seed(4040)
model = GWMPilot(...).to('cuda')

N = 400
M = torch.randn(N, 8, device='cuda') * 0.1
p = torch.randn(N, 8, device='cuda') * 0.05
A = skew(torch.randn(N, 3, 10, 10, device='cuda')) * 0.05
h = F.normalize(torch.randn(N, 512, device='cuda'), dim=-1)

with torch.no_grad():
    M8, p8, A8, h8 = model.stack.forward_no_grad(M, p, A, h)
    M0_r, p0_r, A0_r, h0_r = model.stack.invert(M8, p8, A8, h8)

M_err = ((M0_r - M).norm() / M.norm()).item()
p_err = ((p0_r - p).norm() / p.norm()).item()
A_err = ((A0_r - A).norm() / A.norm()).item()
```

**Expected.**

| Quantity | Reference | Tolerance |
|---|---|---|
| $M$ rel. error | $1.3 \times 10^{-8}$ | $< 10^{-6}$ |
| $p$ rel. error | $0$ | $< 10^{-5}$ |
| $A$ rel. error | $1.0 \times 10^{-8}$ | $< 10^{-6}$ |

**If it fails.** Check that `invert_layer` exactly reverses the order
of operations from `forward`. Common mistakes:
- Forgetting to reverse the sign on one of the half-steps
- Recomputing $V$ at the wrong $M$
- Applying the gauge fix twice (it's idempotent, but the reverse should
  not re-apply it)

### 1.2 Per-layer error budget

**What to verify.** The reconstruction error does not compound across
layers. Each layer contributes $\mathcal{O}(10^{-9})$ independent of
depth.

**How to verify.**

```python
per_layer_M = []
per_layer_A = []

M_cur, p_cur, A_cur, h_cur = M, p, A, h
for idx, layer in enumerate(model.stack.layers):
    with torch.no_grad():
        M_nx, p_nx, A_nx, h_nx = layer(M_cur, p_cur, A_cur, h_cur)
        M_bk, p_bk, A_bk, h_bk = layer.invert_layer(M_nx, p_nx, A_nx, h_nx)
    Me = ((M_bk - M_cur).norm() / M_cur.norm()).item()
    Ae = ((A_bk - A_cur).norm() / A_cur.norm()).item()
    per_layer_M.append(Me)
    per_layer_A.append(Ae)
    M_cur, p_cur, A_cur, h_cur = M_nx, p_nx, A_nx, h_nx
```

**Expected.** Every per-layer error $< 10^{-6}$. Pilot measured
$3.5 \times 10^{-9}$ to $5.7 \times 10^{-9}$ for $M$ and
$1.9 \times 10^{-9}$ to $2.7 \times 10^{-9}$ for $A$.

**If it fails.** The error compounds. Check that each layer's inverse
does not depend on intermediate states from neighboring layers.

---

## 2. Memory scaling in depth

### 2.1 Peak memory growth

**What to verify.** The stack's peak memory (forward + backward) grows
sub-linearly in depth. In the O(1) limit, peak memory is constant.

**How to verify.**

```python
peaks = {}
for n_layers in [2, 4, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    stack = GWMStack(P, n_layers=n_layers).to('cuda')
    M = torch.randn(N, 8, device='cuda', requires_grad=True)
    p = torch.zeros(N, 8, device='cuda')
    A = torch.randn(N, 3, 10, 10, device='cuda', requires_grad=True)
    h = F.normalize(torch.randn(N, 512, device='cuda'), dim=-1)

    M8, p8, A8, h8 = stack(M, p, A, h)
    (M8.pow(2).sum() + A8.pow(2).sum()).backward()

    torch.cuda.synchronize()
    peaks[n_layers] = torch.cuda.max_memory_allocated() / 1024**2

growth = (peaks[8] - peaks[2]) / peaks[2]
```

**Expected.** `growth < 1.0`. Pilot measured **$0.7\%$** (peak grew from
$1366.7$ MB to $1376.2$ MB for 2→8 layers).

**Note.** This measures the *full* forward + backward cycle. Measuring
only forward will show linear growth — the reversibility property only
shows up when the backward is included.

**If it fails.** Either the reversible autograd isn't being used, or
`save_for_backward` is storing intermediates. Check that
`ReversibleStackFn.forward` only saves `(M_o, p_o, A_o, h_o)` — not
any intermediate states.

### 2.2 Forward/backward ratio

**What to verify.** The full training step (forward + backward) uses less
than 3× the memory of forward alone. A naive stack would use ~$L \times$
(20× at 8 layers).

**How to verify.**

```python
# Forward only
torch.cuda.reset_peak_memory_stats()
M_f = M.clone().requires_grad_(True)
A_f = A.clone().requires_grad_(True)
M_o, p_o, A_o, h_o = stack(M_f, p, A_f, h)
torch.cuda.synchronize()
fwd_peak = torch.cuda.max_memory_allocated() / 1024**2

# Forward + backward
torch.cuda.reset_peak_memory_stats()
M_b = M.clone().requires_grad_(True)
A_b = A.clone().requires_grad_(True)
M_o, p_o, A_o, h_o = stack(M_b, p, A_b, h)
(M_o.pow(2).sum() + A_o.pow(2).sum()).backward()
torch.cuda.synchronize()
bwd_peak = torch.cuda.max_memory_allocated() / 1024**2

ratio = bwd_peak / fwd_peak
```

**Expected.** `ratio <= 3.0`. Pilot measured **$1.76\times$**.

---

## 3. Full-stack gradcheck

**What to verify.** At float64, the reversible stack's backward produces
the same Jacobians as numerical differentiation.

**How to verify.**

```python
P_gc, _ = torch.linalg.qr(torch.randn(512, 10, dtype=torch.float64))
P_gc = P_gc.t().contiguous()
stack = GWMStack(P_gc, n_layers=8).to('cpu').double()

M = (torch.randn(8, 8, dtype=torch.float64) * 0.05).requires_grad_(True)
p = (torch.randn(8, 8, dtype=torch.float64) * 1e-3).requires_grad_(True)
A = torch.randn(8, 3, 10, 10, dtype=torch.float64) * 0.01
A = 0.5 * (A - A.transpose(-1, -2))
A = A.detach().requires_grad_(True)
h = F.normalize(torch.randn(8, 512, dtype=torch.float64), dim=-1)

def stack_fn(M, p, A):
    M_o, p_o, A_o, _ = stack(M, p, A, h)
    return M_o, p_o, A_o

torch.autograd.gradcheck(
    stack_fn, (M, p, A),
    eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5,
)
```

**Expected.** `True`.

**Runtime.** ~5–10 minutes at $N = 8$, 8 layers.

**If it fails.** Numerical precision is not the issue at float64 — an
actual backward-pass error is. Common causes:
- Missing term in `ReversibleStackFn.backward`
- Wrong order of layer inversion
- Using `detach()` where you shouldn't in the replay pass

**Note.** This is the definitive test that the reversible backward
produces correct Jacobians. If it passes, the entire autograd chain
from inputs to outputs is verified.

---

## 4. Holonomy parity at stack level

**What to verify.** The composed holonomy over 8 layers is in
$\mathrm{SO}(10)$: $\det H = +1$, orthogonal to FP32 precision.

**How to verify.**

```python
# Collect A snapshots per layer
A_seq = [A.clone()]
with torch.no_grad():
    M, p, A_cur, h_cur = M, p, A.clone(), h
    for layer in stack.layers:
        M, p, A_cur, h_cur = layer(M, p, A_cur, h_cur)
        A_seq.append(A_cur.clone())

# Compose the holonomy
H_total = np.eye(10)
for i in range(1, len(A_seq)):
    A_mean = A_seq[i].mean(0).cpu().numpy()
    omega = 0.5 * (A_mean[0] - A_mean[0].T) * dt
    H_l = scipy.linalg.expm(omega)
    U, _, Vt = np.linalg.svd(H_l)
    H_l = U @ Vt
    if np.linalg.det(H_l) < 0:
        U[:, -1] *= -1
        H_l = U @ Vt
    H_total = H_l @ H_total

det_H = np.linalg.det(H_total)
orth_H = np.linalg.norm(H_total.T @ H_total - np.eye(10), 'fro')
```

**Expected.**

| Quantity | Reference | Tolerance |
|---|---|---|
| $\|\det H - 1\|$ | $1.06 \times 10^{-6}$ | $< 10^{-5}$ |
| $\|H^\top H - I\|_F$ | $9.35 \times 10^{-7}$ | $< 10^{-5}$ |
| $\|H - I\|_F$ | $4.02 \times 10^{-4}$ | $> 10^{-5}$ (non-trivial) |

**Note on the tolerance.** The pilot uses FP32 throughout. Determinant
accumulation in FP32 is accurate only to $\mathcal{O}(K \cdot
\varepsilon_{32}) \approx 10^{-6}$. Threshold at $10^{-5}$ gives
10× margin.

**If it fails.** Either the composition order is wrong, or the polar
projection of each layer is incorrect. Check that each $H_l$ satisfies
$H_l^\top H_l = I$ before composition.

---

## 5. Gradient flow across layers

**What to verify.** All stack parameters receive gradient. The gradient
does not vanish to zero in the deeper layers.

**How to verify.**

```python
model_g = GWMPilot(...).to('cuda')
pts = torch.rand(2000, 7, device='cuda')

# Encode
M0, p0, A0, h0, _, _ = model_g.encode(pts)

# Forward through stack with live gradients
M8, p8, A8, h8 = model_g.forward_stack(M0, p0, A0, h0)

# Loss
L = A8.pow(2).sum() + 0.01 * M8.pow(2).sum()
L.backward()

# Per-layer V_net gradient norm
for idx, layer in enumerate(model_g.stack.layers):
    p_rep = layer.V_net.net[0].weight
    gn = p_rep.grad.norm().item() if p_rep.grad is not None else 0.0
    print(f"Layer {idx}: grad_norm = {gn:.4e}")
```

**Expected.** Every layer receives nonzero gradient. Pilot measured:

| Layer | $\|\partial L / \partial V_\text{net}\|$ |
|---|---|
| 0 | $4.40 \times 10^{-14}$ |
| 1 | $3.17 \times 10^{-14}$ |
| 2 | $2.28 \times 10^{-14}$ |
| … | … |
| 7 | $6.02 \times 10^{-16}$ |

Note: these are *before* the v7 gradient-path fix. After the fix (see
[`../findings/architectural_limit.md`](../findings/architectural_limit.md)),
the layer-0 gradient is $\sim 10^{-6}$.

**The gradient scaling ratio** $L_0 / L_7$ in the pilot was ~73. The
pilot's threshold is $\le 100$ at init.

**If it fails (all gradients zero).** The `ReversibleStackFn` constraint
(from [`../architecture/components/reversible_stack.md`](../architecture/components/reversible_stack.md) §4):
the backward is only called when at least one input requires grad. Check
that `M0` and `h0` have `requires_grad=True` from the encoder.

**If it fails (only layer 0 gets gradient).** A gate in the deeper layers
is zero. Check `v_correction_scale` and that the `current()` function
uses `W_proj` (for `LearnableJLayer`).

---

## 6. Summary of reference values

| Check | Reference | Tolerance | Artifact |
|---|---|---|---|
| Reconstruction (M, 8 layers) | $1.3 \times 10^{-8}$ | $< 10^{-6}$ | `cell_4_results.json` |
| Reconstruction (p, 8 layers) | $0$ | $< 10^{-5}$ | `cell_4_results.json` |
| Reconstruction (A, 8 layers) | $1.0 \times 10^{-8}$ | $< 10^{-6}$ | `cell_4_results.json` |
| Per-layer error budget | $< 6 \times 10^{-9}$ | $< 10^{-6}$ | `cell_4_results.json` |
| Memory growth (2→8 layers) | $0.7\%$ | $< 100\%$ | `cell_4_results.json` |
| Fwd/bwd ratio | $1.76\times$ | $\le 3\times$ | `cell_4_results.json` |
| Full-stack gradcheck | `True` | — | `cell_4_results.json` |
| $\|\det H - 1\|$ | $1.06 \times 10^{-6}$ | $< 10^{-5}$ | `cell_4_results.json` |
| $\|H^\top H - I\|_F$ | $9.35 \times 10^{-7}$ | $< 10^{-5}$ | `cell_4_results.json` |
| Layer-0 gradient norm | $4.40 \times 10^{-14}$ | $> 0$ | `cell_4_results.json` |
| $L_0 / L_7$ ratio | $73$ | $\le 100$ | `cell_4_results.json` |
| Throughput at $N = 10^6$ | $8.1 \times 10^5$ vox/s | $> 10^4$ | `cell_4_results.json` |

---

## 7. Common failure modes

1. **Reversibility fails at depth but passes per-layer.** The inversion
   is applied in the wrong order. Invert layers in reverse (layer $L-1$
   first, layer 0 last).

2. **Memory grows linearly in depth.** You're not using the reversible
   backward — check that the stack is invoked via `stack(...)` (which
   goes through `ReversibleStackFn`), not `stack._run_layers(...)`.

3. **Gradcheck fails at 8 layers but passes at 1.** The custom backward
   has an accumulation error. Check that `torch.autograd.grad` inside
   `ReversibleStackFn.backward` is called with `retain_graph=True` on
   the L.backward() call.

4. **Gradients are exactly zero for all layers.** Inputs to
   `ReversibleStackFn.apply` don't require grad. Re-encode inside the
   training loop, or use `stack._run_layers` directly.

5. **Throughput is much lower than 800K voxels/s.** This depends on
   `N` and the GPU. The pilot's reference is on a Tesla T4 at
   $N = 10^6$.

---

## 8. References

| Aspect | Reference |
|---|---|
| Mathematical formulation | [`../architecture/formalism.md`](../architecture/formalism.md) |
| Reversible stack component | [`../architecture/components/reversible_stack.md`](../architecture/components/reversible_stack.md) |
| The constraint and its consequences | [`../findings/architectural_limit.md`](../findings/architectural_limit.md) |
| Primary artifact | [`../results/cell_4_results.json`](../results/cell_4_results.json) |