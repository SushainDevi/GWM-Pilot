# Verification — Block 15 (Transport)

This document specifies the verification protocol for the Block 15
"Active Torque" task, which establishes the pilot's definitive negative
result: the A-update rule does not transport information across space.

**Scope.** The task design, the three control regimes (v1, v2, v3), and
the specific checks that demonstrate strict locality.

**Prerequisites.** `numpy`, `torch`, trained model optional. Runtime:
~5 minutes for the diagnostic-only portions.

**References.**

- [`../results/block_15_results.json`](../results/block_15_results.json) (v1)
- [`../results/block_15_v2_results.json`](../results/block_15_v2_results.json) (v2)
- [`../results/block_15_v3_results.json`](../results/block_15_v3_results.json) (v3)
- [`../findings/architectural_limit.md`](../findings/architectural_limit.md) (full development)

---

## 1. Task design

### 1.1 The rod

A straight cylinder along $z \in [-1, 1]$ with fixed geometry:

```python
def make_straight_rod_scene(N_pts=2000, device='cuda'):
    torch.manual_seed(42)
    xyz = torch.randn(N_pts, 3, device=device)
    xyz[:, 2] = torch.rand(N_pts, device=device) * 2.0 - 1.0
    xy_norm = xyz[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-9)
    xyz[:, :2] = xyz[:, :2] / xy_norm
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    bx = 0.9 * F.relu(x)**3
    by = 0.6 * F.relu(y)**3
    bz = 0.4 * F.relu(z)**3
    xyz[:, :2] = xyz[:, :2] * (1.0 + bx + by + bz).unsqueeze(-1)
    xyz = xyz + 0.02 * torch.randn_like(xyz)
    rgb = torch.sigmoid(xyz)
    labels = torch.zeros(N_pts, device=device)
    return torch.cat([xyz, rgb, labels.unsqueeze(-1)], dim=1)
```

**Key property.** Every scene uses the *same* point cloud. Only the
torque input differs.

### 1.2 Torque injection

A random torque $\omega \sim \mathcal{N}(0, I)$ is injected into the
momentum at the top of the rod:

```python
top_mask = (M0_rod[:, 3] > 0.4)          # z-coordinate > 0.4
p0_injected = torch.zeros_like(M0_rod)
p0_injected[:, 4:7] = top_mask.float().unsqueeze(-1) \
    * vector_to_bivector(omega).unsqueeze(0) * 5.0
```

The target rotation is $R_{\text{true}} = \exp_{\mathrm{SO}(3)}(\omega)$.

### 1.3 Decoder reads at bottom

The decoder reads $A$ **only at the bottom 20% of the rod**:

```python
bot_mask = (M8[:, 3] < -0.4)
R_pred = model.decode(A8[bot_mask])
```

**Why this isolates transport.** The bottom of the rod sees no torque
input. Any dependency of $R_{\text{pred}}$ on $\omega$ must go through
$A$-mediated transport from top to bottom.

### 1.4 Distribution properties

The torque distribution is symmetric, so:

$$
\mathbb{E}[R_{\text{true}}] \approx I.
$$

**Null baseline (predict I):** mean error $= 92.6°$.
**Null baseline loss:** $3.8022$.

**The task cannot be solved by outputting a fixed prediction.**

---

## 2. The transport ratio

**Definition.** For two independently-sampled torques $\omega_a, \omega_b$:

$$
\text{transport} \;=\; \frac{\|A_8^{\text{bot}}(\omega_a) - A_8^{\text{bot}}(\omega_b)\|}{\tfrac{1}{2}(\|A_8^{\text{bot}}(\omega_a)\| + \|A_8^{\text{bot}}(\omega_b)\|)}
$$

**Interpretation.**
- $0.0$: $A_8^{\text{bot}}$ is independent of torque. No transport.
- $1.0$: $A_8^{\text{bot}}$ differs as much as its own magnitude. Strong transport.

### 2.1 How to compute it

```python
with torch.no_grad():
    torch.manual_seed(4000)
    wa = torch.randn(3, device='cuda')
    wb = torch.randn(3, device='cuda')

    M0, _, A0, h0, vc, _ = model.encode(pts_rod)
    dM0 = sparse_dM0(M0, vc, grid_res=48)

    pa = torch.zeros_like(M0)
    pa[:, 4:7] = top_mask.float().unsqueeze(-1) \
        * vector_to_bivector(wa).unsqueeze(0) * 5.0
    pb = torch.zeros_like(M0)
    pb[:, 4:7] = top_mask.float().unsqueeze(-1) \
        * vector_to_bivector(wb).unsqueeze(0) * 5.0

    _, _, A8_a, _ = model.forward_stack(M0, pa, A0, h0, d_M=dM0)
    _, _, A8_b, _ = model.forward_stack(M0, pb, A0, h0, d_M=dM0)

    A_a = A8_a[bot_mask]
    A_b = A8_b[bot_mask]
    ratio = (A_a - A_b).norm().item() / \
            (0.5 * (A_a.norm().item() + A_b.norm().item()))
```

**Expected (all three regimes).** `ratio = 0.0000`.

**Interpretation.** Bit-identical $A_8^{\text{bot}}$ under different
top torques.

---

## 3. Three control regimes

The pilot ran three configurations to eliminate every alternative
explanation. All three gave `transport_ratio = 0.0000`.

### 3.1 v1 — naive (stack frozen)

**Setup.** Encode the rod once, reuse the same `M0, h0` across all
scenes. `ReversibleStackFn.forward` runs under `torch.no_grad()`; since
all inputs are detached, backward is never triggered.

**Observed gradient at init:**

$$
\frac{\partial L}{\partial \text{encoder}} = 0, \quad
\frac{\partial L}{\partial W_{\text{proj}}} = 0, \quad
\frac{\partial L}{\partial V_{\text{net}}} = 0.
$$

**Transport ratio:** $0.0000$.

**Why this is uninformative.** The stack is frozen. No learning can
occur regardless of architecture.

### 3.2 v2 — patched (healthy gradients, no fix for locality)

**Fix.** Bypass `ReversibleStackFn` via `stack._run_layers`, and compute
the spatial derivative `dM0` from `M0` via sparse finite differences.

**Observed gradient at init:**

$$
\frac{\partial L}{\partial \text{encoder}} = 1.78 \times 10^{2}, \quad
\frac{\partial L}{\partial W_{\text{proj}}} = 9.82, \quad
\frac{\partial L}{\partial V_{\text{net}}} = 4.50 \times 10^{-2}.
$$

**Transport ratio across all 30 epochs:**

```
epoch  0: 0.0000
epoch 15: 0.0000
epoch 29: 0.0000
```

**Training loss trajectory:** converged from 4.76 to a minimum around
epoch 5 (4.00), then increased to 5.13 by epoch 10, then slowly
decreased to 5.13… actually wait, let me check the JSON. From the
v2 result: `epoch_loss` ends at `3.8090`, but earlier epochs had
diverging behavior. Let me re-read.

Actually from block_15_v2_results.json:
- epoch 0: L=4.6673
- epoch 10: L=3.8259
- epoch 29: L=3.8090

Hmm, so it does slowly decrease. Let me be careful. The transport
ratio is exactly 0.0000 though.

Actually looking at the log I have, the v2 loss went:
- epoch 0: 4.6673
- epoch 1: 3.8841
- epoch 5: 3.8448
- epoch 10: 3.8259
- epoch 29: 3.8090

So it goes down. But transport remains at zero.

**Interpretation.** With healthy gradients and correct `d_M`, the
stack still doesn't transport. This rules out "frozen" and "missing
derivative" as explanations.

### 3.3 v3 — frozen decoder (definitive)

**Setup.** Freeze the decoder at initialization:

```python
class FrozenDecoder(nn.Module):
    def __init__(self, K=10, log_scale=100.0):
        super().__init__()
        self.K = K
        self.register_buffer("log_scale", torch.tensor(log_scale))
        self.register_buffer("mu_weights", torch.tensor([1., 1., 1.]))
        Q = torch.zeros(3, K)
        Q[0, 0] = Q[1, 1] = Q[2, 2] = 1.0
        self.register_buffer("Q", Q)

    def forward(self, A8_bottom):
        A_mean = A8_bottom.mean(dim=0)
        log_H = (self.mu_weights.view(3, 1, 1) * A_mean).sum(dim=0)
        H = torch.matrix_exp(log_H * self.log_scale)
        R_raw = self.Q @ H @ self.Q.t()
        R_pred = newton_schulz_polar(R_raw.unsqueeze(0), n_iter=20).squeeze(0)
        return R_pred
```

The decoder has no learnable parameters (all buffers). With the decoder
frozen, **the only way training loss can decrease is by changing
$A_8^{\text{bot}}$**.

**Observed training loss:**

| Epoch | Loss |
|---|---|
| 0 | 4.7589 |
| 11 | 6.0309 |
| 24 | 3.8280 |
| 29 | **3.7916** |

**Null baseline loss:** $3.8022$.

**Final loss is within 0.28% of the null baseline.**

**Transport ratio across all 30 epochs:** $0.0000$.

**This is the definitive test.**

- Gradients are healthy (`∂L/∂W_proj = 9.82`).
- `d_M` is computed and passed.
- The decoder cannot cheat (no parameters).
- Training loss returns to the null baseline.
- Transport ratio remains exactly zero.

**Conclusion.** $A_8^{\text{bot}}$ is torque-independent. The A-update
does not transport.

### 3.4 How to reproduce the three regimes

Each is a small modification to the training loop:

**v1:** Don't pass `d_M` to `forward_stack`, and don't bypass
`ReversibleStackFn`:

```python
M8, p8, A8, h8 = model.stack(M0, p0, A0, h0)   # with default d_M=None
```

**v2:** Compute `dM0` from `M0` and pass it; bypass `ReversibleStackFn`:

```python
dM0 = sparse_dM0(M0.detach(), vc, grid_res=48)
M8, p8, A8, h8 = model.stack._run_layers(M0, p0, A0, h0, d_M=dM0)
```

**v3:** Same as v2, but replace `model.decoder` with `FrozenDecoder()`.

---

## 4. Rule-out table

The three-regime demonstration eliminates these alternatives:

| Hypothesis | Test | Result |
|---|---|---|
| Stack frozen — no gradient | v2 gradient norms | ❌ ruled out ($10^2$ magnitude) |
| `d_M` missing or zero | v2 explicit `dM0` | ❌ ruled out (nonzero passed) |
| `ReversibleStackFn` bug | v2 uses `_run_layers` | ❌ ruled out (identical result) |
| Caps too tight | v2, v3 caps | ❌ ruled out (`a_cap=100`) |
| Signal too weak | v2, v3 | ❌ ruled out (top $A_8 \approx 0.58$) |
| More training time | 30 epochs | ❌ ruled out (never moves) |
| Decoder overfits | v3 frozen decoder | ❌ ruled out (loss returns to null) |
| Wrong `dt_holo` | v2, v3 (`dt_holo=100`) | ❌ ruled out (small $A$ would rotate) |

**The only remaining explanation is structural:** the A-update is strictly
local.

---

## 5. What to check if reproducing

**Sanity checks before running the transport test:**

1. **Top and bottom voxels exist.**

```python
n_top = (M0[:, 3] > 0.4).sum().item()
n_bot = (M0[:, 3] < -0.4).sum().item()
assert n_top > 10, f"only {n_top} top voxels"
assert n_bot > 10, f"only {n_bot} bottom voxels"
```

Pilot: $n_{top} = 60$, $n_{bot} = 35$.

2. **Rod z-range covers both thresholds.**

```python
z = M0[:, 3]
print(f"z range: [{z.min():.3f}, {z.max():.3f}]")
```

Pilot: $[-0.44, 0.45]$.

3. **A8 magnitude is nonzero.**

```python
A8_mag = (A8.norm() / A8.shape[0]).item()
print(f"A8 per-voxel: {A8_mag:.3e}")
```

Pilot: $\sim 5 \times 10^{-2}$.

4. **`d_M` is nonzero.**

```python
dM0 = sparse_dM0(M0.detach(), vc, 48)
print(f"dM0 magnitude: {(dM0.norm() / dM0.shape[0]).item():.3e}")
assert dM0.abs().max().item() > 0
```

5. **Gradients flow to the stack.**

```python
enc_g = sum(p.grad.norm().item()**2
            for p in model.encoder.parameters() if p.grad is not None) ** 0.5
W_g = sum(p.grad.norm().item()**2
          for layer in model.stack.layers for p in layer.W_proj
          if p.grad is not None) ** 0.5
assert enc_g > 0, "encoder gradient is zero"
assert W_g > 0, "W_proj gradient is zero"
```

---

## 6. What the transport ratio tells you

The transport ratio is a **strict** test: it's zero when the bottom
field is bit-identical under different top torques.

| Transport ratio | Interpretation |
|---|---|
| Exactly $0.0$ | Bit-identical. No transport at all. |
| $< 10^{-6}$ | Float32 noise floor. Effectively zero. |
| $10^{-4}$–$10^{-2}$ | Small but nonzero. Weak transport. |
| $> 0.05$ | Meaningful transport. |
| $> 0.2$ | Strong transport. |

The pilot measures exactly $0.0$ across all three regimes, at all 30
epochs, for two independent torque samples. This is not "small" — it is
*zero*.

---

## 7. Summary of reference values

| Check | v1 | v2 | v3 |
|---|---|---|---|
| `∂L/∂encoder` | $0$ | $1.78 \times 10^{2}$ | $1.78 \times 10^{2}$ |
| `∂L/∂W_proj` | $0$ | $9.82$ | $9.82$ |
| `∂L/∂V_net` | $0$ | $4.50 \times 10^{-2}$ | $4.50 \times 10^{-2}$ |
| `d_M` computed | No | Yes | Yes |
| `ReversibleStackFn` used | Yes | No | No |
| Decoder frozen | No | No | Yes |
| Transport ratio (epoch 0) | $0.0000$ | $0.0000$ | $0.0000$ |
| Transport ratio (epoch 15) | $0.0000$ | $0.0000$ | $0.0000$ |
| Transport ratio (epoch 29) | $0.0000$ | $0.0000$ | $0.0000$ |
| Final training loss | — | $3.8090$ | $3.7916$ |
| Null baseline loss | — | $3.8022$ | $3.8022$ |

---

## 8. References

| Aspect | Reference |
|---|---|
| The full finding | [`../findings/architectural_limit.md`](../findings/architectural_limit.md) |
| Gauge field component | [`../architecture/components/gauge_field.md`](../architecture/components/gauge_field.md) |
| The locality constraint | [`../architecture/formalism.md`](../architecture/formalism.md) §14 |
| v1 artifact | [`../results/block_15_results.json`](../results/block_15_results.json) |
| v2 artifact | [`../results/block_15_v2_results.json`](../results/block_15_v2_results.json) |
| v3 artifact | [`../results/block_15_v3_results.json`](../results/block_15_v3_results.json) |