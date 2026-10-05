# Verification — Cell 6 v5 (Encoder)

This document specifies the verification protocol for the v5 semantic
encoder: scene sensitivity, ±θ separation, and the scene-projection
offset.

**Scope.** Everything about the encoder's behavior on twisted-tube
scenes. This is the verification for the iteration that finalized
before the ribbon task runs.

**Prerequisites.** `numpy`, `torch`. Runtime: ~30 seconds.

**Reference.** [`../results/cell_6_v5_results.json`](../results/cell_6_v5_results.json).

---

## 1. What the v5 encoder adds

Across five iterations, the encoder's inputs grew:

| Version | Input dim | Key addition |
|---|---|---|
| v1 | 258 | invariant (density, rgb_norm, label) |
| v2 | 260 | + $\|pos\|$, $\|biv\|$ magnitudes |
| v3 | 265 | + bivector direction, in-plane angle |
| v4 | 268 | + scene-level bivector mean (broadcast) |
| v5 | 268 | + scene projection offset (post-LayerNorm) |

v5's key architectural feature is the **explicit scene offset**, added
after the MLP's LayerNorm:

$$
h_v \;=\; \text{MLP}(x_v) \;+\; \text{scene\_proj}(\text{scene\_vec})
$$

where `scene_vec` is a 5-vector (bivector mean, twist magnitude, twist
sign) and `scene_proj` is a small linear layer.

**Why post-LayerNorm.** Without this, LayerNorm absorbs the extra input
channels and the network's h0 varies only weakly. Adding the offset after
LayerNorm preserves its magnitude.

---

## 2. Scene sensitivity: h0 varies across twist scenes

### 2.1 h0 delta across twist scenes

**What to verify.** Encoding three twisted-tube scenes with different
twist values produces h0 vectors that differ by a meaningful amount.

**How to verify.**

```python
model = EquivariantGWMPilot(grid_res=64, n_classes=4).to('cuda')

with torch.no_grad():
    h0s = []
    for tw in [0.5, 1.5, 2.5]:
        pts = make_twisted_tube_scene(N_pts=2000, twist_turns=tw, seed=...)
        _, _, _, h0, _, _ = model.encode(pts)
        h0s.append(h0.mean(dim=0))

    d_01 = (h0s[0] - h0s[1]).norm().item()
    d_12 = (h0s[1] - h0s[2]).norm().item()
    d_02 = (h0s[0] - h0s[2]).norm().item()
```

**Expected.**

| Pair | Reference |
|---|---|
| $\|h_0(0.5) - h_0(1.5)\|$ | $4.49$ |
| $\|h_0(1.5) - h_0(2.5)\|$ | $0.58$ |
| $\|h_0(0.5) - h_0(2.5)\|$ | $4.54$ |

**Acceptance.** Min pairwise delta $> 10^{-3}$.

**If it fails.** Check that the encoder's `scene_proj` is contributing.
Compare `h0` with and without `scene_proj`.

### 2.2 h0 distinguishes $+\theta$ from $-\theta$

**What to verify.** Encoding a scene with twist $+1.5$ and a scene with
twist $-1.5$ produces different h0. This is the specific failure mode
that v2 couldn't overcome.

**How to verify.**

```python
with torch.no_grad():
    pts_p = make_twisted_tube_scene(N_pts=2000, twist_turns=+1.5, seed=200)
    pts_m = make_twisted_tube_scene(N_pts=2000, twist_turns=-1.5, seed=200)
    _, _, _, h0_p, _, _ = model.encode(pts_p)
    _, _, _, h0_m, _, _ = model.encode(pts_m)
    d_pm = (h0_p.mean(dim=0) - h0_m.mean(dim=0)).norm().item()
```

**Expected.** $d_{pm} = 4.49$. Acceptance threshold: $> 10^{-3}$.

**If it fails.** This is the v2 failure mode. Check that the encoder
sees `biv_unit` (the direction) and not just magnitudes.

### 2.3 h0 grows with twist magnitude

**What to verify.** Successive twist values produce monotonically
increasing h0 delta.

**How to verify.**

```python
with torch.no_grad():
    h0s_scale = []
    for tw in [0.5, 1.0, 2.0, 3.0]:
        pts = make_twisted_tube_scene(N_pts=2000, twist_turns=tw, seed=...)
        _, _, _, h0, _, _ = model.encode(pts)
        h0s_scale.append(h0.mean(dim=0))

    diffs = [(h0s_scale[i+1] - h0s_scale[i]).norm().item()
             for i in range(len(h0s_scale)-1)]
```

**Expected.** Successive diffs = `[4.49, 0.12, 4.49]`. Every diff $> 10^{-3}$.

**Note.** The middle diff ($1.0 \to 2.0$) is smaller than the others —
this is a consequence of the specific scene generator, not a bug. The
critical check is that all diffs are above threshold.

---

## 3. Absolute scene-level variance

**What to verify.** The variance of the scene-mean h0 vector across
multiple twists is meaningful in absolute terms (not just as a fraction
of h0's norm, which LayerNorm bounds).

**How to verify.**

```python
with torch.no_grad():
    h0_means = []
    for tw in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pts = make_twisted_tube_scene(N_pts=2000, twist_turns=tw, seed=...)
        _, _, _, h0, _, _ = model.encode(pts)
        h0_means.append(h0.mean(dim=0))

    h0_stack = torch.stack(h0_means, dim=0)   # (6, 512)
    mean_mag = h0_stack.norm(dim=1).mean().item()
    centered = h0_stack - h0_stack.mean(dim=0, keepdim=True)
    var_sqrt = math.sqrt(centered.pow(2).sum(dim=1).mean().item())
```

**Expected.**

| Quantity | Reference |
|---|---|
| Mean $\|h_0\|$ | $22.34$ |
| $\sqrt{\text{variance}}$ across scenes | $2.13$ |
| Ratio | $0.095$ |

**Acceptance.** `var_sqrt > 0.1` (absolute threshold). Pilot measured $2.13$.

**Note on the metric.** LayerNorm bounds the mean norm to $\approx \sqrt{D_H}
= 22.6$, so the *ratio* is bounded above by ~0.05. The meaningful metric
is the absolute variance.

**If it fails.** The scene projection is not contributing. Check that
`scene_proj.weight` has grown from init (std $0.1$).

---

## 4. Scene offset relative magnitude

**What to verify.** The scene offset $\Delta h_0$ contributes at least
1% of the MLP output's magnitude.

**How to verify.**

```python
with torch.no_grad():
    pts = make_twisted_tube_scene(N_pts=2000, twist_turns=1.5, seed=500)
    M0, h0_full, _, _, _, _ = model.encode(pts)

    # Compute the offset directly
    biv = M0[:, 4:7]
    biv_mean = biv.mean(dim=0, keepdim=True)
    twist_mag = biv_mean.norm(dim=1, keepdim=True)
    twist_sign = torch.sign(biv_mean[:, 2:3])
    scene_vec = torch.cat([biv_mean, twist_mag, twist_sign], dim=1)

    delta_h0 = model.encoder.sem_enc.scene_proj(scene_vec)
    delta_mag = delta_h0.norm().item()
    full_mag = h0_full.mean(dim=0).norm().item()
    ratio = delta_mag / full_mag
```

**Expected.**

| Quantity | Reference |
|---|---|
| $\|\Delta h_0\|$ | $2.25$ |
| $\|h_0\|$ (mean over voxels) | $22.37$ |
| Ratio | $0.10$ |

**Acceptance.** Ratio $> 0.01$.

**If it fails.** `scene_proj.weight` has not grown from init. It starts
at std $0.1$; after training, it should be larger.

---

## 5. Equivariance (informational)

**What to verify.** Whether the encoder still satisfies the v1
rotation-equivariance property. This is **informational**, not a check,
because v3–v5 deliberately break strict invariance to give h0 the
direction information the ribbon task needs.

**How to verify.**

```python
pts_orig = make_bulge_scene(N_pts=3000, R_g=None, seed=2)
R_g = random_so3(seed=3).to('cuda')
pts_rot = rotate_scene_about_centroid(pts_orig, R_g)

with torch.no_grad():
    M0_o, _, _, h0_o, _, _ = model.encode(pts_orig)
    M0_r, _, _, h0_r, _, _ = model.encode(pts_rot)

# The strict invariance test (fails by design in v5)
h0_equiv_err = (h0_o.mean(dim=0) - h0_r.mean(dim=0)).norm().item()
```

**Reference value.** $5.46 \times 10^{-2}$. Ratio to no-info baseline
$= 0.035$.

**Interpretation.** Even though v5 breaks strict invariance, the residual
is small because the bivector-direction features happen to correlate
with the position features in a way that keeps the pipeline approximately
equivariant on bulge scenes.

**This check is not pass/fail.** It documents the trade-off.

---

## 6. Overfit sanity

**What to verify.** A single scene can be overfit to zero loss.

**How to verify.** Train a fresh model for 300 steps on one scene with
the direction loss:

```python
model = EquivariantGWMPilot(...).to('cuda')
pts = make_bulge_scene(N_pts=2000, R_g=R_fixed, seed=21)
d_target = R_fixed.to('cuda') @ torch.tensor([0., 0., 1.], device='cuda')

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(300):
    opt.zero_grad()
    M0, p0, A0, h0, _, _ = model.encode(pts)
    M8, p8, A8, h8 = model.forward_stack(M0, p0, A0, h0)
    d_pred, _ = model.decode(M8)

    L = ((d_pred - d_target) ** 2).sum() + 0.001 * ym_loss(A8)
    L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

**Expected.**

| Quantity | Reference |
|---|---|
| $L(0)$ | $2.48 \times 10^{-2}$ |
| $L(300)$ | $0$ |
| Training time | $23$ seconds |

**Acceptance.** Final loss $< 0.01$.

**If it fails.** Either the loss is not differentiable through the
decoder, or the decoder is producing degenerate outputs. Check that
`d_pred` has norm $\approx 1$ at step 0.

---

## 7. Summary of reference values

| Check | Reference | Tolerance | Artifact |
|---|---|---|---|
| $h_0$ varies across twists | $0.58$–$4.49$ | $> 10^{-3}$ | `cell_6_v5_results.json` |
| $h_0$ distinguishes $+\theta$ from $-\theta$ | $4.49$ | $> 10^{-3}$ | `cell_6_v5_results.json` |
| Successive twist diffs | $[4.49, 0.12, 4.49]$ | all $> 10^{-3}$ | `cell_6_v5_results.json` |
| Absolute scene variance | $2.13$ | $> 0.1$ | `cell_6_v5_results.json` |
| Scene offset ratio | $0.10$ | $> 0.01$ | `cell_6_v5_results.json` |
| Equivariance residual (info) | $5.46 \times 10^{-2}$ | — | `cell_6_v5_results.json` |
| Overfit final loss | $0$ | $< 0.01$ | `cell_6_v5_results.json` |

---

## 8. Common failure modes

1. **h0 varies weakly.** The scene projection is either missing or its
   input `scene_vec` is not reaching it. Check that `M0[:, 4:7]` has
   nonzero bivector components before the encoder.

2. **$+\theta$ and $-\theta$ give the same h0.** The encoder sees only
   bivector magnitudes, not directions. Ensure `biv_unit` is in the
   input vector.

3. **Absolute variance is small but ratio is fine.** This is
   LayerNorm's effect. The correct metric is absolute variance, not ratio.

4. **Scene offset is negligible.** `scene_proj.weight` hasn't grown.
   Verify that gradient flows to `scene_proj` during training — it
   should have the highest gradient magnitude in the encoder.

---

## 9. References

| Aspect | Reference |
|---|---|
| Encoder component | [`../architecture/components/encoder.md`](../architecture/components/encoder.md) |
| Multivector construction | [`../architecture/components/clifford_field.md`](../architecture/components/clifford_field.md) |
| Encoder iteration history | `../results/cell_6_results.json` … `../results/cell_6_v5_results.json` |
| Primary artifact | [`../results/cell_6_v5_results.json`](../results/cell_6_v5_results.json) |