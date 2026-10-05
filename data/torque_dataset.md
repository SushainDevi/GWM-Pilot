# Torque Dataset

The dataset for the active torque task (Block 15). It provides
(input, target) tuples for predicting the rotation induced by a
torque injected at the top of a straight rod.

**Used by:** Block 15 (Active torque) — definitive negative result.
**References:**
- `results/block_15_results.json` (v1)
- `results/block_15_v2_results.json` (v2)
- `results/block_15_v3_results.json` (v3, definitive)

**Generators:** `make_straight_rod_scene()`, `exp_so3()`,
`vector_to_bivector()` in `block_15.py`

---

## 1. What it generates

Each sample is a tuple `(ω, R_true)`:

| Field | Shape | Content |
|---|---|---|
| `ω` | (3,) | Random torque vector |
| `R_true` | (3, 3) | Target rotation $R_\text{true} = \exp_{\mathrm{SO}(3)}(\omega)$ |

Unlike the ribbon dataset, the input is **not a point cloud** — it's a
torque injected into the momentum at the rod's top. The point cloud
(the rod) is generated once and reused for every scene.

## 2. Rod geometry

A straight cylinder along $z \in [-1, 1]$ with asymmetric bumps:

### 2.1 Point sampling

Sample $N_\text{pts}$ random points in a cube, then:

1. **Uniform $z$:** $z \sim \text{Uniform}(-1, 1)$
2. **Cylinder projection:** normalize $(x, y)$ to unit length
3. **Bump modulation:** multiply $(x, y)$ by
   $1 + 0.9 \cdot \text{ReLU}(x)^3 + 0.6 \cdot \text{ReLU}(y)^3 + 0.4 \cdot \text{ReLU}(z)^3$
4. **Jitter:** add $\mathcal{N}(0, 0.02^2)$ noise

**Why the bumps.** The asymmetric bumps give the PCA surface normals a
well-defined orientation, so the multivector field is not degenerate.
Without them, a smooth cylinder has ambiguous normals.

### 2.2 Fixed geometry

The rod generator uses `torch.manual_seed(42)` at the start, so it
produces the **identical point cloud every time**. Every scene shares
the same rod.

**Consequence.** The stack receives the same `M0, h0` for every scene.
Only the injected torque differs.

## 3. Torque dataset

For each scene $i$:

$$
\omega_i \sim \mathcal{N}(0, I_3), \qquad
R_{\text{true}, i} = \exp_{\mathrm{SO}(3)}(\omega_i).
$$

The exponential map is the standard Rodrigues formula:

$$
\exp_{\mathrm{SO}(3)}(\omega) \;=\; I + \sin\theta \cdot K + (1 - \cos\theta) \cdot K^2
$$

where $\theta = \|\omega\|$ and $K$ is the skew-symmetric matrix of the
unit axis $\omega / \theta$.

## 4. Injection

The torque is injected into the momentum at the top of the rod:

$$
p_0[v, 4:7] \;=\; \begin{cases}
    \text{scale} \cdot \text{biv}(\omega) & \text{if } M_0[v, 3] > 0.4 \\
    0 & \text{otherwise}
\end{cases}
$$

where:
- `scale` = 5.0 (magnitude multiplier)
- `biv(ω)` = the Hodge dual of $\omega$, `[-ω_z, -ω_x, -ω_y]`
- $M_0[v, 3] > 0.4$ selects the top ~20% of voxels

**The decoder** reads $A$ **only at the bottom** of the rod
($M_0[v, 3] < -0.4$).

## 5. Parameters

| Parameter | Value | Role |
|---|---|---|
| `N_PTS` | 2,000 | points per rod |
| `ROD_HALF_LEN` | 1.0 | half length of rod |
| `TORQUE_STD` | 1.0 | torque standard deviation |
| `TORQUE_SCALE` | 5.0 | injection magnitude multiplier |
| `TOP_Z_THRESH` | 0.4 | z-threshold for top mask |
| `BOT_Z_THRESH` | -0.4 | z-threshold for bottom mask |
| `N_TRAIN` | 300 | training scenes |
| `N_HELDOUT` | 30 | held-out scenes |

## 6. Generation procedure

```python
def make_straight_rod_scene(N_pts=2000, device="cuda"):
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

def exp_so3(w):
    theta = w.norm().clamp(min=1e-8)
    axis = w / theta
    K = skew(axis)
    I = torch.eye(3)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

# Torque dataset
torch.manual_seed(15150 + 1000)
train_torques = torch.randn(300, 3) * 1.0
train_targets = torch.stack([exp_so3(t) for t in train_torques])
```

## 7. Sanity checks

**Rod shape and z-range.**

```python
pts_rod.shape == (2000, 7)
pts_rod[:, 2].min() >= -1.05
pts_rod[:, 2].max() <=  1.05
```

**Top/bottom voxel counts** (after encoding):

| Region | Count |
|---|---|
| Total voxels | 1357 |
| Top (z > 0.4) | 60 |
| Bottom (z < -0.4) | 35 |

**Target is a rotation.** Every $R_i$ satisfies $R_i^\top R_i = I$ and
$\det R_i = +1$ to machine precision.

**Distribution over training set.**

| Quantity | Reference |
|---|---|
| $\|\omega\|$ mean | $1.537$ rad ($88.1°$) |
| $\|\omega\|$ max | $3.692$ rad ($211.5°$) |
| $\|R - I\|_F$ mean | $\approx 1.28$ |
| Target mean $\|\mathbb{E}[R] - I\|_F$ | $\approx 0.05$ (near identity) |

**Null baseline (predict I):**

$$
\mathcal{L}_{\text{null}} \;=\; \mathbb{E}\left[\|R - I\|_F^2\right] \;\approx\; 3.80.
$$

The mean error is $92.6°$.

## 8. Distribution properties

**Symmetry.** Torques are $\mathcal{N}(0, I)$-distributed, so

$$
\mathbb{E}[R_{\text{true}}] \;\approx\; I.
$$

**Consequence.** Predicting the identity gives loss $3.80$ — bad. The
task cannot be solved by outputting a fixed prediction.

**Why this defeats the mean-attractor.** Unlike the ribbon task, the
torque task's mean is a poor prediction. Any model that wants loss
below the null must produce torque-dependent outputs.

**Yet the model still fails.** The transport failure is not a
mean-attractor problem. It's that the architecture *cannot* produce
torque-dependent A8 at the bottom, regardless of what the loss rewards.

## 9. File format

The dataset is generated in memory (the rod is the same every time; only
the torques differ). For persistence:

```python
torch.save({
    "pts_rod": pts_rod,             # (2000, 7) — same for all scenes
    "train_torques": train_torques, # (300, 3)
    "train_targets": train_targets, # (300, 3, 3)
    "heldout_torques": heldout_torques,
    "heldout_targets": heldout_targets,
}, "/kaggle/working/torque_dataset.pt")
```

**Why no per-scene point clouds.** The rod geometry is fixed. Only the
torque injection differs. Storing 330 identical point clouds would waste
memory.

## 10. How to regenerate

```python
from gwm_pilot.tasks.active_torque.data import (
    make_straight_rod_scene, exp_so3, make_torque_dataset,
)

pts_rod = make_straight_rod_scene(N_pts=2000)
train_data = make_torque_dataset(300, seed=15150 + 1000)
heldout_data = make_torque_dataset(30, seed=15150 + 2000)
```

**Runtime.** ~2 seconds for the full dataset.

## 11. Why this dataset

The torque task is the pilot's definitive test of the architectural
limit. Its design eliminates every confound:

| Design choice | Purpose |
|---|---|
| Rod geometry fixed | Removes per-scene variation except torque |
| Torque injected only at top | Forces transport for top-to-bottom dependency |
| Decoder reads only at bottom | Ensures $A$ must carry the signal |
| Torque distribution $\mathcal{N}(0, I)$ | Defeats mean-attractor |
| Log-normalized density | Ensures encoder produces M0 |

**The result.** Transport ratio = 0.0000 across 30 epochs, for three
control regimes (v1, v2, v3). Bit-identical $A_8^\text{bot}$ under
different top torques.

**Diagnosis.** See `findings/architectural_limit.md` §3 for the full
development of the three regimes and §5 for the rule-out table.

## 12. Variants

The three control regimes use the same dataset but differ in the
training configuration:

| Regime | Difference | Purpose |
|---|---|---|
| v1 | Naive stack (no `d_M`, uses `ReversibleStackFn`) | Baseline; stack frozen |
| v2 | `d_M` computed, `ReversibleStackFn` bypassed | Healthy gradients |
| v3 | v2 + decoder frozen | Removes overfitting channel |

The dataset is identical in all three. Only the training setup changes.
See `configs/block_15_v2_torque.yaml` and
`configs/block_15_v3_frozen.yaml` for the exact configurations.

## 13. References

| Aspect | Reference |
|---|---|
| Task implementations | `configs/block_15_v2_torque.yaml`, `configs/block_15_v3_frozen.yaml` |
| Primary results | `results/block_15_results.json`, `block_15_v2_results.json`, `block_15_v3_results.json` |
| Verification protocol | `verification/block_15_transport.md` |
| Architectural diagnosis | `findings/architectural_limit.md` §3 |
| Model weights (v3) | `weights/block_15_v3_frozen/` |