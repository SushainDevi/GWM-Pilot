# Ribbon Dataset

The dataset for the kinematic ribbon task (Cell 14). It provides
(input, target) tuples for predicting the relative rotation of a
twisted elliptical tube.

**Used by:** Cell 14 (Kinematic ribbon) — partial negative result.
**Reference:** `results/cell_14_results.json`, `cell_14_iterations.json`
**Generator:** `make_ribbon_scene()` in `cell_14.py`

---

## 1. What it generates

Each sample is a tuple `(pts, R_AB, twist_turns)`:

| Field | Shape | Content |
|---|---|---|
| `pts` | (N_pts, 7) | Point cloud with (x, y, z, r, g, b, label) |
| `R_AB` | (3, 3) | Relative rotation between base and tip (target) |
| `twist_turns` | scalar | Total twist in turns |

The task is to predict `R_AB` from `pts`.

## 2. Geometry

### 2.1 The tube

A twisted elliptical tube along the $z$-axis:

- **Length:** $L = 10$
- **Cross-section:** ellipse with semi-axes $a = 1.5$, $b = 0.5$
- **Twist:** linear with $z$: $\theta(z) = \theta_\text{total} \cdot z / L$

### 2.2 Point sampling

Three regions with labels:

| Region | $z$-range | Point count | Label |
|---|---|---|---|
| Base cap | $[0, 0.6]$ | 15% of $N_\text{pts}$ | 1 |
| Bulk | $[1.2, 8.8]$ | 70% | 0 |
| Tip cap | $[9.4, 10.0]$ | 15% | 2 |

**Gaps** between regions (e.g., $z \in (0.6, 1.2)$) are deliberate: they
ensure the anchors are spatially separated from the bulk so that
downstream decoders can identify them via the label channel.

### 2.3 Cross-section sampling

For a point at position $s$ along the tube's length:

$$
\theta \;=\; \theta_\text{total} \cdot s / L
$$

Sample a random azimuthal angle $\phi \sim \text{Uniform}(0, 2\pi)$ and
radial jitter $\rho \sim \text{Uniform}(0.85, 1.15)$. Then:

$$
\begin{aligned}
x_\text{local} &= \rho \, a \cos\phi \\
y_\text{local} &= \rho \, b \sin\phi \\
x_\text{world} &= x_\text{local} \cos\theta - y_\text{local} \sin\theta \\
y_\text{world} &= x_\text{local} \sin\theta + y_\text{local} \cos\theta \\
z_\text{world} &= s
\end{aligned}
$$

### 2.4 RGB and label

- **RGB:** $\sigma(\text{xyz})$, purely cosmetic
- **Label:** 0 (bulk), 1 (base cap), 2 (tip cap)

## 3. Target

The relative rotation is a rotation about the $z$-axis by the total twist:

$$
R_{AB} \;=\; R_z(\theta_\text{total}) \;=\;
\begin{pmatrix}
\cos\theta_\text{total} & -\sin\theta_\text{total} & 0 \\
\sin\theta_\text{total} &  \cos\theta_\text{total} & 0 \\
0 & 0 & 1
\end{pmatrix}.
$$

**Target distribution.** Twist is sampled uniformly in
$[0.5, 2.5]$ turns, so $\theta_\text{total} \in [\pi, 5\pi]$. The wrapped
mean is approximately $2\pi \cdot 1.46 \approx 9.18$ rad, wrapped to
$\approx 2.90$ rad.

## 4. Parameters

| Parameter | Value | Role |
|---|---|---|
| `L_TUBE` | 10.0 | total tube length |
| `ELLIPSE_A` | 1.5 | semi-major axis |
| `ELLIPSE_B` | 0.5 | semi-minor axis |
| `N_PTS` | 4,000 | points per scene |
| `TWIST_RANGE` | (0.5, 2.5) | twist in turns |
| `N_TRAIN` | 300 | training scenes |
| `N_HELDOUT` | 30 | held-out scenes |
| `SEED_OFFSET_TRAIN` | 1000 | offset for train seeds |
| `SEED_OFFSET_HELDOUT` | 5000 | offset for held-out seeds |

## 5. Generation procedure

```python
def make_ribbon_scene(N_pts=4000, twist_range=(0.5, 2.5),
                       seed=0, device="cuda"):
    torch.manual_seed(seed)
    twist_turns = uniform(twist_range)
    total_twist = 2 * pi * twist_turns

    N_base = int(0.15 * N_pts)
    N_tip  = int(0.15 * N_pts)
    N_bulk = N_pts - N_base - N_tip

    base = sample_cap(N_base, 0.0, 0.6, label=1)
    bulk = sample_cap(N_bulk, 1.2, 8.8, label=0)
    tip  = sample_cap(N_tip,  9.4, 10.0, label=2)

    pts = concat([base, bulk, tip], dim=0)
    R_AB = rot_z(total_twist)
    return pts, R_AB, twist_turns
```

**Determinism.** Each scene is fully determined by its seed. The same
seed produces the same point cloud and target.

## 6. Sanity checks

**Shape.**

```python
pts.shape == (4000, 7)
R_AB.shape == (3, 3)
```

**Target is a rotation.**

```python
assert torch.allclose(R_AB.T @ R_AB, torch.eye(3), atol=1e-5)
assert abs(torch.linalg.det(R_AB).item() - 1.0) < 1e-5
```

**Distribution over training set.**

| Quantity | Reference |
|---|---|
| `twist_turns` mean | 1.461 |
| `twist_turns` min | 0.504 |
| `twist_turns` max | 2.498 |
| Target mean $\|R - I\|_F$ | 1.10 |

**Baselines.**

The null baseline (always predict $I$) has mean loss:

$$
\mathcal{L}_{I} \;=\; \mathbb{E}\left[\|R_z(\theta) - I\|_F^2\right]
\;=\; \mathbb{E}\left[4(1 - \cos\theta)\right] \;\approx\; 4.12.
$$

## 7. Distribution properties

**Symmetry.** Twists are sampled from $[0.5, 2.5]$ — all positive.
Predicting the mean twist gives loss $\approx 0.50$, which is well below
the trivial-I baseline. **This is the mean-attractor the ribbon task
suffers from.**

**Why the model collapses to the mean.** The architecture encodes the
scene into h0, but the stack cannot route h0 to the Wilson-line decoder
across a spatial path. The best a local A-integral can do is produce the
same output regardless of scene. That output converges to the mean twist.

**Comparison with the torque task.** The torque task's targets are
$\mathcal{N}(0, I)$-distributed, so the mean is $I$ — a bad prediction
(loss ~3.80). The ribbon task's mean is a good prediction (loss ~0.50).
This is why the ribbon task has the mean-attractor problem and the
torque task does not.

## 8. File format

The dataset is generated in memory. For persistence:

```python
torch.save({
    "pts": [s[0] for s in scenes],
    "R_AB": [s[1] for s in scenes],
    "twist_turns": [s[2] for s in scenes],
}, "/kaggle/working/ribbon_dataset.pt")
```

**Alternative.** Since generation is fast (~0.2 seconds for 300 scenes),
the dataset can be regenerated on demand rather than persisted.

## 9. How to regenerate

```python
from gwm_pilot.tasks.ribbon.data import make_ribbon_dataset

train_data = make_ribbon_dataset(300, seed=1000)
heldout_data = make_ribbon_dataset(30, seed=5000)
```

**Runtime.** ~0.2 seconds for the full dataset.

## 10. Why this dataset

The ribbon task is designed to test whether the architecture can learn a
*non-local* geometric transformation. The Wilson-line decoder requires
$A$ to carry information along a path of ~40 voxels. This is the
architectural limit's direct target.

**Reference result:** probe err_z = 0.956 rad (55°), held-out err_z =
1.660 rad (95°). The probe works; the held-out fails. The model
converges to predicting the mean twist.

**Diagnosis.** See `findings/architectural_limit.md` §4 and the
`cell_14_iterations.json` artifact for the eight iterations that led to
this conclusion.

## 11. References

| Aspect | Reference |
|---|---|
| Task implementation | `configs/cell_14_ribbon.yaml` |
| Primary result | `results/cell_14_results.json` |
| Iteration history | `results/cell_14_iterations.json` |
| Architectural diagnosis | `findings/architectural_limit.md` §4 |
| Verification protocol | `verification/cell_6_v5_encoder.md` |