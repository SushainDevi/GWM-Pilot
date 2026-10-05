# Encoder

The encoder converts a point cloud into the initial state `(M_0, h_0, A_0)`
that the stack operates on. This document specifies the voxelization, the
multivector construction, and the semantic fiber.

For the mathematical formulation, see
[`../formalism.md#3-multivector-field`](../formalism.md#3-multivector-field).

---

## 1. Structure

```
point cloud (N_pts, 7)
    │
    ▼
┌──────────────────────────────┐
│  EquivariantVoxelizer        │   centroid-radius normalization
│                              │   → sparse voxel grid
└──────────────────────────────┘
    │
    ├──► voxel_coords (N_vox, 3) int32
    ├──► voxel_feats  (N_vox, 7)  (mean_xyz, mean_rgb, first_lab)
    ├──► counts       (N_vox,)    int64
    ├──► scene_info   (2, 3)      (centroid, radius × 3)
    ├──► perm         (N_pts,)    sort permutation
    └──► inv          (N_pts,)    voxel id per sorted point
    │
    ▼
┌──────────────────────────────┐
│  EquivariantCliffordFieldBuilder │   M_0 construction
│                              │   (density, position, PCA normal, orientation)
└──────────────────────────────┘
    │
    ├──► M_0 (N_vox, 8)
    │
    ▼
┌──────────────────────────────┐
│  InvariantSemanticEncoder    │   h_0 from invariant features
│                              │   (density, rgb_norm, magnitudes, direction, scene offset)
└──────────────────────────────┘
    │
    └──► h_0 (N_vox, 512)
```

$A_0$ is initialized to zero (flat connection).

## 2. Voxelizer

### 2.1 Centroid-radius normalization

A point cloud is normalized by its centroid and maximum radius:

$$
x'_{\text{point}} \;=\; \frac{x_{\text{point}} - \bar{x}_{\text{scene}}}{r_{\max}}.
$$

**Why not AABB.** Axis-aligned bounding box normalization
$(x - \min)/(\max - \min)$ is not rotation-equivariant: a rotated scene
has a different AABB, and the normalized coordinates don't transform by
$R_g$. Centroid-radius normalization is: under a global rotation $R_g$
about the centroid, `centroid` is invariant and `r_max` is invariant, so
the normalized coordinates transform by $R_g$.

**Verified.** Centroid invariance error $< 10^{-7}$, radius invariance
error $< 10^{-6}$ in
[`../../results/cell_6_v5_results.json`](../../results/cell_6_v5_results.json).

### 2.2 Sparse voxelization

Points are assigned to integer grid coordinates via floor:

$$
\text{vox} \;=\; \lfloor (x' + 1) \cdot 0.5 \cdot (R - 1) \rfloor
$$

where $R$ is the grid resolution.

**Why sparse.** Only occupied voxels are stored. For a scene with $N_{pts}$
points and grid resolution $R$, the number of occupied voxels is typically
$N_{vox} \ll R^3$. This is important for memory and computation.

**Storage.** Coordinates are sorted by a linearized key (Ki * R² + Kj * R +
Kk) so that lookups are `searchsorted`-based, not hash-based.

### 2.3 Scene info

Two quantities are passed downstream as `scene_info`:
- **centroid** $\bar{x}_{\text{scene}}$: the point cloud's mean
- **radius** $r_{\max}$: the max distance from centroid

Both are invariant under global rotation about the centroid.

## 3. Multivector construction

For each occupied voxel $v$, `EquivariantCliffordFieldBuilder` constructs
$M_v$ from the voxel's centroid, density, and PCA surface normal:

$$
\begin{aligned}
M_v[0] &= \text{log-normalized density} \\
M_v[1:4] &= \text{centroid-radius position} \\
M_v[4:7] &= \text{dual of PCA surface normal} \\
M_v[7] &= \text{outward-orientation sign}
\end{aligned}
$$

See [`clifford_field.md`](clifford_field.md) §3 for the full construction.

**PCA normal.** For each voxel, the within-voxel point block is centered
and SVD'd; the smallest right-singular vector is the local surface normal.
The normal is oriented outward (against the voxel's position vector from
the scene centroid). Voxels with fewer than 3 points have `M_v[4:7] = 0`.

## 4. Semantic fiber $h$

The semantic fiber $h \in \mathbb{R}^{512}$ is the encoder's learned
per-voxel feature. It has gone through five iterations in the pilot.

### 4.1 The five iterations

| Version | Inputs | Purpose |
|---|---|---|
| v1 | density, rgb_norm, label_embed | Baseline invariant encoder |
| v2 | + $\|pos\|$, $\|biv\|$ | Give $h$ scene variation |
| v3 | + bivector direction, $\pm\theta$ angle | Distinguish $+\theta$ from $-\theta$ |
| v4 | + scene-level bivector mean (broadcast) | Explicit scene descriptor |
| v5 | + scene projection offset | Strongest scene-conditioning |

### 4.2 The v5 encoder

**Inputs (265 → 268 dims):**

$$
[\text{density}, \; \|rgb\|, \; \|M[1:4]\|, \; \|M[4:7]\|, \; \text{biv\_unit}, \; \sin\!\theta_{\text{biv}}, \; \cos\!\theta_{\text{biv}}, \; \overline{\text{biv}}, \; \text{label\_embed}]
$$

where:
- density = `M[:, 0]`
- rgb_norm = $\|rgb\|$
- pos_mag = $\|M[:, 1:4]\|$
- biv_mag = $\|M[:, 4:7]\|$
- biv_unit = $M[:, 4:7] / \|M[:, 4:7]\|$
- sin/cos of $\theta_{\text{biv}} = \text{atan2}(B_{yz}, B_{zx})$
- scene-mean bivector $\overline{\text{biv}} = \text{mean}_v(M[:, 4:7])$, broadcast

**MLP:**

```
Linear(265, 512) → GELU → Linear(512, 512) → LayerNorm
```

**Scene offset (v5):**

$$
h_v \;=\; \text{MLP}(x_v) \;+\; \text{scene\_proj}(\text{scene\_vec})
$$

where `scene_vec` = `[biv_mean(3), twist_mag(1), twist_sign(1)]` and
`scene_proj` is a linear layer (5 → 512) with small init (std 0.1).

### 4.3 Verified scene sensitivity

| Check | Reference | Value |
|---|---|---|
| $h_0$ varies across twists | `results/cell_6_v5_results.json` | $0.58$–$4.49$ |
| $h_0$ distinguishes $+\theta$ from $-\theta$ | `results/cell_6_v5_results.json` | $4.49$ |
| Scene-level variance | `results/cell_6_v5_results.json` | $2.13$ |
| Scene offset relative magnitude | `results/cell_6_v5_results.json` | $0.10$ |

**Reproduce.** See [`../../verification/cell_6_v5_encoder.md`](../../verification/cell_6_v5_encoder.md).

## 5. What the encoder produces

The `VoxelEncoder.forward` returns:

```python
M0, h0, A0, voxel_coords, scene_info
```

with:
- `M0`: (N_vox, 8) multivector field
- `h0`: (N_vox, 512) semantic fiber
- `A0`: (N_vox, 3, 10, 10) zero tensor
- `voxel_coords`: (N_vox, 3) int32 grid coordinates
- `scene_info`: (2, 3) float32

## 6. The invariance/covariance balance

The encoder is designed so that:
- `M0` is **covariant** under global rotation (positions and normals
  transform by $R_g$; magnitudes and signs are invariant).
- `h0` is **invariant** under global rotation for v1–v2; **not invariant**
  for v3–v5 (direction features are covariant).

**Trade-off.** The v3–v5 encoder breaks strict rotation-invariance to give
$h$ the twist-direction information needed for the ribbon task. This is
acceptable when the scene is at canonical orientation (as in the ribbon
task) but not for tasks requiring strict equivariance (as in the rotation
task).

## 7. References

| Aspect | Reference |
|---|---|
| Mathematical formulation | [`../formalism.md`](../formalism.md) |
| Encoder verification protocol | [`../../verification/cell_6_v5_encoder.md`](../../verification/cell_6_v5_encoder.md) |
| Multivector details | [`clifford_field.md`](clifford_field.md) |
| Encoder iteration artifacts | `results/cell_6_results.json` … `results/cell_6_v5_results.json` |