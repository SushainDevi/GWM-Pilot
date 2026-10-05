# FDTD Dataset

The dataset for the FDTD field prediction task (Cell 12). It provides
(input, target) tuples for a single step of a 3D wave equation on a
periodic grid.

**Used by:** Cell 12 (FDTD field prediction) — the pilot's positive result.
**Reference:** `results/block_9_v2_summary.json`
**Generator:** `make_fdtd_dataset()` in `block_9.py`

---

## 1. What it generates

Each sample is a tuple `(A_0, E_0, B_0) → A_1` on a $16^3$ periodic grid:

| Array | Shape | Content |
|---|---|---|
| `A_0` | (3, 16, 16, 16) | Vector field at time $t$ |
| `E_0` | (3, 16, 16, 16) | Small random perturbation (input feature only) |
| `B_0` | (3, 16, 16, 16) | Discrete curl of $A_0$ (input feature only) |
| `A_1` | (3, 16, 16, 16) | Vector field at time $t + \Delta t$ (target) |

The task is to predict `A_1` from `(A_0, E_0, B_0)`.

## 2. Physics

The dataset simulates one step of the 3D wave equation on a periodic
domain:

$$
\frac{\partial^2 A}{\partial t^2} \;=\; c^2 \nabla^2 A
$$

discretized by explicit central-time / central-space (FDTD leapfrog):

$$
A^{n+1} \;=\; 2 A^n - A^{n-1} + \left(\frac{c \, \Delta t}{\Delta x}\right)^2 \nabla^2 A^n.
$$

**Stability.** The CFL number is

$$
\text{cour} \;=\; \left(\frac{c \, \Delta t}{\Delta x}\right)^2 \;=\; 0.25.
$$

In 3D, stability requires $\text{cour} \le 1/3 \approx 0.333$. The pilot's
value of 0.25 is stable with margin.

## 3. Parameters

| Parameter | Value | Role |
|---|---|---|
| `L` | 16 | grid side length |
| `dt` | 0.5 | time step |
| `c` | 1.0 | wave speed |
| `dx` | 1.0 | spatial spacing |
| `N_SAMPLES` | 10,000 | total samples |
| `N_CHUNKS` | 10 | disk chunks |
| `CHUNK_SIZE` | 1,000 | samples per chunk |
| `N_MODES` | 4 | Fourier modes summed per sample |
| `MODE_AMP` | 0.5 | amplitude per mode |
| `E_AMP` | 0.05 | $E_0$ perturbation scale |

## 4. Generation procedure

### 4.1 Initial field $A_0$

For each sample, sum $N_\text{modes}$ random plane waves:

$$
A_0(\mathbf{x}) \;=\; \text{Re}\!\left[\sum_{m=1}^{N_\text{modes}}
    \text{amp}_m \, e^{i (\mathbf{k}_m \cdot \mathbf{x} + \phi_m)}\right]
$$

where:
- $\mathbf{k}_m \in \{-3, \ldots, 3\}^3$ integer wavenumbers, $\mathbf{k}_m \ne 0$
- $\phi_m \sim \text{Uniform}(0, 2\pi)$ random phase
- $\text{amp}_m \sim \text{Uniform}(0.5, 1.5) \cdot \text{MODE\_AMP}$

The imaginary part is discarded; each component of $A_0$ uses its own
random modes.

**Zero-mean.** After summing modes, each component is mean-subtracted so
the field integrates to zero on the periodic domain.

### 4.2 $E_0$ (input feature)

$$
E_0 \;\sim\; \mathcal{N}(0, \text{E\_AMP}^2)
$$

elementwise Gaussian noise. This is a random "electric field" that
perturbs the update through `A_0 + dt·E_0` but is not part of the
dynamical system.

### 4.3 $A_1$ (target)

One FDTD step from $A_0$ with zero initial velocity:

$$
A_1 \;=\; 2 A_0 - A_{-1} + \text{cour} \cdot \nabla^2 A_0
$$

with $A_{-1} = A_0$ (zero velocity assumption). The Laplacian is computed
on the periodic grid via `numpy.roll`:

$$
\nabla^2 A(\mathbf{x}) \;\approx\; \sum_{\mu \in \{x, y, z\}}
    \left[ A(\mathbf{x} + \hat{\mu}) + A(\mathbf{x} - \hat{\mu}) - 2 A(\mathbf{x}) \right].
$$

### 4.4 $B_0$ (input feature)

Discrete curl of $A_0$:

$$
B_0 \;=\; \nabla \times A_0
$$

via central differences on the periodic grid:

$$
(\nabla \times A)_i \;=\; \epsilon_{ijk} \partial_j A_k,
\qquad
\partial_j A_k \;\approx\; \frac{A_k(\mathbf{x} + \hat{\jmath}) - A_k(\mathbf{x} - \hat{\jmath})}{2 \, \Delta x}.
$$

$B_0$ is provided as a physically-consistent "magnetic field" input to
the encoder.

## 5. Sanity checks

After generation, verify:

**Array shapes.**

```python
assert A_0.shape == (10000, 3, 16, 16, 16)
assert A_1.shape == (10000, 3, 16, 16, 16)
```

**Frobenius norms (across samples).**

| Quantity | Reference |
|---|---|
| $\|A_0\|$ mean | $\approx 81.3$ |
| $\|A_1 - A_0\|$ mean | $\approx 37.8$ |
| $\|E_0\|$ mean | $\approx 5.54$ |
| $\|B_0\|$ mean | $\approx 75.0$ |

**Integrity.** $A_1$ should satisfy the FDTD update exactly (verifiable
by recomputing it from $A_0$).

## 6. Distribution properties

**Target mean.** Over the dataset:

$$
\mathbb{E}[A_1] \;\approx\; \mathbb{E}[A_0] \;\approx\; 0.
$$

The target is zero-mean. Predicting $\bar{A}_1 = 0$ gives a
MSE of $\approx \mathbb{E}[\|A_1\|^2]/N_\text{voxels} \approx 80$ — very
bad. The trivial "predict identity" baseline (predicting $A_1 = A_0$)
gives MSE $\approx 0.117$, which is the reference baseline in the pilot.

**Why this defeats the mean-attractor.** Unlike the ribbon task, the
FDTD task cannot be solved by outputting a fixed prediction. The target
varies per sample. Per-voxel processing is required.

## 7. File format

Samples are stored in 10 chunks as compressed `.npz` files:

```
/kaggle/working/fdtd_data/
    traj_chunk_000.npz    (1000 samples, ~150 MB)
    traj_chunk_001.npz
    ...
    traj_chunk_009.npz
```

Each chunk contains:

```python
{
    "A_0": np.ndarray,    # (1000, 3, 16, 16, 16) float32
    "A_1": np.ndarray,    # (1000, 3, 16, 16, 16) float32
    "E_0": np.ndarray,    # (1000, 3, 16, 16, 16) float32
    "B_0": np.ndarray,    # (1000, 3, 16, 16, 16) float32
}
```

**Total disk:** ~1.5 GB.

## 8. How to regenerate

```python
from gwm_pilot.tasks.fdtd.data import generate_fdtd_dataset

A_0, A_1, E_0, B_0 = generate_fdtd_dataset(
    dir="/kaggle/working/fdtd_data",
    force=False,          # if True, regenerate even if cached
)
```

**Runtime:** ~75 seconds for the full 10 chunks on a T4.

**Determinism.** The generator is seeded per chunk
(`SEED + 10_000 + chunk_idx`). Regenerating gives identical samples.

## 9. Split

The 10,000 samples are split by a fixed permutation (seed 4242):

| Split | Fraction | Count |
|---|---|---|
| Train | 80% | 8,000 |
| Val | 10% | 1,000 |
| Test | 10% | 1,000 |

**Split procedure.** `np.random.seed(SEED); perm = np.random.permutation(N)`.
The first 8,000 go to train, the next 1,000 to val, the last 1,000 to test.

**Reshape for the model.** Each sample `(3, 16, 16, 16)` is reshaped to
`(N_vox, 3)` where `N_vox = 4096`. The channel order is permuted so the
model sees voxel-major tensors: `(B, V, 3)` with `V = 4096`.

## 10. Why this dataset

The FDTD task is a local-to-local mapping: predict per-voxel output from
per-voxel input. This is the exact class of task that the pilot's
architecture can solve.

**Reference result:** test MSE $0.022$ vs trivial baseline $0.117$, a
$5.27\times$ improvement. Supervised MLP baseline: $0.020$, ratio $1.10$.

## 11. References

| Aspect | Reference |
|---|---|
| Task implementation | `configs/cell_12_fdtd.yaml` |
| Primary result | `results/block_9_v2_summary.json` |
| Architecture constraint | `findings/architectural_limit.md` §4 |