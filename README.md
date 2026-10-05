# GWM-Pilot v1

**A reversible Yang-Mills stack for geometric deep learning.** And the
specific architectural limit that defines its envelope.

This repository documents a pilot study on a new class of geometric neural
architecture. It contains the verification results, trained models, and
technical finding from that pilot. Every claim below is backed by a
specific measurement in `results/` — no numbers in this README are
asserted without a reference.

---

## Verified properties

The architecture has seven properties that were verified to machine
precision in the pilot. Each is listed with its measured value and the
JSON that establishes it.

| Property | Measured | Check |
|---|---|---|
| **O(1) memory in depth** | Peak grows **0.7%** from 2→8 layers | [`cell_4_results.json`](results/cell_4_results.json) |
| **Exact reversibility** | **1.3×10⁻⁸** relative reconstruction error (fp32, 8 layers) | [`cell_4_results.json`](results/cell_4_results.json) |
| **Symplectic energy conservation** | **0.002%** drift over 200 steps | [`cell_3_results.json`](results/cell_3_results.json) |
| **Gauge covariance** | Wu-Yang residual **< 10⁻⁷** | [`cell_2_results.json`](results/cell_2_results.json) |
| **Full-stack gradcheck** | Passes at float64, 8 layers, exact Jacobians | [`cell_4_results.json`](results/cell_4_results.json) |
| **Gradient-path fix** | `∂L/∂V_net` raised **10⁸×** over base layer | [`cell_1_v7` smoke test](findings/architectural_limit.md) |
| **Newton-Schulz gauge fix** | Orthogonality error **1.2×10⁻⁷** (SVD-free) | [`cell_2_results.json`](results/cell_2_results.json) |

**Why these numbers matter.** The first three are the ones most often
asserted and least often verified in reversible-network papers. The pilot
provides clean numbers for each:

- **O(1) memory** means the 8-layer stack uses the same peak memory as a
  2-layer stack. A naive implementation would use 4× more. The saving is
  exact — peak grows by 0.7%, within measurement noise.
- **Reversibility** means the stack's `invert()` reconstructs the input
  from the output to 8 significant figures. This isn't approximate; it's
  the same accuracy as a direct re-computation.
- **Symplectic energy conservation** means the leapfrog integrator
  preserves total energy over 200 steps. A naive Euler integrator would
  drift by percent-level over the same horizon.

The full verification protocol for each is in [`verification/`](verification/).

---

## The finding

The A-update rule

$$
A^{(l+1)}_v \;=\; A^{(l)}_v \;+\; \Delta t \cdot J_\mu\!\left(M^{(l+1)}_v,\; h_v\right)
$$

is **strictly local in voxel space.** No term in this update allows one
voxel's A-field to influence a neighbor's. The consequence is
unambiguous:

- Tasks that require only **per-voxel A-processing** → **succeed**
- Tasks that require A to **carry information across space** → **fail**

The failure is not an optimization artifact. It's a structural property
of the update rule, and the pilot demonstrates this under three
independent control regimes.

**Full write-up:** [`findings/architectural_limit.md`](findings/architectural_limit.md) (4 pages)
**One-page summary:** [`findings/summary.md`](findings/summary.md)

---

## Results at a glance

Four downstream tasks, each designed to test a different aspect of the
architecture. Every outcome is explained by the same property.

| Task | Decoder reads | Requires transport? | Result |
|---|---|---|---|
| **FDTD field prediction** | A per-voxel | No | ✅ **5.27×** over trivial baseline |
| **Rotation direction** | M only | No (A unused) | ⚪ Neutral, **8.14°** saturated |
| **Kinematic ribbon** | A along a path | Yes | ❌ Partial, **95°** held-out |
| **Active torque** | A at rod bottom | Yes | ❌ Negative, transport = **0.0000** |

The torque task is the definitive one. A random torque is injected at the
top of a straight rod; the decoder reads A at the bottom. If A transports,
the bottom sees a torque-dependent signal. It doesn't:

```
transport_ratio = 0.0000    # bit-identical A8_bottom, all 30 epochs
```

Not "small." Not "below precision." **Exactly zero, under every control.**

---

## Repository contents

```
gwm-pilot/
├── README.md                     This file
├── LICENSE                       Apache 2.0
├── CITATION.cff                  Machine-readable citation
│
├── findings/                     The pilot's conclusions
│   ├── summary.md                1-page overview
│   └── architectural_limit.md    The technical finding (4 pages)
│
├── architecture/                 The architecture specification
│   ├── overview.md               Component-level description
│   ├── formalism.md              Mathematical foundation
│   └── components/               6 files, one per primitive
│
├── verification/                 Reproducible verification protocols
│   ├── cell_1_v7.md              Primitives
│   ├── cell_2_holonomy.md        Gauge covariance, holonomy
│   ├── cell_4_stack.md           Reversibility, memory, gradcheck
│   ├── cell_6_v5_encoder.md      Scene-aware encoder
│   └── block_15_transport.md     The definitive negative
│
├── configs/                      7 YAML files, one per experiment
│
├── results/                      22 JSON verification and task results
│
├── weights/                      Trained models + checkpoints
│   ├── cell_14_ribbon_v7/        Ribbon task (partial result)
│   ├── block_15_v3_frozen/       Torque task (definitive negative)
│   └── checkpoints/              Stage 3B training snapshots
│
└── data/                         3 dataset specifications
```

Every claim in the findings is traceable to a JSON in `results/`. The
exact mapping is in Section 9 of
[`findings/architectural_limit.md`](findings/architectural_limit.md).

---

## Quick start

### Load a trained model

```python
from safetensors.torch import load_file
import json

state = load_file("weights/cell_14_ribbon_v7/model.safetensors")
with open("weights/cell_14_ribbon_v7/config.json") as f:
    config = json.load(f)

print(f"Parameters: {config['n_params']:,}")
print(f"Final probe err_z: {config['final_metrics']['probe_err_z_rad']:.3f} rad")
print(f"Final held-out err_z: {config['final_metrics']['heldout_err_z_rad']:.3f} rad")
```

The `state` dict matches the model's `state_dict()` layout. Hyperparameters
are in the config JSON; the architecture spec is in
[`architecture/`](architecture/).

### Verify the finding

```python
import json
with open("results/block_15_v3_results.json") as f:
    r = json.load(f)

print(f"Transport ratio: {r['training']['transport_ratio'][-1]:.4f}")
print(f"Verdict: {r['verdict']}")
# Output:
#   Transport ratio: 0.0000
#   Verdict: definitive_negative
```

The `block_15_v3_results.json` file contains the full training trace —
30 epochs, all diagnostics, and the verdict. Every number in the finding
is a field in this JSON.

### Verify the primitives

See [`verification/cell_1_v7.md`](verification/cell_1_v7.md) for a
reader-runnable protocol. Each check has expected values and tolerance
thresholds; the whole suite runs in ~60 seconds on CPU.

---

## The architecture, in one paragraph

A point cloud is voxelized to a sparse 3D grid. Each occupied voxel
carries an 8-component Clifford multivector $M$ (density, position, PCA
normal, pseudoscalar), a semantic fiber $h \in \mathbb{R}^{512}$, and an
$\mathfrak{so}(10)$ connection $A$ (one $10 \times 10$ skew-symmetric
matrix per spatial direction). The stack alternates between a symplectic
leapfrog step on $(M, p)$ and a gauge update on $A$. Both steps are
wrapped in custom `autograd.Function`s so the stack is reversible (O(1)
memory in depth) and the integrator is symplectic (energy-conserving).
A decoder reads either $M_8$ or $A_8$ at the output — the choice
determines whether the task exercises the gauge field.

**Full specification:** [`architecture/overview.md`](architecture/overview.md)
**Mathematical foundation:** [`architecture/formalism.md`](architecture/formalism.md)

---

## What's complete

- ✅ All 22 verification and task results (`results/`)
- ✅ Both trained models with full hyperparameter configs (`weights/`)
- ✅ All 4 training checkpoints (`weights/checkpoints/`)
- ✅ Architecture specification (8 files, `architecture/`)
- ✅ Verification protocols (5 files, `verification/`)
- ✅ Experiment configs (7 files, `configs/`)
- ✅ Dataset specifications (3 files, `data/`)
- ✅ Findings: summary + full technical write-up (`findings/`)
- ✅ Apache 2.0 license, machine-readable citation

The pilot release is complete. No further runs, experiments, or
documentation are pending.

---

## What's next

The next version adds spatial coupling to the A-update — a discrete
Laplacian or diffusion term:

$$
A^{(l+1)}_v \;=\; A^{(l)}_v \;+\; \Delta t \cdot \left[
    J_\mu\!\left(M^{(l+1)}_v, h_v\right)
    \;+\; \lambda \sum_{v' \sim v} \left(A^{(l)}_{v'} - A^{(l)}_v\right)
\right]
$$

This is a different architecture. Every primitive in the current
verification suite requires re-derivation, and every downstream task
requires re-running. The v1 release is frozen at the pre-coupling
version so that the finding reported here is reproducible without
ambiguity about which version was being tested.

---

## Citation

If you use this work, please cite:

```bibtex
@software{gwm_pilot_2026,
  title  = {GWM-Pilot v1: A Reversible Yang-Mills Stack for Geometric Deep Learning},
  author = {Devi, Sushain},
  year   = {2026},
  note   = {Pilot release with a specific architectural finding},
  url    = {https://github.com/SushainDevi/GWM-Pilot}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

---

## License

Copyright 2026 Sushain Devi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

## Contact

**Author:** Sushain Devi
**Repository:** https://github.com/SushainDevi/GWM-Pilot

For questions about the finding, see
[`findings/architectural_limit.md`](findings/architectural_limit.md).
For reproducibility issues, open an issue on GitHub.