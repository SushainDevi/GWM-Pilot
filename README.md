# GWM-Pilot v1

**A reversible Yang-Mills stack for geometric deep learning.**

This repository contains the artifacts of a pilot study on a new class of
geometric neural architecture. The architecture composes Clifford
multivector fields, a learned so(10) gauge connection, a symplectic
leapfrog integrator, and a reversible autograd stack. The pilot's central
result is a **specific architectural limit** that determines when the
architecture succeeds and when it fails — with an empirical demonstration
under three independent control regimes.

## The finding

The A-update rule

$$
A^{(l+1)}_v \;=\; A^{(l)}_v \;+\; \Delta t \cdot J_\mu\!\left(M^{(l+1)}_v,\; h_v\right)
$$

is **strictly local in voxel space**. No term in this update allows one
voxel's A-field to influence a neighbor's. The consequence is unambiguous:
tasks that require only per-voxel A-processing succeed; tasks that require
A to carry information across space fail — and the failure is not an
optimization artifact but a structural property of the update rule.

**Full write-up:** [`findings/architectural_limit.md`](findings/architectural_limit.md)
**One-page summary:** [`findings/summary.md`](findings/summary.md)

## Results at a glance

| Task | Decoder reads | Requires transport? | Result |
|---|---|---|---|
| FDTD field prediction | A per-voxel | No | ✅ 5.27× over trivial baseline |
| Rotation direction | M only | No (A unused) | ⚪ Neutral, 8.14° saturated |
| Kinematic ribbon | A along a path | Yes | ❌ Partial, 95° held-out |
| Active torque | A at rod bottom | Yes | ❌ Negative, transport = 0.0000 |

Every task's outcome is explained by the same property. The architecture
has a well-defined envelope: local-to-local mappings succeed,
A-independent tasks are neutral, non-local transport fails.

## Repository contents

```
pilot/
├── README.md                     This file
├── findings/                     The pilot's conclusions
│   ├── summary.md                1-page overview
│   └── architectural_limit.md    The technical finding (4 pages)
├── architecture/                 Component specifications (in progress)
│   └── components/
├── verification/                 Verification protocols (in progress)
├── configs/                      Experiment YAMLs (in progress)
├── results/                      22 JSON verification and task results
│   └── figures/
├── weights/                      Trained models + training checkpoints
│   ├── cell_14_ribbon_v7/        Ribbon task (partial result)
│   │   ├── model.safetensors
│   │   └── config.json
│   ├── block_15_v3_frozen/       Torque task (definitive negative)
│   │   ├── model.safetensors
│   │   └── config.json
│   └── checkpoints/              Stage 3B training snapshots (4 files)
└── data/                         Dataset specifications (in progress)
```

Every claim in the findings is backed by a JSON in `results/`. The exact
mapping between claims and artifacts is in Section 9 of
[`findings/architectural_limit.md`](findings/architectural_limit.md).

## Quick start

### Load a trained model

```python
from safetensors.torch import load_file
import json

# Load the ribbon model (Cell 14)
state = load_file("weights/cell_14_ribbon_v7/model.safetensors")
with open("weights/cell_14_ribbon_v7/config.json") as f:
    config = json.load(f)

print(f"Parameters: {config['n_params']:,}")
print(f"Final probe err_z: {config['final_metrics']['probe_err_z_rad']:.3f} rad")
print(f"Final held-out err_z: {config['final_metrics']['heldout_err_z_rad']:.3f} rad")
```

The `state` dict matches the model's `state_dict()` layout. To reconstruct
the model class, see `architecture/components/` (in progress) or the
config JSON for the exact hyperparameters.

### Verify a claim

Every JSON in `results/` is self-describing. To check the transport
finding:

```python
import json
with open("results/block_15_v3_results.json") as f:
    r = json.load(f)

print(f"Transport ratio: {r['training']['transport_ratio'][-1]:.4f}")
print(f"Verdict: {r['verdict']}")
# Expected: 0.0000, definitive_negative
```

## What's complete and what's in progress

**Complete:**
- All primitive verification JSONs (cells 2–4)
- All encoder iteration JSONs (cell 6, v1–v5)
- Both trained models (safetensors) with full hyperparameter configs
- Training checkpoints from stage 3B
- `findings/summary.md` and `findings/architectural_limit.md`

**In progress:**
- `architecture/` — component specifications (8 markdown files)
- `verification/` — verification protocol for each primitive (5 files)
- `configs/` — experiment YAMLs (7 files, derivable from weight metadata)
- `data/` — dataset specifications (3 files)

The pilot artifacts are frozen. The remaining work is documentation.

## The deeper documents

- **[`findings/summary.md`](findings/summary.md)** — one-page overview, for skimming
- **[`findings/architectural_limit.md`](findings/architectural_limit.md)** — the technical finding: formal statement, three-experiment evidence chain, cross-task confirmation, what would fix it, and an honest accounting of what remains untested
- **`architecture/overview.md`** *(coming)* — what the architecture is
- **`architecture/components/`** *(coming)* — one document per primitive
- **`verification/`** *(coming)* — protocol to reproduce each claim

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

## Contact

**Author:** Sushain Devi
**Repository:** https://github.com/SushainDevi/GWM-Pilot

For questions about the finding, see
[`findings/architectural_limit.md`](findings/architectural_limit.md).
For reproducibility issues, open an issue on GitHub.