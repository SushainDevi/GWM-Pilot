# GWM-Pilot v1 — Summary

**We built a reversible Yang-Mills stack** — a geometric deep learning
architecture where information flows through a learned non-abelian gauge
field. The stack composes (i) Clifford multivector fields over a sparse
voxel grid, (ii) an so(10) connection updated per voxel, (iii) a symplectic
leapfrog integrator, and (iv) a reversible autograd stack with O(1) memory
in depth. Every primitive is verified to machine precision across ~60
checks (see `results/`).

**The pilot's central finding is a specific architectural limit.** The
A-update rule

    A_next = A + dt · J_μ(M)

is strictly local in voxel space. No term in this update allows a voxel's
A-field to influence a neighbor's. Information cannot propagate across
spatial distance through A. The consequence is unambiguous: tasks requiring
only local A-processing succeed; tasks requiring A to *transport* information
across space fail — and the failure is not an optimization artifact but a
structural property of the update rule.

## Results

| Task | Decoder reads | Requires transport? | Result |
|---|---|---|---|
| FDTD field prediction | A per-voxel | No | ✅ Positive — 5.27× over trivial |
| Rotation direction | M only | No (A unused) | ⚪ Neutral — 8.14° saturated |
| Kinematic ribbon | A along a path | Yes | ❌ Partial — 95° held-out |
| Active torque | A at rod bottom | Yes | ❌ Negative — transport ratio = 0.0000 |

Every task's outcome is explained by the same property. There is no
second failure mode.

## Evidence for the finding

The torque task provides the definitive test. A random torque is injected
into the momentum at the top of a straight rod; the decoder reads A at the
bottom. If A transports, the bottom sees a torque-dependent signal. It
doesn't:

- **v1 (naive):** `∂L/∂W_proj = 0.0` — the reversible stack did not
  backprop to the layer parameters.
- **v2 (patched):** `∂L/∂W_proj = 9.82`, `∂L/∂encoder = 1.78e+02`,
  spatial derivative `d_M` computed and passed. Gradients healthy.
  **`transport_ratio = 0.0000`** across all 30 epochs.
- **v3 (frozen decoder):** decoder frozen at init, so training loss can
  only drop by changing `A8_bottom`. Loss returns to the null baseline
  (3.79 vs 3.80 null). **`transport_ratio = 0.0000`.**

Bit-identical `A8_bottom` under independently sampled top torques. Not
"small." Not "below precision." Exactly zero, under every control.

## Why it matters

The architecture *can* learn. FDTD reached 5.27× improvement over the
trivial baseline; the encoder iterations reduced probe error on the ribbon
task from the ~91° null to 55°; the Wilson line is gauge-covariant to 1e-7.
What it cannot do is route information through A across space.

**This gives the architecture a specific envelope:** local-to-local
mappings succeed, A-independent tasks are neutral, and non-local transport
fails. The envelope is more informative than a "the architecture works"
claim, because it tells you exactly when to reach for it and when not to.

## What's next

The next version adds spatial coupling to the A-update — a discrete
Laplacian or diffusion term:

    A_next = A + dt · (J_μ(M) + λ · ∇²A)

This is a one-line change to the layer, and it is a different architecture.
The current verification suite (reversibility, gradcheck, gauge covariance)
does not automatically extend to the coupled update. The v1 release is
frozen at the pre-coupling version so the finding is reproducible. The
next version will re-verify every primitive from scratch and re-run every
downstream task.

## How to verify

Every claim in this document is backed by a JSON in `results/`:

- **Primitives** — `results/cell_2_results.json`, `cell_3_results.json`,
  `cell_4_results.json`
- **Encoder iterations** — `results/cell_6_results.json` through
  `cell_6_v5_results.json`
- **FDTD (positive)** — `results/block_9_v2_summary.json`
- **Rotation (neutral)** — `results/cell_13_results.json`,
  `cell_13_v3_sweep.json`
- **Ribbon (partial)** — `results/cell_14_results.json`,
  `cell_14_iterations.json`
- **Torque (definitive)** — `results/block_15_results.json`,
  `block_15_v2_results.json`, `block_15_v3_results.json`

Trained models are in `weights/cell_14_ribbon_v7/` and
`weights/block_15_v3_frozen/`, both in `safetensors` format. Load with
`safetensors.torch.load_file`. Each model directory contains a
`config.json` with the exact hyperparameters used.

The verification protocol for each primitive is documented in
`verification/` (in progress). Architectural details are in
`architecture/` (in progress). The full finding is developed in
`findings/architectural_limit.md`.