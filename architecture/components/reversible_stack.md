# Reversible Autograd Stack

The stack composes $L$ layers and uses a custom `autograd.Function` to
run in reverse during backward, giving O(1) memory in depth. This document
specifies the mechanism, its constraint, and its verified properties.

For the mathematical formulation, see
[`../formalism.md#11-reversibility`](../formalism.md#11-reversibility).

---

## 1. What it is

A container of $L$ `Layer` modules with a reversible forward/backward:

```python
class ReversibleStackFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stack, M, p, A, h):
        ctx.stack = stack
        with torch.no_grad():
            M_o, p_o, A_o, h_o = stack._run_layers(M, p, A, h)
        ctx.save_for_backward(M_o, p_o, A_o, h_o)
        return M_o, p_o, A_o, h_o

    @staticmethod
    def backward(ctx, gM, gp, gA, gh):
        M_o, p_o, A_o, h_o = ctx.saved_tensors
        stack = ctx.stack

        with torch.no_grad():
            M_in, p_in, A_in, h_in = stack._invert_layers(M_o, p_o, A_o, h_o)

        M_r = M_in.detach().requires_grad_(True)
        p_r = p_in.detach().requires_grad_(True)
        A_r = A_in.detach().requires_grad_(True)
        h_r = h_in.detach().requires_grad_(True)

        with torch.enable_grad():
            M_o2, p_o2, A_o2, h_o2 = stack._run_layers(M_r, p_r, A_r, h_r)
            torch.autograd.backward(
                [M_o2, p_o2, A_o2, h_o2],
                [gM, gp, gA, gh],
                retain_graph=False,
            )

        return (None, M_r.grad, p_r.grad, A_r.grad, h_r.grad)
```

## 2. How it works

**Forward.** Runs all $L$ layers under `torch.no_grad()`. Saves only the
final state `(M_o, p_o, A_o, h_o)`. Intermediate states are discarded.

**Backward.**

1. Invert the layers in reverse order to reconstruct the inputs
   `(M_in, p_in, A_in, h_in)`. This uses the exact invertibility of each
   layer (see §3).
2. Re-run the forward under `torch.enable_grad()` starting from the
   reconstructed inputs.
3. Backprop the outer gradients through the re-run graph.
4. Return the input gradients.

**Memory cost.** Forward stores `O(1)` states (just the outputs). Backward
allocates `O(L)` states temporarily during the re-run, but frees them
when the backward completes. Peak memory is `O(1)` in depth.

## 3. Layer invertibility

Each layer is exactly reversible. Given `(M_out, p_out, A_out, h_out)`,
the inverse:

1. Undoes the gauge update: `A_prev = A_out - dt · J_μ(M_out, h_out)`.
2. Undoes the leapfrog: reconstructs `(M_sym, p_half, p_prev)` from
   `(M_out, p_out)` and the potential.
3. Undoes the transport: `M_prev = M_sym + dt · A_prev · ∂M_prev`.
4. Returns `h_prev = h_out` (gauge fix is idempotent, no inverse needed).

See [`symplectic_layer.md`](symplectic_layer.md) and
[`gauge_field.md`](gauge_field.md) for the per-step details.

## 4. The constraint

**The backward is only called when at least one input requires grad.**

`torch.autograd.Function` triggers backward based on the `requires_grad`
flag of the arguments passed to `.apply()`. If all of `M, p, A, h` have
`requires_grad = False`, PyTorch does not invoke the custom backward,
and the layer parameters inside `stack` receive **zero gradient**.

**This constraint is a common source of confusion.** If you encode an
input once with `torch.no_grad()` and reuse the encoded tensor across
a batch, the stack parameters never train. See §4 of
[`../../findings/architectural_limit.md`](../../findings/architectural_limit.md)
for the diagnostic and the two fixes (re-encode per batch, or bypass via
`_run_layers`).

## 5. Usage patterns

**Standard usage:**

```python
stack = GWMStack(P, n_layers=8, dt=0.01)
M8, p8, A8, h8 = stack(M, p, A, h)     # invokes ReversibleStackFn
```

**Bypass for direct backprop:**

```python
M8, p8, A8, h8 = stack._run_layers(M, p, A, h)
```

The bypass runs the layer loop directly. It preserves the graph through
all parameters but stores intermediate states (`O(L)` memory).

**Forward without grad:**

```python
with torch.no_grad():
    M8, p8, A8, h8 = stack.forward_no_grad(M, p, A, h)
```

**Invert the stack:**

```python
M0, p0, A0, h0 = stack.invert(M8, p8, A8, h8)
```

## 6. Verified properties

| Property | Reference | Value |
|---|---|---|
| Reconstruction error (M) at 8 layers | `results/cell_4_results.json` | $1.3 \times 10^{-8}$ |
| Reconstruction error (p) at 8 layers | `results/cell_4_results.json` | $0$ |
| Reconstruction error (A) at 8 layers | `results/cell_4_results.json` | $1.0 \times 10^{-8}$ |
| Peak memory growth (2→8 layers) | `results/cell_4_results.json` | $0.7\%$ |
| Forward/backward memory ratio | `results/cell_4_results.json` | $1.76\times$ |
| Full-stack gradcheck (float64) | `results/cell_4_results.json` | `True` |

**Reproduce.** See [`../../verification/cell_4_stack.md`](../../verification/cell_4_stack.md).

## 7. When to use it

**Use `ReversibleStackFn` when:**
- Training with `M, p, A, h` as live leaves (the standard case).
- Memory is at a premium.
- The stack is deep (≥8 layers).

**Bypass it (`_run_layers`) when:**
- Inputs are detached but stack parameters still need gradient.
- Debugging (easier to inspect the graph).
- The stack is shallow (<4 layers) and memory is not a concern.

## 8. References

| Aspect | Reference |
|---|---|
| Mathematical formulation | [`../formalism.md`](../formalism.md) |
| Per-layer details | [`symplectic_layer.md`](symplectic_layer.md), [`gauge_field.md`](gauge_field.md) |
| Verification protocol | [`../../verification/cell_4_stack.md`](../../verification/cell_4_stack.md) |
| The constraint's consequence | [`../../findings/architectural_limit.md`](../../findings/architectural_limit.md) §3 |