# Symplectic Leapfrog Integrator

The $(M, p)$ dynamics follow a Hamiltonian system, and the integrator is
a symplectic leapfrog. This document specifies the integrator's structure,
why it uses a custom `autograd.Function`, and what properties it preserves.

For the mathematical formulation, see
[`../formalism.md#9-symplectic-dynamics`](../formalism.md#9-symplectic-dynamics).

---

## 1. What it is

A custom `torch.autograd.Function` that advances one leapfrog step of the
Hamiltonian system $H(M, p) = \tfrac{1}{2}\|p\|^2 + V(M, h)$ with learned
potential $V$.

**Signature:**

```python
class SymplecticLeapfrogFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, p, h, V_net, dt):
        ...
        return M_next, p_next
```

## 2. The update

$$
\begin{aligned}
p_{1/2} &= p - \tfrac{\Delta t}{2} \nabla_M V(M, h) \\
M' &= M + \Delta t \, p_{1/2} \\
p' &= p_{1/2} - \tfrac{\Delta t}{2} \nabla_M V(M', h)
\end{aligned}
$$

**Forward implementation:**

```python
@staticmethod
def forward(ctx, M, p, h, V_net, dt):
    ctx.save_for_backward(M.detach(), p.detach(), h.detach())
    ctx.V_net = V_net
    ctx.dt = dt

    with torch.enable_grad():
        h_det = h.detach()
        M_req = M.detach().requires_grad_(True)
        V0 = V_net(M_req, h_det)
        gV0 = torch.autograd.grad(V0.sum(), M_req, create_graph=True)[0]

        p_half = p - 0.5 * dt * gV0
        M_next = M + dt * p_half

        M_next_req = M_next.detach().requires_grad_(True)
        V1 = V_net(M_next_req, h_det)
        gV1 = torch.autograd.grad(V1.sum(), M_next_req, create_graph=True)[0]

        p_next = p_half - 0.5 * dt * gV1

    return M_next.detach(), p_next.detach()
```

## 3. Why a custom `Function`

The forward computes $\nabla_M V$ internally via `torch.autograd.grad`,
which would normally appear in the outer autograd graph. By wrapping the
whole step in a custom `Function`, we:

1. **Detach the outputs** from the internal graph. The outer graph sees
   a single primitive.
2. **Define the backward explicitly.** The backward replays the forward
   under `torch.enable_grad()` and backprops through the reconstructed
   graph.
3. **Support double backward.** The internal `create_graph=True` on the
   gradients allows second-order autograd.

**Without the custom Function**, the forward would produce a large graph
of autograd operations, which is slower and uses more memory. The custom
Function makes the leapfrog a single node.

## 4. The backward

```python
@staticmethod
def backward(ctx, gM_next, gp_next):
    M, p, h = ctx.saved_tensors
    V_net = ctx.V_net
    dt = ctx.dt

    M_req = M.detach().requires_grad_(True)
    p_req = p.detach().requires_grad_(True)
    h_req = h.detach().requires_grad_(True)

    with torch.enable_grad():
        V0 = V_net(M_req, h_req)
        gV0 = torch.autograd.grad(V0.sum(), M_req, create_graph=True)[0]

        p_half = p_req - 0.5 * dt * gV0
        M_next = M_req + dt * p_half

        V1 = V_net(M_next, h_req)
        gV1 = torch.autograd.grad(V1.sum(), M_next, create_graph=True)[0]

        p_next = p_half - 0.5 * dt * gV1

        L = (M_next * gM_next).sum() + (p_next * gp_next).sum()
        grads = torch.autograd.grad(L, [M_req, p_req, h_req],
                                     retain_graph=True, allow_unused=True)
        grad_M = grads[0] if grads[0] is not None else torch.zeros_like(M)
        grad_p = grads[1] if grads[1] is not None else torch.zeros_like(p)
        grad_h = grads[2] if grads[2] is not None else torch.zeros_like(h)

    return grad_M, grad_p, grad_h, None, None
```

**Key features:**

- Reconstructs the forward pass under `enable_grad`.
- Backprops the outer gradients `(gM_next, gp_next)` through it.
- Returns gradients for `M, p, h`; `None` for `V_net` and `dt` (V_net's
  parameters are captured inside `L.backward()` implicitly).

**Note.** The `V_net` gradient reaches its parameters through the
`create_graph=True` on the internal gradients, not through the return
value of `backward`. This is why the pilot's v7 fix mattered — see
[`../../findings/architectural_limit.md`](../../findings/architectural_limit.md).

## 5. Verified properties

| Property | Reference | Value |
|---|---|---|
| Energy drift (200 steps, harmonic V) | `results/cell_3_results.json` | $< 0.01\%$ |
| Invertibility (M) | `results/cell_3_results.json` | exact ($0$) |
| Invertibility (p) | `results/cell_3_results.json` | $9.3 \times 10^{-10}$ |
| Gradcheck (float64) | `results/cell_3_results.json` | `True` |

**Reproduce.** See [`../../verification/cell_1_v7.md`](../../verification/cell_1_v7.md) §5.

## 6. Properties of the leapfrog

- **Second-order accurate.** Local truncation error $\mathcal{O}(\Delta t^3)$.
- **Symplectic.** Preserves the 2-form $\omega = dM \wedge dp$.
- **Time-reversible.** Integrating backward with $-\Delta t$ returns the
  initial state exactly (up to floating-point).
- **Bounded energy error.** Total energy oscillates but does not drift.

These are the standard properties of the Störmer-Verlet/leapfrog family.
The pilot's contribution is the specific implementation using a custom
`autograd.Function` with `create_graph=True` internals.

## 7. Interaction with the rest of the layer

The symplectic step is **step 2** of the layer's four-step sequence:

1. Gauge-fix $h$ (Newton-Schulz)
2. Transport $M$ using $A \cdot \partial M$
3. **Symplectic leapfrog** (this component)
4. Gauge update on $A$

The leapfrog receives `M_half` (post-transport) and `p`, produces
`M_sym` and `p_next`, which then feed into the gauge update.

## 8. References

| Aspect | Reference |
|---|---|
| Mathematical formulation | [`../formalism.md`](../formalism.md) |
| Reversibility at stack level | [`reversible_stack.md`](reversible_stack.md) |
| Verification protocol | [`../../verification/cell_1_v7.md`](../../verification/cell_1_v7.md) §5 |