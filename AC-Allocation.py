#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Differentiable Stage-1 optimisation for an approximate compressor tree
---------------------------------------------------------------------

* Straight-Through rounding -> realistic integer behaviour in forward.
* Pylon-compiled constraint ensures each final column has ≤ 2 bits.
* Works on GPU if available.

Replace AREA_W / ERR_W / a32_*/a42_* functions and the minibatch loader
with your technology-accurate data.
"""
import math, torch, pylon
from pylon.constraint import constraint
from pylon.brute_force_solver import SatisfactionBruteForceSolver
from data_loader import *
# ----------------------------  basic setup  ----------------------------

# multiplier parameters -------------------------------------------------

# ------------------  STE: column-wise integer projection  --------------
def ste_counts(V_col: torch.Tensor, p_col: torch.Tensor) -> torch.Tensor:
    """
    V_col : [B]  expected bits arriving in this column
    p_col : [5]  softmax probs for the 5 compressor types (same for all B)

    Returns [B,5] HARD integer counts in forward; gradient ≈ soft expectation.
    """
    n_soft = (p_col * V_col.unsqueeze(-1)) / IN_BITS   # soft expectation

    # greedy integer rounding ------------------------------------------------
    n_int  = torch.floor(n_soft)
    bits_used = (n_int[:, :4] * IN_BITS[:4]).sum(-1, keepdim=True)          # [B,1]
    remain    = (V_col.unsqueeze(-1) - bits_used).clamp(min=0)   # leftover bits
    n_int[..., 4] = remain.squeeze(-1)                           # dump to dummy

    # straight-through glue --------------------------------------------------
    return n_int + (n_soft - n_soft.detach())

# ---------------------------------------------------------------------------
#  Residual-aware STE  (no “safe column” branch)
# ---------------------------------------------------------------------------
def ste_counts (V: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    V : [B, C]          integer bit-counts entering each column
    p : [B, C, 5]       softmax probabilities for 5 compressor types
    returns n_int : [B, C, 5]  integer counts (forward) + STE gradient
    """
    # continuous expectation
    n_soft  = (p * V.unsqueeze(-1)) / IN_BITS          # [B,C,5]
    n_floor = torch.floor(n_soft)                      # [B,C,5]

    # bits consumed by exact / approx compressors (k = 0..3)
    used = (n_floor[..., :4] * IN_BITS[:4]).sum(-1)    # [B,C]
    left = (V - used).clamp(min=0)                     # [B,C]

    # fractional residuals guide which real comps get an extra unit
    frac  = n_soft[..., :4] - n_floor[..., :4]         # [B,C,4]
    n_add = torch.zeros_like(n_floor[..., :4])         # [B,C,4]

    for width, k in sorted(zip(IN_BITS[:4].tolist(), range(4))):
        add = ((left >= width) & (frac[..., k] > 0)).int()
        n_add[..., k] += add
        left -= add * width

    n_dummy = left                                     # whatever bits remain

    # assemble integer tensor
    n_int = torch.cat([(n_floor[..., :4] + n_add).int(),
                       n_dummy.unsqueeze(-1)           ], dim=-1)  # [B,C,5]

    # straight-through estimator: gradient flows via n_soft
    return n_int + (n_soft - n_soft.detach())

# ------------------  one CSA stage  (STE version) ----------------------
# def propagate_ste(V, U, p_colwise, bit_weight):
#     """
#     V  : [B, N_COL]   expected bit counts
#     U  : [B, N_COL]   expected 1-bit counts
#     p_colwise : [N_COL,5]  softmax logits for this stage
#     returns  (V_next, U_next, area_increment, err_increment)
#     """
#     B = V.size(0)
#     # build hard counts column by column (keeps small Python loop)
#     n_cols = [
#         ste_counts(V[:, j], p_colwise[j]) for j in range(N_COL)
#     ]                                   # list of [B,5]
#     n = torch.stack(n_cols, dim=1)      # [B, N_COL, 5]

#     # Bernoulli parameter per column ----------------------------------------
#     pi = U / (V + 1e-9)

#     area_inc = (n * AREA_W).sum()                            # scalar
#     # err_inc  = ((n * ERR_W).sum(-1) * bit_weight).sum()      # scalar
#     # expected MED contribution per column (shape [B, N_COL])
#     med_cols = (
#         n[:, :, 2] * err_a32(pi)           # approx 3:2
#         + n[:, :, 3] * err_a42(pi)           # approx 4:2
#     )

#     # weighted by 2^j significance and summed over batch & columns
#     err_inc = (med_cols * bit_weight).sum()

#     # expected 1-outputs -----------------------------------------------------
#     ones_same = torch.zeros_like(V)
#     ones_next = torch.zeros_like(V)
#     for k in range(5):
#         ones_same += n[:, :, k] * F_S[k](pi)
#         ones_next += n[:, :, k] * F_C[k](pi)

#     # total bit outputs ------------------------------------------------------
#     sum_bits   = n.sum(-1)
#     carry_bits = n[:, :, :4].sum(-1)     # dummy has no carry

#     # shift carries
#     shift = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
#     U_next = ones_same + shift(ones_next)
#     V_next = sum_bits  + shift(carry_bits)

#     return V_next, U_next, area_inc, err_inc

# ---------------------------------------------------------------------------
#  one CSA stage with STE **and** "same-column outputs" for approx comps
# ---------------------------------------------------------------------------
def propagate_ste(V, U, p_colwise, bit_weight):
    """
    V           : [B, N_COL]  expected bit-counts entering this stage
    U           : [B, N_COL]  expected 1-bit counts
    p_colwise   : [N_COL, 5]  softmax probs for 5 comp. types (column-wise)
    bit_weight  : [N_COL]     2**j positional weights for MED

    Returns
        V_next, U_next        (shapes [B, N_COL])
        area_inc, err_inc     (scalars, added to running totals)
    """
    B = V.size(0)

    # ---- STE rounding ------------------------------------------------------
    n_list = [ste_counts(V[:, j], p_colwise[j]) for j in range(N_COL)]
    n      = torch.stack(n_list, dim=1)                       # [B, N_COL, 5]

    # ---- Column-local ‘probability of 1’ ----------------------------------
    pi = U / (V + 1e-9)                                       # [B, N_COL]

    # ---- Area --------------------------------------------------------------
    area_inc = (n * AREA_W).sum()

    # ---- MED (π-dependent formulas) ----------------------------------------
    med_cols = (n[:, :, 2] * err_a32(pi) +           # approx 3:2
                n[:, :, 3] * err_a42(pi))            # approx 4:2
    err_inc  = (med_cols * bit_weight).sum()         # scalar

    # -----------------------------------------------------------------------
    #  Outputs per type
    #     k = 0 → exact 3:2   (sum + carry)
    #     k = 1 → exact 2:2   (sum + carry)
    #     k = 2 → approx 3:2  (sum + "carry", but BOTH stay in column j)
    #     k = 3 → approx 4:2  (sum + "carry", but BOTH stay in column j)
    #     k = 4 → dummy pass  (one bit, same column)
    # -----------------------------------------------------------------------
    ones_same  = torch.zeros_like(V)          # 1-bit expectations in column j
    ones_next  = torch.zeros_like(V)          # 1-bit expectations in column j+1
    sum_bits   = torch.zeros_like(V)          # total bit-count in column j
    carry_bits = torch.zeros_like(V)          # bit-count emitted to j+1

    # exact 3:2 (k=0)
    ones_same  += n[:, :, 0] * F_S[0](pi)
    ones_next  += n[:, :, 0] * F_C[0](pi)
    sum_bits   += n[:, :, 0]
    carry_bits += n[:, :, 0]

    # exact 2:2 (k=1)
    ones_same  += n[:, :, 1] * F_S[1](pi)
    ones_next  += n[:, :, 1] * F_C[1](pi)
    sum_bits   += n[:, :, 1]
    carry_bits += n[:, :, 1]

    # approx 3:2 (k=2)  — BOTH outputs stay in column j
    ones_same  += n[:, :, 2] * (F_S[2](pi) + F_C[2](pi))
    sum_bits   += 2 * n[:, :, 2]              # two outputs, same column

    # approx 4:2 (k=3)  — BOTH outputs stay in column j
    ones_same  += n[:, :, 3] * (F_S[3](pi) + F_C[3](pi))
    sum_bits   += 2 * n[:, :, 3]

    # dummy (k=4)
    ones_same  += n[:, :, 4] * F_S[4](pi)
    sum_bits   += n[:, :, 4]                  # exactly one passed bit

    # ---- shift carries from column j   to column j+1 -----------------------
    shift = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    U_next = ones_same + shift(ones_next)
    V_next = sum_bits  + shift(carry_bits)

    return V_next, U_next, area_inc, err_inc

# -------------  Pylon constraint: “last row ≤ 2 bits per col” ----------
def last_row_ok(V_final: torch.Tensor) -> torch.Tensor:
    return (V_final <= 2).all(dim=-1)    # returns [B] boolean

# last_row_loss = constraint(last_row_ok, FuzzyLogicSolver()).to(device)
last_row_loss = constraint(last_row_ok, SatisfactionBruteForceSolver())


loader  = make_loader(n_bits, batch=BATCH, exhaustive=False)

# -----------------------   training hyper-params   ---------------------
logits  = torch.randn(N_STAGE, N_COL, 5, device=device, requires_grad=True)
optim   = torch.optim.Adam([logits], lr=2e-3)

lambda_area, lambda_err  = 1.0, 1.0
lambda_row_initial, lambda_row_final = 0.1, 20.0
anneal_steps = 6000
tau_initial, tau_final = 1.0, 0.1       # softmax temperature schedule

bit_weight = torch.pow(2, torch.arange(N_COL, device=device, dtype=torch.float))

with torch.no_grad():
    V,U = next(loader)
    area0 = err0 = 0.
    for i in range(N_STAGE):
        p_i = torch.softmax(logits[i], dim=-1)
        V,U,dA,dE = propagate_ste(V,U,p_i,bit_weight)
        area0 += dA
        err0  += dE
lambda_area = 1.0
lambda_err  = (area0 / err0).clamp(min=1e-8).item()
lambda_err = 0.01
print(f"λ_err auto-scaled to {lambda_err:.3e}")

# -- geometric ramp for λ_row -----------------------------------------------
warm_steps, ramp_steps = 500, 4_000
λ_lo, λ_hi = 10, 5_000.0
def λ_row(step: int):
    if step < warm_steps:
        return λ_lo
    t = min(1.0, (step - warm_steps) / ramp_steps)
    return λ_lo * (λ_hi / λ_lo) ** t

# ---------------------------  training loop  ---------------------------
tau = tau_initial
for step in range(1<<20):
    P0, U0 = next(loader)
    V, U   = P0.clone(), U0.clone()

    area_acc = torch.tensor(0.0, device=device)
    err_acc  = torch.tensor(0.0, device=device)

    # temperature annealing
    tau = tau_initial * (tau_final / tau_initial) ** min(1.0, step / anneal_steps)

    for i in range(N_STAGE):
        p_i = torch.softmax(logits[i] / tau, dim=-1)         # [N_COL,5]
        V, U, dA, dE = propagate_ste(V, U, p_i, bit_weight)
        area_acc += dA
        err_acc  += dE

    # Pylon constraint weight annealing
    # lam_row = lambda_row_initial * (
    #     (lambda_row_final / lambda_row_initial) ** min(1.0, step / anneal_steps)
    # )

    loss = (lambda_area * area_acc +
            lambda_err  * err_acc  +
            λ_row(step) * torch.relu(V - 2).pow(2).sum())

    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 10 == 0:
        ok_ratio = last_row_ok(V).float().mean().item()
        print(f"step {step:5d} | loss={loss.item():.3f} "
              f"| area={area_acc.item():.2f} err={err_acc.item():.4f} "
              f"| row-ok={ok_ratio:.3f} λ_row={λ_row(step):.2f} constraint={torch.relu(V - 2).sum():.2f} ")
    
    if step % 1000 == 0 and last_row_ok(V).float().mean().item() > 0.99:
        with torch.no_grad() and open(f"./Training_log/AC_Allocation_{step}.txt", "w", encoding="utf-8") as f:
            # freeze probabilities at very low τ so softmax ≈ one-hot
            p_final = torch.softmax(logits / tau, dim=-1)          # [S,C,5]

            # initialise bit-count vector for a canonical PP cone (1,2,3,…,n,…,1)
            V = torch.tensor([min(j + 1, 2 * n_bits - 1 - j, n_bits)
                            for j in range(N_COL)],
                            dtype=torch.float, device=device).unsqueeze(0)   # [1,C]
            U = torch.zeros_like(V)     # probabilities irrelevant for counting

            names = ["ex-3:2", "ex-2:2", "ap-3:2", "ap-4:2"]        # k = 0..3
            header = "stage  " + " ".join(f"c{j:02d}" for j in range(N_COL))
            print("\n" + header, file=f)

            for s in range(N_STAGE):
                n_cols = [ste_counts(V[0, j], p_final[s, j]) for j in range(N_COL)]
                n_int  = torch.stack(n_cols, dim=0).int()           # [C,5]

                # print the 4×C table for this stage
                for k, name in enumerate(names):
                    row = " ".join(f"{n_int[j, k]:5d}" for j in range(N_COL))
                    print(f"{name:<7} {row}", file=f)
                print("-" * len(header), file=f)       # separator between stages

                # propagate bit-counts for next stage
                sum_bits   = n_int.sum(-1)
                carry_bits = n_int[:, :2].sum(-1)                    # exact comps only
                V = (sum_bits + torch.cat([torch.zeros(1, device=device),
                                        carry_bits[:-1]])).unsqueeze(0)

# ---------------------  integer solution extraction  -------------------
# -----------------------------------------------------------------------
# 9.  dump per-stage, per-column counts of the FOUR real compressors
# -----------------------------------------------------------------------
with torch.no_grad():
    # freeze probabilities at very low τ so softmax ≈ one-hot
    p_final = torch.softmax(logits / tau, dim=-1)          # [S,C,5]

    # initialise bit-count vector for a canonical PP cone (1,2,3,…,n,…,1)
    V = torch.tensor([min(j + 1, 2 * n_bits - 1 - j, n_bits)
                      for j in range(N_COL)],
                     dtype=torch.float, device=device).unsqueeze(0)   # [1,C]
    U = torch.zeros_like(V)     # probabilities irrelevant for counting

    names = ["ex-3:2", "ex-2:2", "ap-3:2", "ap-4:2"]        # k = 0..3
    header = "stage  " + " ".join(f"c{j:02d}" for j in range(N_COL))
    print("\n" + header)

    for s in range(N_STAGE):
        n_cols = [ste_counts(V[0, j], p_final[s, j]) for j in range(N_COL)]
        n_int  = torch.stack(n_cols, dim=0).int()           # [C,5]

        # print the 4×C table for this stage
        for k, name in enumerate(names):
            row = " ".join(f"{n_int[j, k]:5d}" for j in range(N_COL))
            print(f"{name:<7} {row}")
        print("-" * len(header))       # separator between stages

        # propagate bit-counts for next stage
        sum_bits   = n_int.sum(-1)
        carry_bits = n_int[:, :2].sum(-1)                    # exact comps only
        V = (sum_bits + torch.cat([torch.zeros(1, device=device),
                                   carry_bits[:-1]])).unsqueeze(0)