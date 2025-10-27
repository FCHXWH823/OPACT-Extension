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
import os
from pylon.constraint import constraint
from pylon.brute_force_solver import SatisfactionBruteForceSolver
# from data_loader import *
from fast_data_loader import *
from constraint import Constraint, ConstraintOptimizer
from collections import deque
from writejson import dump_column_allocations
# ----------------------------  basic setup  ----------------------------

# multiplier parameters -------------------------------------------------

# ------------------  STE: column-wise integer projection  --------------
def ste_counts (V: torch.Tensor, p: torch.Tensor, use_hard: bool = False) -> torch.Tensor:
    """
    V : [B, C]          integer bit-counts entering each column
    p : [B, C, 5]       softmax probabilities for 5 compressor types
    returns n_int : [B, C, 5]  integer counts (forward) + STE gradient
    """
    # continuous expectation
    n_soft  = (p * V.unsqueeze(-1)) / IN_BITS          # [B,C,5]
    if not use_hard:
        return n_soft                                     # “soft phase” (no rounding)
    n_floor = torch.floor(n_soft)                      # [B,C,5]

    # bits consumed by exact / approx compressors (k = 0..3)
    used = (n_floor[..., :4] * IN_BITS[:4]).sum(-1)    # [B,C]
    left = (V - used).clamp(min=0)                     # [B,C]

    # fractional residuals guide which real comps get an extra unit
    # frac  = n_soft[..., :4] - n_floor[..., :4]         # [B,C,4]
    # n_add = torch.zeros_like(n_floor[..., :4])         # [B,C,4]

    # for width, k in sorted(zip(IN_BITS[:4].tolist(), range(4))):
    #     add = ((left >= width) & (frac[..., k] > 0)).int()
    #     n_add[..., k] += add
    #     left -= add * width

    n_dummy = left                                     # whatever bits remain

    # assemble integer tensor
    n_int = torch.cat([(n_floor[..., :4]).int(),
                       n_dummy.unsqueeze(-1)           ], dim=-1)  # [B,C,5]

    # straight-through estimator: gradient flows via n_soft
    return n_int + (n_soft - n_soft.detach())
# ---------------------------------------------------------------------------
#  Residual-aware STE  (no “safe column” branch)
# ---------------------------------------------------------------------------
# def ste_counts (V: torch.Tensor, p: torch.Tensor, use_hard: bool = False) -> torch.Tensor:
#     """
#     V : [B, C]          integer bit-counts entering each column
#     p : [B, C, 5]       softmax probabilities for 5 compressor types
#     returns n_int : [B, C, 5]  integer counts (forward) + STE gradient
#     """
#     # continuous expectation
#     n_soft  = (p * V.unsqueeze(-1)) / IN_BITS          # [B,C,5]
#     if not use_hard:
#         return n_soft                                     # “soft phase” (no rounding)
#     n_floor = torch.floor(n_soft)                      # [B,C,5]

#     # bits consumed by exact / approx compressors (k = 0..3)
#     used = (n_floor[..., :4] * IN_BITS[:4]).sum(-1)    # [B,C]
#     left = (V - used).clamp(min=0)                     # [B,C]

#     # fractional residuals guide which real comps get an extra unit
#     frac  = n_soft[..., :4] - n_floor[..., :4]         # [B,C,4]
#     n_add = torch.zeros_like(n_floor[..., :4])         # [B,C,4]

#     for width, k in sorted(zip(IN_BITS[:4].tolist(), range(4))):
#         add = ((left >= width) & (frac[..., k] > 0)).int()
#         n_add[..., k] += add
#         left -= add * width

#     n_dummy = left                                     # whatever bits remain

#     # assemble integer tensor
#     n_int = torch.cat([(n_floor[..., :4] + n_add).int(),
#                        n_dummy.unsqueeze(-1)           ], dim=-1)  # [B,C,5]

#     # straight-through estimator: gradient flows via n_soft
#     return n_int + (n_soft - n_soft.detach())

def ste_counts(V_col: torch.Tensor, p_col: torch.Tensor, use_hard: bool = False) -> torch.Tensor:
    """
    One-column straight-through rounding with “best-fraction” refinement.

    Parameters
    ----------
    V_col : 1-D or 0-D tensor  (B,)  or ()
        Number of *bits* entering this column for each batch element.
    p_col : 1-D tensor (5,)            –  softmax probabilities for
        the 5 compressor types.  Same for every batch element.

    Returns
    -------
    n_int : tensor  (B, 5)
        Integer compressor counts used in the **forward** pass.
        Gradients propagate as if the continuous expectation `n_soft`
        had been used (straight-through estimator).
    """

    # Ensure V_col is at least 1-D so indexing with [:] works everywhere
    if V_col.dim() == 0:
        V_col = V_col.unsqueeze(0)                 # shape → [1]

    # --- 1. continuous expectation ----------------------------------------
    n_soft = (p_col * V_col.unsqueeze(-1)) / IN_BITS  # [B,5]

    # --- 2. integer floor  (guaranteed feasible) --------------------------
    n_floor = torch.floor(n_soft)                     # [B,5]  float
    used    = (n_floor[:, :4] * IN_BITS[:4]).sum(-1)  # [B]
    left    = (V_col - used).clamp(min=0)             # [B]

    # --- 3. fractional deficits  ------------------------------------------
    frac  = n_soft[:, :4] - n_floor[:, :4]            # [B,4]
    n_add = torch.zeros_like(frac)                    # [B,4]
    widths = IN_BITS[:4].tolist()                     # [3,2,3,4]
    width_t = torch.tensor(widths, device=left.device, dtype=left.dtype)

    # Add at most one compressor per iteration, always picking the type that
    # maximises  (fractional_deficit / width)  *and* still fits in 'left'.
    while True:
        fits  = left.unsqueeze(-1) >= width_t               # [B,4]
        score = torch.where(fits, frac * width_t, 0.0)      # [B,4]
        best_k = score.argmax(dim=-1)                       # [B] int idx
        best_w = width_t[best_k]                            # [B]

        # Stop if no candidate actually fits any batch element
        if torch.all(best_w > left):
            break

        mask = (best_w <= left)                             # [B] bool
        if not mask.any():
            break

        n_add[torch.arange(left.size(0)), best_k] += mask.int()
        left -= best_w * mask                               # update residual

        # early exit when nothing but dummy fits
        if left.max() < 2:
            break

    # --- 4. residual 1-bit (or 0) → dummy compressor ----------------------
    n_dummy = left.int()                                    # [B]

    # --- 5. assemble integer tensor + straight-through gradient ----------
    n_int = torch.cat([(n_floor[:, :4] + n_add).int(),
                       n_dummy.unsqueeze(-1)], dim=-1)      # [B,5]
    # gradient flows through n_soft
    return n_int + (n_soft - n_soft.detach())

# ---------------------------------------------------------------------------
#  Greedy ceil-then-floor  —  per-column version
# ---------------------------------------------------------------------------
# def ste_counts(V_col: torch.Tensor, p_col: torch.Tensor) -> torch.Tensor:
#     """
#     V_col : [B]    integer bit-count entering *this* column (batch dim B)
#     p_col : [5]    softmax probs for the 5 compressor types (shared for B)
#     return  : [B,5]  integer counts (forward) with STE gradient = ∂n_soft
#     """
#     # continuous expectation  [B,5]
#     n_soft = (p_col * V_col.unsqueeze(-1)) / IN_BITS

#     # integer tensor to fill
#     n_int  = torch.zeros(V_col.size(0), 5,
#                          device=V_col.device, dtype=torch.int)

#     remaining = V_col.clone()                      # [B]

#     # iterate over the FOUR real compressors
#     for k, width in enumerate(IN_BITS[:4].tolist()):     # (3,2,3,4)
#         desired = torch.ceil(n_soft[:, k]).int()         # [B]
#         max_fit = (remaining // width).int()             # [B]
#         use     = torch.minimum(desired, max_fit)        # [B]
#         n_int[:, k] = use
#         remaining   -= use * width                       # update budget

#     # leftover bits → dummy
#     n_int[:, 4] = remaining                              # [B]

#     # STE: identical gradient to n_soft
#     return n_int.float() + (n_soft - n_soft.detach())


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
def propagate_ste(V, U, p_colwise, bit_weight, use_hard = False):
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
    n_list = [ste_counts(V[:, j], p_colwise[j], use_hard) for j in range(N_COL)]
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
# optim   = torch.optim.Adam([logits], lr=2e-3)
optim   = torch.optim.RMSprop([logits], lr=2e-3)
# last_row_constraint = Constraint(2, 'le', alpha=0.5) # KL >= 5
# ConstraintOptimizer is a normal Optimizer, but step() does gradient ascent instead of descent.
# constraint_opt = ConstraintOptimizer(
#     torch.optim.RMSprop, last_row_constraint.parameters(), 2e-3
# )

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
lambda_err = 0.0001
print(f"λ_err auto-scaled to {lambda_err:.3e}")

# -- geometric ramp for λ_row -----------------------------------------------
window_size = 20
window      = deque(maxlen=window_size)   # stores bools
frozen      = False                       # becomes True once freeze criterion met

window = deque(maxlen=window_size)   # stores bools
warm_steps, ramp_steps = 500, 4_000
λ_lo, λ_hi = 10, 5_000.0
lambda_row = λ_lo  # initial value, will be updated in training loop
def λ_row(step: int):
    if step < warm_steps:
        return λ_lo
    t = min(1.0, (step - warm_steps) / ramp_steps)
    return λ_lo * (λ_hi / λ_lo) ** t

# ---------------------------  training loop  ---------------------------
os.system(f"mkdir -p Training_log_{lambda_err}_ste_counts")
tau = tau_initial
SOFT_STEPS = 0
for step in range(6250):
    use_hard = step >= SOFT_STEPS
    P0, U0 = next(loader)
    V, U   = P0.clone(), U0.clone()

    area_acc = torch.tensor(0.0, device=device)
    err_acc  = torch.tensor(0.0, device=device)

    # temperature annealing
    tau = tau_initial * (tau_final / tau_initial) ** min(1.0, step / anneal_steps)

    for i in range(N_STAGE):
        p_i = torch.softmax(logits[i] / tau, dim=-1)         # [N_COL,5]
        V, U, dA, dE = propagate_ste(V, U, p_i, bit_weight,use_hard)
        area_acc += dA
        err_acc  += dE
        # print(V[0,:])

    # Pylon constraint weight annealing
    # lam_row = lambda_row_initial * (
    #     (lambda_row_final / lambda_row_initial) ** min(1.0, step / anneal_steps)
    # )
    pi = U / (V + 1e-9)                                       # [B, N_COL]

    # loss = (lambda_area * area_acc +
    #         lambda_err  * (err_acc / (2 ** (2*n_bits) - 1)  + (torch.relu(V - 2)*bit_weight).sum())
    #         )

    loss = (lambda_area * area_acc +
            lambda_err  * err_acc +
            lambda_row * (torch.relu(V - 2)*torch.ones_like(bit_weight,device=device,dtype=torch.float)).sum())
    # loss = (lambda_area * area_acc +
    #         lambda_err  * err_acc  +
    #         last_row_constraint(V).sum())

    optim.zero_grad()
    # constraint_opt.zero_grad()
    loss.backward()
    optim.step()
    # constraint_opt.step()
    if step % 10 == 0:
        ok_ratio = last_row_ok(V).float().mean().item()
        print(f"step {step:5d} | loss={loss.item():.3f} "
              f"| area={area_acc.item():.2f} err={err_acc.item():.4f} "
              f"| row-ok={ok_ratio:.3f} λ_row={lambda_row:.2f} constraint={torch.relu(V - 2).sum():.2f} ")
        if not frozen:
            window.append(ok_ratio >= 0.99)  # True if last row ok
            if len(window) == window_size and sum(window) >= 10:
                frozen = True
            else:
                lambda_row = λ_row(step)  # update row constraint weight
    
    if step % 10 == 0:
        with torch.no_grad() and open(f"./Training_log_{lambda_err}_ste_counts/AC_Allocation_{step}.txt", "w", encoding="utf-8") as f:
            # freeze probabilities at very low τ so softmax ≈ one-hot
            p_final = torch.softmax(logits / tau, dim=-1)          # [S,C,5]

            Vs = []
            N_INT = []
            # initialise bit-count vector for a canonical PP cone (1,2,3,…,n,…,1)
            V = torch.tensor([max(min(j + 1, 2 * n_bits - 1 - j, n_bits), 0)
                            for j in range(N_COL)],
                            dtype=torch.float, device=device)   # [1,C]
            Vs.append(V.clone())
            
            names = ["ex-3:2", "ex-2:2", "ap-3:2", "ap-4:2"]        # k = 0..3
            header = "stage  " + " ".join(f"c{j:02d}" for j in range(N_COL))
            print("\n" + header, file=f)

            for s in range(N_STAGE):
                n_cols = [ste_counts(V[j], p_final[s, j],True) for j in range(N_COL)]
                n_int  = torch.stack(n_cols, dim=0).int().squeeze()           # [C,5]
                N_INT.append(n_int.clone())
                # print the 4×C table for this stage
                for k, name in enumerate(names):
                    row = " ".join(f"{n_int[j, k]:5d}" for j in range(N_COL))
                    print(f"{name:<7} {row}", file=f)
                

                # propagate bit-counts for next stage
                sum_bits   = (n_int[:,0]+n_int[:,1]+2*n_int[:,2]+2*n_int[:,3]+n_int[:,4])
                carry_bits = (n_int[:,0]+n_int[:,1])                    # exact comps only
                V = (sum_bits + torch.cat([torch.zeros(1, device=device),
                                        carry_bits[:-1]]))
                Vs.append(V.clone())
                print(V.tolist(), file=f)  # print the V tensor for debugging
                print("-" * len(header), file=f)       # separator between stages
            
            dummy_stage = torch.zeros(N_COL, 5, dtype=torch.int32, device=device)
            dummy_stage[:, 4] = V.to(torch.int32)   # all bits → dummy
            N_INT.append(dummy_stage)                                  # add as final stage

            dump_column_allocations(
                       torch.stack(N_INT,dim=0).int(), torch.stack(Vs,dim=0).int(), f"./Training_log_{lambda_err}_ste_counts/AC_Allocation_{step}.json")
            print(f"step {step:5d} | loss={loss.item():.3f} "
              f"| area={area_acc.item():.2f} err={err_acc.item():.4f} "
              f"| row-ok={ok_ratio:.3f} λ_row={λ_row(step):.2f} constraint={torch.relu(V - 2).sum():.2f} ",file=f)
            
    # ---------------------  CSV output for stage2  -----------------------
    # import csv

    # if step % 100 == 0:
    #     with torch.no_grad(), open(f"./allocate_data/AC_Allocation_{step}.csv", "w", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["stage", "comp_type", "col", "count"])  # header

    #         p_final = torch.softmax(logits / tau, dim=-1)  # [S,C,5]

    #         V = torch.tensor([min(j + 1, 2 * n_bits - 1 - j, n_bits)
    #                         for j in range(N_COL)],
    #                         dtype=torch.float, device=device).unsqueeze(0)   # [1,C]
    #         U = torch.zeros_like(V)

    #         names = ["ex-3:2", "ex-2:2", "ap-3:2", "ap-4:2", "dummy"]

    #         for s in range(N_STAGE):
    #             n_cols = [ste_counts(V[0, j], p_final[s, j]) for j in range(N_COL)]
    #             n_int  = torch.stack(n_cols, dim=0).int()   # [C,5]

    #             # 每个 comp_type、每个 col 一行
    #             for k, name in enumerate(names):
    #                 for j in range(N_COL):
    #                     writer.writerow([s, name, j, int(n_int[j, k])])

    #             # propagate
    #             sum_bits   = n_int.sum(-1)
    #             carry_bits = n_int[:, :2].sum(-1)
    #             V = (sum_bits + torch.cat([torch.zeros(1, device=device),
    #                                     carry_bits[:-1]])).unsqueeze(0)

# ---------------------  integer solution extraction  -------------------
# -----------------------------------------------------------------------
# 9.  dump per-stage, per-column counts of the FOUR real compressors
# -----------------------------------------------------------------------
with torch.no_grad():
    # freeze probabilities at very low τ so softmax ≈ one-hot
    p_final = torch.softmax(logits / tau, dim=-1)          # [S,C,5]

    # initialise bit-count vector for a canonical PP cone (1,2,3,…,n,…,1) [min(j + 1, 2 * n_bits - 1 - j, n_bits) for j in range(n_col)]
    V = torch.tensor([min(j + 1, 2 * n_bits - 1 - j, n_bits)
                      for j in range(N_COL)],
                     dtype=torch.float, device=device).unsqueeze(0)   # [1,C]
    U = torch.zeros_like(V)     # probabilities irrelevant for counting

    names = ["ex-3:2", "ex-2:2", "ap-3:2", "ap-4:2"]        # k = 0..3
    header = "stage  " + " ".join(f"c{j:02d}" for j in range(N_COL))
    print("\n" + header)

    for s in range(N_STAGE):
        n_cols = [ste_counts(V[0, j], p_final[s, j]) for j in range(N_COL)]
        n_int  = torch.stack(n_cols, dim=0).int().squeeze()           # [C,5]

        # print the 4×C table for this stage
        for k, name in enumerate(names):
            row = " ".join(f"{n_int[j, k]:5d}" for j in range(N_COL))
            print(f"{name:<7} {row}")
        print("-" * len(header))       # separator between stages

        # propagate bit-counts for next stage
        sum_bits   = (n_int[:,0]+n_int[:,1]+2*n_int[:,2]+2*n_int[:,3]+n_int[:,4])
        carry_bits = (n_int[:,0]+n_int[:,1])                    # exact comps only
        V = (sum_bits + torch.cat([torch.zeros(1, device=device),
                                   carry_bits[:-1]])).unsqueeze(0)