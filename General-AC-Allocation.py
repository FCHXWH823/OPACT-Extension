#!/usr/bin/env python
# ───────────────────────── DEPENDENCIES ────────────────────────────────────
import json, math, time, textwrap
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from fast_data_loader import *
from general_writejson import dump_column_allocations
import numpy as np
import os
import torch.nn.functional as F
# PROJECT CONSTANTS (unchanged parts) --------------------------------------
from general_parameters import (
    N_BITS, N_STAGE, N_COL, BATCH,
    tau_initial, tau_final, warm_steps,
    lambda_area, lambda_err, lambda_row,
    device,
    # NEW dynamic‑library symbols  ↓↓↓

    IN_BITS, AREA_W, F_S, F_C, F_ERR,
    K_TYPES, K_DUMMY_IDX, COMP_NAMES, CARRY_TO_NEXT, anneal_steps,
    last_row_ok, window_size, window, frozen,
    λ_row
)

K_REAL = K_TYPES - 1                          # exclude dummy

# ────────────────────────── DATA LOADER (vectorised) ───────────────────────
# def one_counts_batch(a_int: torch.Tensor, b_int: torch.Tensor) -> torch.Tensor:
#     """Bit‑popcount of every weight column for a minibatch of operands."""
#     B      = a_int.size(0)
#     n_bits = N_BITS
#     cols   = 2 * n_bits - 1

#     arange = torch.arange(n_bits, device=device)
#     a_bits = ((a_int.unsqueeze(-1) >> arange) & 1).float()        # [B, n_bits]
#     b_bits = ((b_int.unsqueeze(-1) >> arange) & 1).float()

#     inp    = a_bits.unsqueeze(0)                                  # [1,B,L]
#     kernel = torch.flip(b_bits, [1]).unsqueeze(1)                 # [B,1,L]
#     conv   = F.conv1d(inp, kernel, groups=B, padding=n_bits - 1)  # [1,B,2n-1]
#     counts = conv.squeeze(0).transpose(0, 1).int()                # [B,2n-1]

#     # pad redundant MSB columns
#     pad = torch.zeros(B, N_COL - cols, dtype=torch.int32, device=device)
#     return torch.cat([counts, pad], dim=1)                        # [B,N_COL]


# def make_loader(batch=BATCH):
#     while True:
#         a = torch.randint(0, 1 << N_BITS, (batch,), device=device)
#         b = torch.randint(0, 1 << N_BITS, (batch,), device=device)
#         U0 = one_counts_batch(a, b).float()                       # [B,N_COL]
#         # canonical pattern P0
#         phys = torch.tensor(
#             [min(j + 1, 2 * N_BITS - 1 - j, N_BITS) for j in range(2*N_BITS-1)],
#             dtype=torch.float32, device=device
#         )
#         P0 = torch.cat([phys, torch.zeros(N_STAGE, device=device)]).expand(batch, -1)
#         yield P0, U0

# ────────────────────────── STRAIGHT‑THROUGH COUNTS ────────────────────────
def ste_counts(
    V_col:    torch.Tensor,       # [B]  current bit‑count in this column
    p_col:    torch.Tensor,       # [K]  softmax probs for K compressors (+dummy)
    use_hard: bool = False,
    score_mode: str = "bits"      # "bits"  or  "count"
) -> torch.Tensor:
    """
    Returns
    -------
    n_int : [B, K_TYPES]  integer counts in the forward pass
            with STE gradients flowing through `n_soft`.
    """

    # ---- ensure 1‑D ------------------------------------------------------
    if V_col.dim() == 0:
        V_col = V_col.unsqueeze(0)            # allow scalar input (B == 1)

    # ---- continuous expectation -----------------------------------------
    n_soft = (p_col * V_col.unsqueeze(-1)) / IN_BITS     # [B, K_TYPES]
    if not use_hard:
        return n_soft

    # ---- integer skeleton ------------------------------------------------
    K_REAL  = K_TYPES - 1                       # last index = dummy
    n_floor = torch.floor(n_soft)               # [B, K_TYPES]
    used    = (n_floor[:, :K_REAL] * IN_BITS[:K_REAL]).sum(-1)   # [B]
    left    = (V_col - used).clamp(min=0)                      # [B]

    n_int = torch.cat([(n_floor[:, :K_REAL]).int(),
                       left.int().unsqueeze(-1)                  # dummy column
                      ], dim=-1)                               # [B, K_TYPES]
    return n_int + (n_soft - n_soft.detach())

    frac    = n_soft[:, :K_REAL] - n_floor[:, :K_REAL]         # [B, K_REAL]
    widths  = IN_BITS[:K_REAL]                                 # [K_REAL]
    n_add   = torch.zeros_like(frac, dtype=torch.int32)        # to be filled



    # ---- greedy loop: recompute score after each allocation --------------
    while True:
        resid = frac - n_add.float()                           # deficit left
        fits  = left.unsqueeze(-1) >= widths                   # [B, K_REAL]
        if not torch.any(fits):
            break

        base_score = resid * widths
        score = torch.where(fits, base_score, torch.zeros_like(base_score))

        best   = score.argmax(dim=-1)                          # [B] best type
        best_w = widths[best]                                  # [B] width(s)

        valid  = best_w <= left                                # can we place it?
        if not torch.any(valid):
            break

        idx = torch.arange(left.size(0), device=device)
        n_add[idx, best] += valid.int()                        # allocate
        left -= best_w * valid

        if (left < widths.min()).all():                        # nothing fits
            break

    # ---- pack result -----------------------------------------------------
    n_dummy = left.int()                                       # leftover bits
    n_int = torch.cat([(n_floor[:, :K_REAL] + n_add).int(),
                       n_dummy.unsqueeze(-1)                  # dummy column
                      ], dim=-1)                               # [B, K_TYPES]

    # ---- straight‑through estimator glue -------------------------------
    return n_int + (n_soft - n_soft.detach())

def ste_counts(V_col, p_col, use_hard=False, score_mode="bits"):
    """
    V_col : [B]   current #bits in this column
    p_col : [K]   softmax probs over compressor library (+ dummy)
    Returns : [B,K] integer forward / soft backward
    """
    if V_col.dim() == 0: V_col = V_col.unsqueeze(0)
    n_soft = (p_col * V_col.unsqueeze(-1)) / IN_BITS               # [B,K]

    if not use_hard:
        return n_soft

    n_floor = torch.floor(n_soft)
    used    = (n_floor[:, :K_REAL] * IN_BITS[:K_REAL]).sum(-1)
    left    = (V_col - used).clamp(min=0)

    frac  = n_soft[:, :K_REAL] - n_floor[:, :K_REAL]
    n_add = torch.zeros_like(frac)
    widths= IN_BITS[:K_REAL]

    score_base = frac * widths
    while True:
        fits  = left.unsqueeze(-1) >= widths
        score = torch.where(fits, score_base, 0.0)
        best  = score.argmax(dim=-1)
        best_w= widths[best]
        if torch.all(best_w > left): break
        mask = (best_w <= left)
        if not mask.any(): break
        n_add[torch.arange(left.size(0), device=device), best] += mask.int()
        left -= best_w * mask
        if left.max() < widths.min(): break

    n_dummy = left.int()
    n_int = torch.cat([(n_floor[:, :K_REAL] + n_add).int(),
                       n_dummy.unsqueeze(-1)], dim=-1)
    return n_int + (n_soft - n_soft.detach())


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
    used    = (n_floor[:, :K_REAL] * IN_BITS[:K_REAL]).sum(-1)  # [B]
    left    = (V_col - used).clamp(min=0)             # [B]

    # --- 3. fractional deficits  ------------------------------------------
    frac  = n_soft[:, :K_REAL] - n_floor[:, :K_REAL]            # [B,4]
    n_add = torch.zeros_like(frac)                    # [B,4]
    widths = IN_BITS[:K_REAL].tolist()                     # [3,2,3,4]
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
    n_int = torch.cat([(n_floor[:, :K_REAL] + n_add).int(),
                       n_dummy.unsqueeze(-1)], dim=-1)      # [B,5]
    # gradient flows through n_soft
    return n_int + (n_soft - n_soft.detach())

# ────────────────────── PROPAGATE ONE CSA STAGE ────────────────────────────
# ---------------------------------------------------------------------------
#  helper:  shift a tensor one column to the left (toward MSB, +1 index)
# ---------------------------------------------------------------------------
shift_left = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

# ---------------------------------------------------------------------------
#  propagate one CSA stage through all columns
# ---------------------------------------------------------------------------

def propagate_stage(V, U, p_colwise, *, use_hard = False):
    """
    Parameters
    ----------
    V, U        : [B, N_COL]   current bit‑count / 1‑count per column
    p_colwise   : [N_COL, K_TYPES]   softmax probabilities for this stage
    use_hard    : bool  –  False in warm‑up (n_soft), True afterwards (STE)

    Returns
    -------
    V_next, U_next   : [B, N_COL]   state for next stage
    area_inc         : scalar       area contributed by this stage
    med_inc          : scalar       MED contributed by this stage
    """
    B = V.size(0)

    # 1. allocate compressors column‑wise (STE)
    n_cols = [ste_counts(V[:, j], p_colwise[j], use_hard=use_hard)
              for j in range(N_COL)]                       # list of [B,K]
    n_int = torch.stack(n_cols, dim=1)                     # [B, N_COL, K_TYPES]

    # 2. derive per‑column 1‑probability  (avoid /0)
    p_i = U / (V + 1e-9)                                   # [B, N_COL]

    # 3. AREA increment   Σ compressors × weight
    area_inc = (n_int * AREA_W).sum()                      # scalar

    # 4. MED increment    Σ compressors × analytic polynomial(p)
    med_cols = torch.zeros_like(V)
    for k in range(K_REAL):
        # if 'EC' not in COMP_NAMES[k]: 
        med_prob = F_ERR[k](p_i)
        med_cols += n_int[:, :, k] * med_prob              # bits × per‑bit MED
    bit_weight = torch.pow(2, torch.arange(N_COL, device=device, dtype=torch.float))
    med_inc = (med_cols * bit_weight).sum(-1)                # scalar
    # med_inc = (med_cols * bit_weight).sum()
    # mse_inc = ((med_cols * bit_weight).sum(-1) ** 2).sum()

    # 5. Expected 1s in *same* vs *next* column
    ones_same = torch.zeros_like(V)
    ones_next = torch.zeros_like(V)

    for k in range(K_REAL):
        s_prob = F_S[k](p_i)   # [B,N_COL]
        c_prob = F_C[k](p_i)

        if CARRY_TO_NEXT[k]:
            ones_same += n_int[:, :, k] * s_prob           # Sum bit stays
            ones_next += n_int[:, :, k] * c_prob           # Carry shifts
        else:
            ones_same += n_int[:, :, k] * (s_prob + c_prob)  # both stay

    # dummy: passes bits unchanged, no carry
    ones_same += n_int[:, :, K_DUMMY_IDX] * F_S[K_DUMMY_IDX](p_i)  # [B,N_COL]

    # 6. Bit‑count propagation ------------------------------------------------
    #    (how many *physical* bits feed the next stage)
    sum_bits   = torch.zeros_like(V)
    carry_bits = torch.zeros_like(V)

    for k in range(K_REAL):
        if CARRY_TO_NEXT[k]:
            sum_bits   += n_int[:, :, k]          # 1 bit stays
            carry_bits += n_int[:, :, k]          # 1 bit shifts
        else:
            sum_bits   += 2 * n_int[:, :, k]      # both stay

    sum_bits += n_int[:, :, K_DUMMY_IDX]          # dummy passes 1 bit

    V_next = sum_bits + shift_left(carry_bits)    # bits for next stage
    U_next = ones_same + shift_left(ones_next)    # 1‑counts for next stage

    return V_next, U_next, area_inc, med_inc

# def propagate_stage(V, U, p_colwise, use_hard = False):
#     """
#     V           : [B, N_COL]  expected bit-counts entering this stage
#     U           : [B, N_COL]  expected 1-bit counts
#     p_colwise   : [N_COL, 5]  softmax probs for 5 comp. types (column-wise)
#     bit_weight  : [N_COL]     2**j positional weights for MED

#     Returns
#         V_next, U_next        (shapes [B, N_COL])
#         area_inc, err_inc     (scalars, added to running totals)
#     """
#     B = V.size(0)

#     # ---- STE rounding ------------------------------------------------------
#     n_list = [ste_counts(V[:, j], p_colwise[j], use_hard) for j in range(N_COL)]
#     n      = torch.stack(n_list, dim=1)                       # [B, N_COL, 5]

#     # ---- Column-local ‘probability of 1’ ----------------------------------
#     pi = U / (V + 1e-9)                                       # [B, N_COL]

#     # ---- Area --------------------------------------------------------------
#     area_inc = (n * AREA_W).sum()

#     # ---- MED (π-dependent formulas) ----------------------------------------
#     bit_weight = torch.pow(2, torch.arange(N_COL, device=device, dtype=torch.float))
#     med_cols = (n[:, :, 0] * F_ERR[0](pi) +           # approx 3:2
#                 n[:, :, 1] * F_ERR[1](pi))            # approx 4:2
#     err_inc  = (med_cols * bit_weight).sum()         # scalar

#     # -----------------------------------------------------------------------
#     #  Outputs per type
#     #     k = 0 → exact 3:2   (sum + carry)
#     #     k = 1 → exact 2:2   (sum + carry)
#     #     k = 2 → approx 3:2  (sum + "carry", but BOTH stay in column j)
#     #     k = 3 → approx 4:2  (sum + "carry", but BOTH stay in column j)
#     #     k = 4 → dummy pass  (one bit, same column)
#     # -----------------------------------------------------------------------
#     ones_same  = torch.zeros_like(V)          # 1-bit expectations in column j
#     ones_next  = torch.zeros_like(V)          # 1-bit expectations in column j+1
#     sum_bits   = torch.zeros_like(V)          # total bit-count in column j
#     carry_bits = torch.zeros_like(V)          # bit-count emitted to j+1

#     # exact 3:2 (k=0)
#     ones_same  += n[:, :, 3] * F_S[3](pi)
#     ones_next  += n[:, :, 3] * F_C[3](pi)
#     sum_bits   += n[:, :, 3]
#     carry_bits += n[:, :, 3]

#     # exact 2:2 (k=1)
#     ones_same  += n[:, :, 2] * F_S[2](pi)
#     ones_next  += n[:, :, 2] * F_C[2](pi)
#     sum_bits   += n[:, :, 2]
#     carry_bits += n[:, :, 2]

#     # approx 3:2 (k=2)  — BOTH outputs stay in column j
#     ones_same  += n[:, :, 0] * (F_S[0](pi) + F_C[0](pi))
#     sum_bits   += 2 * n[:, :, 0]              # two outputs, same column

#     # approx 4:2 (k=3)  — BOTH outputs stay in column j
#     ones_same  += n[:, :, 1] * (F_S[1](pi) + F_C[1](pi))
#     sum_bits   += 2 * n[:, :, 1]

#     # dummy (k=4)
#     ones_same  += n[:, :, 4] * F_S[4](pi)
#     sum_bits   += n[:, :, 4]                  # exactly one passed bit

#     # ---- shift carries from column j   to column j+1 -----------------------
#     shift = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
#     U_next = ones_same + shift(ones_next)
#     V_next = sum_bits  + shift(carry_bits)

#     return V_next, U_next, area_inc, err_inc


# ───────────────────────────── TRAINING LOOP ───────────────────────────────
loader  = make_loader(N_BITS, batch=BATCH, exhaustive=False)
logits  = torch.randn(N_STAGE, N_COL, K_TYPES, device=device, requires_grad=True)
# logits = torch.zeros(N_STAGE, N_COL, K_TYPES, device=device, requires_grad=True)
# with torch.no_grad():
#     logits[:, :, K_DUMMY_IDX] += 1.0
# logits = torch.empty(N_STAGE, N_COL, K_TYPES, device=device, requires_grad=True).uniform_(-0.02, 0.02)
# bias = torch.full((K_TYPES,), -10.0, device=device)
# bias[-2] = (0.0)            # index of exact_3:2
# bias[-3] = 0.0            # index of exact_2:2
# logits += bias           # broadcast over stage & column
opt    = torch.optim.RMSprop([logits], lr=2e-2)
# opt = torch.optim.Adam([logits], lr=2e-3)  # Adam optimizer

def temperature(step):
    t = min(1.0, step / anneal_steps)
    return tau_initial * (tau_final / tau_initial) ** t


os.system(f"mkdir -p Training_log_{lambda_err}_ste_counts_general_mae_RMSprop")

with torch.no_grad():
    V,U = next(loader)
    area0 = err0 = 0.
    for i in range(N_STAGE):
        p_i = torch.softmax(logits[i], dim=-1)
        V,U,dA,dE = propagate_stage(V,U,p_i)
        area0 += dA
        err0  += dE

for step in range(6250):
    P0, U0 = next(loader)                         # [B,N_COL]
    V, U   = P0.clone(), U0.clone()
    area_acc = torch.tensor(0.0, device=device)
    med_acc  = torch.tensor([0.0] * BATCH, device=device)
    # med_acc = torch.tensor(0.0, device=device)

    tau = temperature(step)
    use_hard = step >= 0

    for s in range(N_STAGE):
        p = torch.softmax(logits[s] / tau, dim=-1)    # [N_COL,K]
        V, U, dA, dMed = propagate_stage(V, U, p, use_hard=use_hard)
        area_acc += dA
        med_acc  += dMed

    # mse_acc = (med_acc ** 2).sum()
    mse_acc = med_acc.abs().sum()  # use absolute MED for loss
    # huber_loss = F.huber_loss(med_acc, torch.zeros_like(med_acc), delta=1.0)

    bit_weight = torch.pow(2, torch.arange(N_COL, device=device, dtype=torch.float))
    # loss = lambda_err * mse_acc
    # loss = lambda_area * torch.relu(area_acc-750000) + lambda_err * mse_acc
    loss = lambda_area * area_acc + lambda_err * mse_acc + lambda_row * (torch.relu(V - 2)*torch.ones_like(bit_weight,device=device,dtype=torch.float)).sum()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        ok_ratio = last_row_ok(V).float().mean().item()
        print(f"step {step:5d} | loss={loss.item():.3f} "
              f"| area={area_acc.item():.2f} err={med_acc.abs().sum().item():.4f} "
              f"| row-ok={ok_ratio:.3f} λ_row={lambda_row:.2f} constraint={torch.relu(V - 2).sum():.2f} ")
        if not frozen:
            window.append(ok_ratio >= 0.99)  # True if last row ok
            if len(window) == window_size and sum(window) >= 10:
                frozen = True
            else:
                lambda_row = λ_row(step)  # update row constraint weight
    
    if step % 10 == 0:
        # ─────────────────────────────  LOG / DUMP  ──────────────────────────────
        with torch.no_grad(), open(f"./Training_log_{lambda_err}_ste_counts_general/"
                                f"AC_Allocation_{step}.txt",
                                "w", encoding="utf-8") as f:

            # freeze probabilities – tau already tiny at this point
            p_final = torch.softmax(logits / tau, dim=-1)      # [N_STAGE, N_COL, K_TYPES]

            Vs, N_INT = [], []
            # canonical PP cone   (1,2,3,…,n,…,1)  + padding columns
            V = torch.tensor([max(min(j + 1, 2 * N_BITS - 1 - j, N_BITS), 0)
                            for j in range(N_COL)],
                            dtype=torch.float, device=device)          # [N_COL]
            Vs.append(V.clone())

            names  = COMP_NAMES[:-1]        # exclude dummy for table
            header = "stage  " + " ".join(f"c{j:02d}" for j in range(N_COL))
            print("\n" + header, file=f)

            for s in range(N_STAGE):
                # integer counts for every column (use_hard=True)
                n_cols = [ste_counts(V[j], p_final[s, j], use_hard=True)
                        for j in range(N_COL)]
                n_int = torch.stack(n_cols, dim=0).int().squeeze()       # [N_COL, K_TYPES]
                N_INT.append(n_int.clone())

                # print K_REAL × C table
                for k, name in enumerate(names):
                    row = " ".join(f"{int(n_int[j, k]):5d}" for j in range(N_COL))
                    print(f"{name:<10} {row}", file=f)

                # ---------- propagate bit‑counts for next stage -------------------
                sum_bits   = torch.zeros(N_COL, device=device)
                carry_bits = torch.zeros(N_COL, device=device)

                for k in range(K_REAL):
                    if CARRY_TO_NEXT[k]:
                        sum_bits   += n_int[:, k]              # 1 stays
                        carry_bits += n_int[:, k]              # 1 shifts
                    else:
                        sum_bits   += 2 * n_int[:, k]          # both stay

                # dummy passes its one bit
                sum_bits += n_int[:, K_DUMMY_IDX]

                V = sum_bits + torch.cat([torch.zeros(1, device=device), carry_bits[:-1]])
                Vs.append(V.clone())

                print("V_next:", V.tolist(), file=f)
                print("-" * len(header), file=f)

            # ---------- append dummy stage ---------------------------------------
            dummy_stage = torch.zeros(N_COL, K_TYPES, dtype=torch.int32, device=device)
            dummy_stage[:, K_DUMMY_IDX] = V.to(torch.int32)    # all remaining bits
            N_INT.append(dummy_stage)

            dump_column_allocations(
                torch.stack(N_INT, dim=0).int(),
                torch.stack(Vs,  dim=0).int(),
                f"./Training_log_{lambda_err}_ste_counts_general/AC_Allocation_{step}.json"
            )

            ok_ratio = float((V <= 2).float().mean())
            print(
                (f"step {step:5d} | loss={loss.item():.3f} "
                f"| area={area_acc.item():.2f}  med={med_acc.abs().sum().item():.4f} "
                f"| row‑ok={ok_ratio:.3f}  λ_row={λ_row(step):.2f} "
                f"constraint={(torch.relu(V-2)).sum().item():.0f}"),
                file=f
            )
