import torch

if torch.cuda.is_available():
    device = "cuda"                     # NVIDIA / AMD discrete GPU
elif torch.backends.mps.is_available():
    device = "mps"                      # Apple-silicon GPU
else:
    device = "cpu"                      # fallback

torch.manual_seed(0)

N_COL   = 31     # partial-product width  (2·n_bits-1 for an n×n multiplier)
N_STAGE = 4      # safe upper bound on CSA layers
BATCH   = 64     # input patterns per SGD step

# compressor meta-data --------------------------------------------------
#  k :     0            1            2              3               4
# type : exact-3:2 | exact-2:2 | approx-3:2 | approx-4:2 | dummy-pass
IN_BITS = torch.tensor([3, 2, 3, 4, 1], device=device, dtype=torch.float)

# *** placeholders — replace with tech numbers ***
AREA_W  = torch.tensor([5.05, 2.66, 2.66, 4.78, 0.0], device=device)   # cell area
ERR_W   = torch.tensor([0.0, 0.0, 1/64, 7/128, 0.0], device=device) # MED weight

# ------------- Bernoulli output polynomials  (truth-table derived) -----
def ha_S(pi): return 2 * pi * (1 - pi)
def ha_C(pi): return pi ** 2
def fa_S(pi): return 3 * pi * (1 - pi) ** 2 + pi ** 3
def fa_C(pi): return 3 * pi ** 2 * (1 - pi) + pi ** 3

# Example approximate 3:2  (swap in your real cell)
def a32_S(pi): return 2 * pi * (1 - pi) ** 2 + pi ** 3
def a32_C(pi): return 3 * pi ** 2 * (1 - pi) + pi ** 3

# Example approximate 4:2  (swap in your real cell)
def a42_S(pi):
    return (
          4 * pi * (1 - pi) ** 3
        + 6 * pi ** 2 * (1 - pi) ** 2
        + 4 * pi ** 3 * (1 - pi)
        + pi ** 4
    )

def a42_C(pi):
    return (
          6 * pi ** 2 * (1 - pi) ** 2
        + 4 * pi ** 3 * (1 - pi)
        + pi ** 4
    )

F_S = (fa_S, ha_S, a32_S, a42_S, lambda pi: pi)          # k-order list
F_C = (fa_C, ha_C, a32_C, a42_C, lambda pi: pi * 0)       # dummy no carry

n_bits = (N_COL + 1) // 2          # derive operand width from N_COL

# ---------------------------------------------------------------------------
#  Error (MED) of ONE approximate compressor under input probability π
#  derived from the screenshot formulas
# ---------------------------------------------------------------------------
def err_a32(pi):
    # MED3:2 = p0·p1·p2   (all inputs share same Bernoulli π ⇒ π³)
    return pi ** 3

def err_a42(pi):
    # MED4:2 = p0p1p2 + p1p2p3 + 2·p0p1p3·(1-p2)
    # With i.i.d. inputs (prob=π)  ⇒ 4·π³ − 2·π⁴
    return 4 * pi ** 3 - 2 * pi ** 4
