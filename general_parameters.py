# ────────────────────────────────────────────────────────────────────────────
#  Compressor library is now loaded from:
#      ./ACs/ac_lib.json      ← created by extract_lib.py
#      ./ACs/pairs.py         ← auto‑generated lambdas
#  This makes parameters.py completely agnostic to how many (or which) ACs
#  you drop into ./ACs.
# ────────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path
from ACs.pairs import F_S_LIST, F_C_LIST, F_ERR_LIST     # auto‑generated
from collections import deque

LIB_JSON = Path("/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json")
assert LIB_JSON.exists(), "Run extract_lib.py first to build ac_lib.json"

# ---------- read metadata ---------------------------------------------------
lib_meta = json.loads(LIB_JSON.read_text())   # dict  {name: {...}}

# preserve deterministic order (same as pairs.py used)
AC_NAMES = sorted(lib_meta)

# ---------- convert to torch tensors ---------------------------------------
import torch

if torch.cuda.is_available():
    device = "cuda"                     # NVIDIA / AMD discrete GPU
elif torch.backends.mps.is_available():
    device = "mps"                      # Apple-silicon GPU
else:
    device = "cpu"                      # fallback

N_COL   = 31     # partial-product width  (2·n_bits-1 for an n×n multiplier)
N_STAGE = 4      # safe upper bound on CSA layers
N_COL += N_STAGE
BATCH   = 1024     # input patterns per SGD step
N_BITS = (N_COL + 1 - N_STAGE) // 2          # derive operand width from N_COL

IN_BITS = torch.tensor(
    [lib_meta[n]["width"]  for n in AC_NAMES] + [1], dtype=torch.float32, device=device
)  # final +1 is dummy passthrough

AREA_W  = torch.tensor(
    [lib_meta[n]["area"]   for n in AC_NAMES] + [0.0], dtype=torch.float32, device=device
)

# ――― optional: analytic MED weight per instance ―――
# We use the probability‑aware polynomial (better) instead of a fixed scalar.
# Keep ERR_W as zeros; propagate_ste will use F_ERR_LIST instead.
ERR_W   = torch.zeros_like(AREA_W)

# ---------- convenience constants ------------------------------------------
K_TYPES        = IN_BITS.numel()         # real compressors + dummy
K_DUMMY_IDX    = K_TYPES - 1
COMP_NAMES     = AC_NAMES + ["dummy"]    # for pretty printing



#  F_S, F_C, F_ERR  are lists of callables that accept a numpy array of p
F_S = F_S_LIST            # length K_TYPES
F_C = F_C_LIST
F_ERR = F_ERR_LIST
# ────────────────────────────────────────────────────────────────────────────

CARRY_TO_NEXT = torch.tensor(
    [("ew" not in name) for name in COMP_NAMES[:-1]],
    dtype=torch.bool,
    device=device
)

anneal_steps = 6000
tau_initial, tau_final = 1.0, 0.1       # softmax temperature schedule

lambda_area = 1.0
lambda_err = 0.0001


warm_steps, ramp_steps = 500, 4_000
λ_lo, λ_hi = 10, 5_0.0
lambda_row = λ_lo  # initial value, will be updated in training loop
def λ_row(step: int):
    if step < warm_steps:
        return λ_lo
    t = min(1.0, (step - warm_steps) / ramp_steps)
    return λ_lo * (λ_hi / λ_lo) ** t


window_size = 20
frozen      = False                       # becomes True once freeze criterion met

window = deque(maxlen=window_size)   # stores bools

def last_row_ok(V_final: torch.Tensor) -> torch.Tensor:
    return (V_final <= 2).all(dim=-1)    # returns [B] boolean