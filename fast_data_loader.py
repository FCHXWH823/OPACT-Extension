import torch
import torch.nn.functional as F
from parameters import *
# ---------------------------------------------------------------------------
# Fast vectorised counts via 1-D convolution
# ---------------------------------------------------------------------------
def one_counts_batch(a_int: torch.Tensor,
                     b_int: torch.Tensor,
                     n_bits: int,
                     n_col_total: int) -> torch.Tensor:
    """
    a_int, b_int : [B]  unsigned integers  (0 … 2**n_bits-1)
    n_bits       : operand width
    n_col_total  : 2*n_bits-1 + N_STAGE    (physical + padding)

    Returns
    -------
    U : [B, n_col_total]  integer '1'-bit counts per weight column
    """
    B      = a_int.size(0)
    device = a_int.device
    cols_phys = 2 * n_bits - 1              # exact length of A×B product

    # --- bit-slice operands ------------------------------------------------
    arange = torch.arange(n_bits, device=device)
    a_bits = ((a_int.unsqueeze(-1) >> arange) & 1).float()    # [B, n_bits]
    b_bits = ((b_int.unsqueeze(-1) >> arange) & 1).float()    # [B, n_bits]

    # --- 1-D convolution to compute diagonal sums -------------------------
    #
    # For each batch element k:
    #   conv_k(t) = Σ_i  a_bits[k,i] * b_bits[k, n_bits-1-(t-i)]
    # which equals Σ_i a_i · b_{t-i}  – exactly the column popcount.
    #
    # We implement this with grouped conv1d so every sample has its own kernel.
    #
    inp   = a_bits.unsqueeze(0)                    # [1, B, n_bits]
    kernel= torch.flip(b_bits, dims=[1]).unsqueeze(1)   # [B, 1, n_bits]
    conv  = F.conv1d(inp, kernel,
                     groups=B, padding=n_bits-1)   # [1, B, 2n-1]
    counts_phys = conv.squeeze(0)  # [B, 2n-1]

    # --- zero-pad for the N_STAGE redundant MSB columns -------------------
    if n_col_total > cols_phys:                            # normally true
        pad = torch.zeros(B, n_col_total - cols_phys,
                          dtype=counts_phys.dtype, device=device)
        counts = torch.cat([counts_phys, pad], dim=1)      # [B, n_col_total]
    else:
        counts = counts_phys

    return counts


# ---------------------------------------------------------------------------
# 2. Loader with vectorised counting
# ---------------------------------------------------------------------------
def make_loader(n_bits: int, batch: int = 64, exhaustive: bool = False):
    """
    n_bits     : operand width  -> physical columns = 2*n_bits-1
    batch      : patterns yielded each iteration
    exhaustive : if True  iterate deterministically over all operands
                 if False random-sample operands each call
    """
    n_col_total = 2 * n_bits - 1 + N_STAGE
    assert n_col_total == N_COL, "N_COL and n_bits inconsistent"

    # constant physical pattern  (then zero-pad)
    pattern_phys = [min(j + 1, 2 * n_bits - 1 - j, n_bits)
                    for j in range(2 * n_bits - 1)]
    pattern = pattern_phys + [0] * N_STAGE
    P0_const = torch.tensor(pattern, dtype=torch.float, device=device)

    # generator -------------------------------------------------------------
    def loader():
        a_idx = b_idx = 0
        total = 1 << n_bits

        while True:
            if exhaustive:
                # build batch as a contiguous slice of the 2D enumeration
                a_batch = torch.arange(a_idx, a_idx + batch, device=device) % total
                b_batch = torch.arange(b_idx, b_idx + batch, device=device) % total
                # advance indices for next call
                b_idx = (b_idx + batch) % total
                if b_idx == 0:
                    a_idx = (a_idx + batch) % total
            else:
                a_batch = torch.randint(0, total, (batch,), device=device)
                b_batch = torch.randint(0, total, (batch,), device=device)

            U0 = one_counts_batch(a_batch, b_batch,
                                  n_bits, n_col_total)        # [B, n_col_total]
            P0 = P0_const.expand(batch, -1)                   # same for all B
            yield P0, U0

    return loader()
