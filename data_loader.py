import torch
from parameters import *
# ---------------------------------------------------------------------------
# New loader:  constant P0  +  pattern-dependent U0
# ---------------------------------------------------------------------------
def make_loader(n_bits: int, batch: int = 64, exhaustive: bool = False):
    """
    n_bits     : operand width  (⇒ result width = 2*n_bits-1 columns)
    batch      : patterns per yield
    exhaustive : if True, enumerate a×b pairs systematically starting from 0;
                 if False, sample operands uniformly at random.
    """
    n_col = 2 * n_bits - 1                 # should equal N_COL globally
    assert n_col == N_COL, "N_COL and n_bits inconsistent"

    # ------------- constant P0 : [1,2,3,...,N,N-1,...,1] -------------------
    pattern = [min(j + 1, 2 * n_bits - 1 - j, n_bits) for j in range(n_col)]
    P0_const = torch.tensor(pattern, dtype=torch.float, device=device)  # [n_col]

    # ------------- helper to get U0 for ONE operand pair -------------------
    def one_counts(a: int, b: int) -> torch.Tensor:
        """
        Returns [n_col] tensor with the number of 1-bits in each product column
        for operands a, b (both treated as n_bits-bit unsigned integers).
        """
        Ua = torch.tensor([(a >> i) & 1 for i in range(n_bits)], device=device)
        Ub = torch.tensor([(b >> j) & 1 for j in range(n_bits)], device=device)

        U = torch.zeros(n_col, device=device)
        for i in range(n_bits):
            if Ua[i] == 0:
                continue
            # The AND of bit i of a with every bit of b contributes
            # to columns i + j  (0 ≤ j < n_bits).
            cols = torch.arange(i, i + n_bits, device=device)
            U[cols] += Ub        # add 1 where b’s bit is 1
        return U                 # shape [n_col]

    # ------------- generator ------------------------------------------------
    def loader():
        a = b = 0
        total = 1 << n_bits
        while True:
            U_batch = []
            for _ in range(batch):
                if exhaustive:
                    # systematic enumeration
                    U_batch.append(one_counts(a, b))
                    # advance lexicographically
                    b += 1
                    if b == total:
                        b = 0
                        a = (a + 1) % total
                else:
                    # uniform random sampling
                    a_rand = torch.randint(0, total, ()).item()
                    b_rand = torch.randint(0, total, ()).item()
                    U_batch.append(one_counts(a_rand, b_rand))
            U0 = torch.stack(U_batch, dim=0)             # [B, n_col]
            P0 = P0_const.expand(batch, -1)              # repeat SAME P0
            yield P0, U0

    return loader()

# ---------------------------------------------------------------------------
# Replace the old toy_loader with the new one
# ---------------------------------------------------------------------------

# loader  = make_loader(n_bits, batch=BATCH, exhaustive=False)
