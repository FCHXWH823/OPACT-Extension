import torch
from parameters import *

# TODO: random seed
def IP_generator(n_bits: int, batch: int, exhaustive: bool = False):
    """
    Parameters:
    - n_bits     : operand width  (⇒ result width = 2*n_bits-1 columns)
    - batch      : patterns per yield
    - exhaustive : if True, enumerate a*b pairs systematically starting from 0;
                 if False, sample operands uniformly at random.

    Returns:
    - a loader that yields:
        {
            P0: shape [N_COL], maximum number of bits in each column
            BITS: shape [B, N_COL, max_col_height], each position is 0 or 1
        }
    """

    n_col = 2 * n_bits - 1                 # should equal N_COL globally
    # assert n_col == N_COL, "N_COL and n_bits inconsistent"
    
    pattern = [min(j + 1, 2 * n_bits - 1 - j, n_bits) for j in range(n_col)]
    P0_const = torch.tensor(pattern, dtype=torch.float, device=device)

    def one_bitmap(a: int, b: int) -> torch.Tensor:
        """
        return shape [N_COL, max_col_height] tensor
        fill each column with AND results of bits from a and b, padding with 0s to max_col_height.
        """
        Ua = [(a >> i) & 1 for i in range(n_bits)]
        Ub = [(b >> j) & 1 for j in range(n_bits)]

        cols = [[] for _ in range(n_col)]
        for i in range(n_bits):
            if Ua[i] == 0: continue
            for j in range(n_bits):
                col = i + j
                bit = Ua[i] & Ub[j]
                cols[col].append(bit)

        # padding 到统一长度
        padded = [
            torch.tensor(col + [0] * (n_bits - len(col)),
                         dtype=torch.float32, device=device)
            for col in cols
        ]
        return torch.stack(padded, dim=0)  # [N_COL, n_bits]

    def loader():
        a = b = 0
        total = 1 << n_bits
        while True:
            batch_bits = []
            for _ in range(batch):
                if exhaustive:
                    bits = one_bitmap(a, b)
                    b += 1
                    if b == total:
                        b = 0
                        a = (a + 1) % total
                else:
                    a_rand = torch.randint(0, total, ()).item()
                    b_rand = torch.randint(0, total, ()).item()
                    # print(f"Operands: a = {a_rand:04b}, b = {b_rand:04b}")
                    bits = one_bitmap(a_rand, b_rand)
                batch_bits.append(bits)
            BITS = torch.stack(batch_bits, dim=0)          # [B, N_COL, n_bits]
            P0 = P0_const.expand(batch, -1)                # [B, N_COL]
            yield P0, BITS

    return loader()

# test the loader
if __name__ == "__main__":
    n_bits = 4
    N_COL = 2 * n_bits - 1  # should equal N_COL globally
    loader = IP_generator(n_bits=n_bits, batch=2)
    P0, BITS = next(loader)

    print("P0 shape:", P0.shape)  # [2, 7]
    print("BITS shape:", BITS.shape)  # [2, 7, 4]

    for i in range(BITS.shape[0]):
        print(f"\nSample #{i}:")
        P0_b = P0[i, :].tolist()
        print(f"P0 = {P0_b}")
        for j in range(BITS.shape[1]):
            bits_in_col = BITS[i, j, :].tolist()
            print(f"  Col {j}: bits = {bits_in_col}")

