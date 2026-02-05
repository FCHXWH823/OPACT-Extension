import torch, random
from parameters import *
from tqdm import trange

torch.manual_seed(5029)

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
            BITS: shape [B, N_COL, max_col_height], each position is 0 or 1
        }
    """

    n_col = 2 * n_bits - 1                 # should equal N_COL globally
    # assert n_col == N_COL, "N_COL and n_bits inconsistent"
    
    pattern = [min(j + 1, 2 * n_bits - 1 - j, n_bits) for j in range(n_col)]
    # P0_const = torch.tensor(pattern, dtype=torch.float, device=device)

    def one_bitmap(a: int, b: int) -> torch.Tensor:
        """
        return shape [N_COL, max_col_height] tensor
        fill each column with AND results of bits from a and b, padding with 0s to max_col_height.
        """
        Ua = [(a >> i) & 1 for i in range(n_bits)]
        Ub = [(b >> j) & 1 for j in range(n_bits)]

        cols = [[] for _ in range(n_col)]
        for i in range(n_bits):
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
        total = 1 << n_bits
        mask = total - 1
        space = 1 << (2 * n_bits)

        t = 0
        stride = 0x9e3779b97f4a7c15  # 黄金比例常数，打散效果好
        a = 1
        b = -1

        while True:
            batch_bits = []
            # a = 1
            # b = -1
            if exhaustive:
                for _ in trange(batch, desc="Loading Test Data"):

                    # hash 打散
                    x = (t * stride) & (space - 1)
                    a = (x >> n_bits) & mask
                    b = x & mask
                    t += 1
                    
                    # b+=1
                    # if b == total:
                    #     b = 1
                    #     a+=1
                    # print(f"Operands: a = {a:0{n_bits}b} ({a}), b = {b:0{n_bits}b} ({b})")
                    bits = one_bitmap(a, b)
                    batch_bits.append(bits)
            else:
                for _ in range(batch):
                    a = torch.randint(0, total, ()).item()
                    b = torch.randint(0, total, ()).item()
                    # b+=1
                    # if b == total:
                    #     b = 1
                    #     a+=1
                    # print(f"Operands: a = {a:0{n_bits}b} ({a}), b = {b:0{n_bits}b} ({b})")
                    bits = one_bitmap(a, b)
                    batch_bits.append(bits)
            BITS = torch.stack(batch_bits, dim=0)          # [B, N_COL, n_bits]
            # P0 = P0_const.expand(batch, -1)                # [B, N_COL]
            yield BITS

    return loader()

# test the loader
if __name__ == "__main__":
    n_bits = 30
    N_COL = 2 * n_bits - 1  # should equal N_COL globally
    loader = IP_generator(n_bits=n_bits, batch=10, exhaustive=False)
    print("train1")
    BITS = next(loader)
    print("train2")
    BITS = next(loader)
    
    loader = IP_generator(n_bits=n_bits, batch=10, exhaustive=True)
    print("test")
    BITS = next(loader)
    # print("BITS shape:", BITS.shape)  # [Batch, N_COL, n_bits]

    # for i in range(BITS.shape[0]):
    #     print(f"\nSample #{i}:")
    #     for j in range(BITS.shape[1]):
    #         bits_in_col = BITS[i, j, :].tolist()
    #         print(f"  Col {j}: bits = {bits_in_col}")

