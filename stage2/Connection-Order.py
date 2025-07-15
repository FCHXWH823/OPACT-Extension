import torch
import torch.nn.functional as F
from typing import List, Dict

from generator import *
from compressors import *

# pp, pb shape = [N_STAGE, N_COL, N_PIN/ N_BIT]
# comp[i][j] = list of compressors in stage i, column j, each like {'type': 'a32', 'pins': [0,1,2], 'bits': [0,1]}
# cp[i][j] = [n_bit, n_pin] float tensor with requires_grad=True

# 假设：
N_STAGE = 2
N_COL = 5
n_bits = (N_COL+1) // 2

# 构造一个 demo count tensor，每个 stage、每个 column 上人为设置的压缩器个数
demo_counts = torch.zeros((N_STAGE, N_COL, 5), dtype=torch.float)

# 手动指定每个 compressor 数量（整数）
# Stage 0
# demo_counts[0, 0, 0] = 0  # ex-3:2
demo_counts[0, 1, 1] = 1  # ex-2:2
demo_counts[0, 3, 1] = 1  # ex-2:2
demo_counts[0, 2, 4] = 3  # dummy
# demo_counts[0, 3, 3] =   # ap-4:2
demo_counts[0, 0, 4] = 1  # dummy
demo_counts[0, 4, 4] = 1  # dummy

# Stage 1
demo_counts[1, 0, 4] = 1
demo_counts[1, 1, 4] = 1
demo_counts[1, 2, 3] = 1
demo_counts[1, 3, 4] = 1
demo_counts[1, 4, 4] = 2

pin_index = [ [0 for _ in range(N_COL)] for _ in range(N_STAGE) ]
bit_index = [ [0 for _ in range(N_COL)] for _ in range(N_STAGE) ]

def build_comp(counts_tensor: torch.Tensor):
    """
    convert compressor count of each stage and column from counts tensor (shape [N_STAGE, N_COL, 5]) into structured compressor list.
    parameter:
        counts_tensor: shape = [N_STAGE, N_COL, 5]

    output:
        comp: List[stage][col] = List[compressor dict]
    """
    assert counts_tensor.shape == (N_STAGE, N_COL, 5)
    comp = []

    for s in range(N_STAGE):
        stage = []
        for j in range(N_COL):
            n_ex32, n_ex22, n_ap32, n_ap42, n_dummy = counts_tensor[s, j].int().tolist()

            comps_col = []
            def alloc_pins(n):
                idx = list(range(pin_index[s][j], pin_index[s][j] + n))
                pin_index[s][j] += n
                return idx

            def alloc_bits(n, carry=False):
                col_out = j + 1 if carry else j
                idx = list(range(bit_index[s][col_out], bit_index[s][col_out] + n))
                bit_index[s][col_out] += n
                return [(col_out, b) for b in idx]


            # init compressors in this column
            # type: 0 = exact-3:2, 1 = exact-2:2, 2 = approx-3:2, 3 = approx-4:2
            # 5 = dummy pass (no compression, just pass through pins)
            for _ in range(n_ex32):
                comps_col.append({'type': 0,
                                  'pins': alloc_pins(3),
                                  'bits': alloc_bits(1, carry=False)+alloc_bits(1, carry=True)})

            for _ in range(n_ex22):
                comps_col.append({'type': 1,
                                  'pins': alloc_pins(2),
                                  'bits': alloc_bits(1, carry=False)+alloc_bits(1, carry=True)})

            for _ in range(n_ap32):
                comps_col.append({'type': 2,
                                  'pins': alloc_pins(3),
                                  'bits': alloc_bits(2, carry=False)})

            for _ in range(n_ap42):
                comps_col.append({'type': 3,
                                  'pins': alloc_pins(4),
                                  'bits': alloc_bits(2, carry=False)})
            if n_dummy > 0:
                for _ in range(n_dummy):
                    comps_col.append({'type': 4,
                                      'pins': alloc_pins(1),
                                      'bits': alloc_bits(1, carry=False)})

            stage.append(comps_col)
        comp.append(stage)
    return comp

def init_param() -> List[List[torch.Tensor]]:
    cp_logits = []
    for s in range(N_STAGE-1):  # stage[i] → stage[i+1]
        stage_cp_logits = []
        for j in range(N_COL):
            param = torch.randn(bit_index[s][j], pin_index[s+1][j], requires_grad=True, device=device)
            stage_cp_logits.append(param)
        cp_logits.append(stage_cp_logits)
    return cp_logits

def compress_stage(pp_stage, comp_stage, stage):
    """
    Input:
        pp_col: [N_PIN] 当前 column 的 pin 概率
        comp_list: list of compressors with pins and bits
    Output:
        pb_col: [N_BIT] 当前 column 的 bit 概率
        comp_err_col: scalar, compressor error
    """
    pb_stage = [torch.zeros(bit_index[stage][c], device=device) for c in range(N_COL)]
    comp_err = torch.zeros(N_COL, device=device)
    for j in range(N_COL):
        pp_col = pp_stage[j]
        for comp in comp_stage[j]:
            inputs = [pp_col[i] for i in comp['pins']]
            (S, C), comp_err_tmp = F_COMPRESSOR[comp['type']](inputs)
            pb_stage[comp['bits'][0][0]][comp['bits'][0][1]] += S
            if comp['type'] < 4:  # only add carry for non-dummy compressors
                pb_stage[comp['bits'][1][0]][comp['bits'][1][1]] += C

            comp_err[j] += comp_err_tmp

    return pb_stage, comp_err


def connect_stage(pb_stage, logits_stage):
    """
    Input:
        pb_col: [N_BIT] 当前列的 bits 概率
        cp_col: [N_BIT, N_PIN] 概率参数
    Output:
        pp_next_col: [N_PIN] 下一层 column 的 pin 概率
        conn_err: scalar, connection error
    """
    pp_next_stage = []
    conn_err = torch.zeros(N_COL, device=device)
    for j in range(N_COL):
        # softmax over pin dim to get connection probabilities
        conn_prob = F.softmax(logits_stage[j], dim=0)  # [n_bit, n_pin]

        # simulate connection
        pp_next_col = pb_stage[j] @ conn_prob # [n_pin]
        pp_next_stage.append(pp_next_col)
        conn_err[j] = pb_stage[j].sum() - pp_next_col.sum()
    return pp_next_stage, conn_err

def compute_pattern_error(input_pattern, comp, logits):
    """
    execute full compressor propagation and connection propagation for a single input pattern, and compute the error.
    """
    # pp = [input_pattern]
    pp = list(input_pattern)
    total_err = torch.zeros(N_COL, device=device)

    pb, comp_err_stage = compress_stage(pp, comp[0], 0)
    total_err += comp_err_stage

    for i in range(N_STAGE - 1):
        pp_stage, conn_err_stage = connect_stage(pb, logits[i])

        # pp.append(torch.stack(pp_next_stage))

        pb, comp_err_stage = compress_stage(pp_stage, comp[i+1], i+1)

        total_err += comp_err_stage
        total_err += conn_err_stage

    # weighted sum over columns (e.g., 2^{j})
    weight = 2 ** torch.arange(N_COL, device=device)
    pattern_err = torch.abs(torch.sum(total_err * weight))

    return pattern_err


def compute_batch_loss(input_batch, comp, cp):
    loss = 0.0
    for pattern in input_batch:
        loss += compute_pattern_error(pattern, comp, cp)
    return loss / len(input_batch)

def train(comp, logits, num_steps=1000, batch_size=64, lr=1e-2):
    # init DataLoader
    loader = IP_generator(n_bits, batch=batch_size, exhaustive=False)

    # collect parameters for optimization
    cp_params = [p for stage in logits for col in stage for p in [col] if p.requires_grad]
    optimizer = torch.optim.Adam(cp_params, lr=lr)

    for step in range(num_steps):
        # get input pattern batch [B, N_COL, N_PIN]
        _, input_batch = next(loader)

        loss = compute_batch_loss(input_batch, comp, logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1 == 0 or step == num_steps - 1:
            print(f"[Step {step}] Loss = {loss.item():.6f}")

def evaluate(cp, comp, loader, num_batches=10):
    with torch.no_grad():
        total_loss = 0.0
        for _ in range(num_batches):
            _, input_batch = next(loader)
            total_loss += compute_batch_loss(input_batch, comp, cp).item()
    return total_loss / num_batches


# 初始化参数
comp = build_comp(demo_counts)
logits = init_param()
# for j, col in enumerate(comp[0]):
#     print(f"Column {j}:")
#     for c in col:
#         print(c)

# 启动训练
train(comp, logits, num_steps=200, batch_size=64, lr=1e-2)
