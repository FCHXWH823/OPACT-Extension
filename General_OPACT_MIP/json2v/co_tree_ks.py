# co_tree_ks.py
from __future__ import annotations
from typing import List
from myhdl import block, Signal, intbv, always_comb, instances
from comp_lut import CompLib, LUTCompressor
from KoggeStoneAdder import KoggeStone   # reuse your adder
from BasicModule import *
import re
import myhdl_compressors_sop as lib  # your SOP-based compressor blocks


def _mk_bits(n: int) -> List[Signal]:
    return [Signal(bool(0)) for _ in range(n)]


@block
def PPG_MSB(cols: List[List[Signal]], A: Signal, B: Signal,
            width: int, extra_cols: int):
    """
    Populate partial products into MSB‑first columns (index 0 = MSB).

    C = (2*width - 1) + extra_cols
    For AND(A[i], B[k]): LSB col  l = i + k
                         MSB col  j = C - 1 - l
    """
    N, C = width, len(cols)
    assert C == (2 * N - 1) + extra_cols

    @always_comb
    def gen_pp():
        for j in range(C):                # MSB → LSB
            l = (C - 1) - j               # LSB‑first column index

            # i_min = max(0, l - (N‑1))  (use plain if/else so MyHDL infers type)
            i_min = 0
            if l - (N - 1) > 0:
                i_min = l - (N - 1)

            n_bits = l - i_min + 1        # #PP bits that belong to this column
            if n_bits > len(cols[j]):     # cap to allocated slots (unlikely)
                n_bits = len(cols[j])

            for s in range(n_bits):
                i = i_min + s
                k = l - i
                cols[j][s].next = A[i] & B[k]
            # over‑allocated slots (if any) stay at default ‘0’

    return gen_pp

@block
def PPG_cols(cols, A, B, N, stages_num):
    """
    Build partial products into MSB-first columns.
    Total columns C = (2*N - 1) + stages_num; index 0 = MSB.

    For LSB-first column index l = i + k.
    Our MSB-first column index is j = C-1-l.

    We assign each column j exactly the first n_j pairs (i, k=l-i) in
    increasing i, where n_j == len(cols[j]) (created at elaboration time).
    """
    C = (2*N - 1) + stages_num
    assert len(cols) == C

    # Precompute, at elaboration time, the per-column slot counts and the
    # (i0, l) bases used to index A[i] & B[k] deterministically in @always_comb
    L = [0] * C      # number of PP bits in column j (== len(cols[j]))
    I0 = [0] * C     # starting i for column j
    LSB = [0] * C    # LSB-first column index l = C-1-j

    for j in range(C):
        l = (C - 1) - j                 # LSB-first column index
        # valid i are: max(0, l-(N-1)) .. min(N-1, l)
        i_min = l - (N - 1) if l - (N - 1) > 0 else 0
        i_max = (N - 1) if (N - 1) < l else l
        ncol  = (i_max - i_min + 1)
        # sanity: len(cols[j]) should match the number of PP terms we will drive
        if len(cols[j]) != ncol:
            # If mismatch, we still cap by the allocated slots to stay safe
            ncol = min(ncol, len(cols[j]))
        L[j] = ncol
        I0[j] = i_min
        LSB[j] = l

    @always_comb
    def gen_pp():
        for j in range(C):
            # derive everything mathematically – no list look‑ups
            l = (C - 1) - j                     # LSB‑first column index
            i_min = l - (N - 1) if l - (N - 1) > 0 else 0
            n_bits = min(len(cols[j]), l - i_min + 1)

            for s in range(n_bits):
                i = i_min + s
                k = l - i
                cols[j][s].next = A[i] & B[k]
        # No explicit clearing needed: every allocated slot gets a deterministic driver.
        # (If you had over-allocated slots, they'd remain at elaboration as they don't exist.)

    return gen_pp

@block
def PPG(OUTS,A,B,width):
    startIndex = 0; endIndex = 0;
    instances_list = [];
    outs = [];
    for i in range(2*width-1):
        col_len = width - abs(i-width+1);
        endIndex += col_len;
        col_sigs_list = [Signal(intbv(0)[1:]) for j in range(col_len)];
        #col_sigs = ConcatSignal(*reversed(col_sigs_list));
        for j in range(col_len):
            #out_tmp = Signal(intbv(0)[1:]);
            if i <= width-1:
                AND_1 = AND(col_sigs_list[j],A(i-j),B(j)); # instances' output can't be parts of a bit vector
                #print("%s %s %s" % (startIndex + j,i-j,j));
                #print("%s*%s " % (i-j,j))
            else:
                AND_1 = AND(col_sigs_list[j],A(width-1-j),B(i+j+1-width));
                #print("%s*%s " % (width-1-j,i+j+1-width));
                #print("%s %s %s" % (startIndex + j,width-1-j,i+j+1-width));
            instances_list.append(AND_1);
        #print("------------")
        outs += list(col_sigs_list);
        #outs += col_sigs_list;
        startIndex = endIndex;
    outs_vec = ConcatSignal(*outs)  # can't be put into alway_comb section
                                               # in converted verilog files, the concat signals are the same ones in sig list
    @always_comb
    def comb():
        OUTS.next = outs_vec;
    return instances_list,comb;

def _py_ident(name):
    s = re.sub(r'[^0-9a-zA-Z_]', '_', name)
    if s[0].isdigit():
        s = "comp_" + s
    return s

def make_id2block(lib_names):
    """Map type id -> MyHDL block constructor (positional ports: (C,S, in...))."""
    id2blk = []
    for name in lib_names:
        fn = getattr(lib, _py_ident(name))
        id2blk.append(fn)
    return id2blk

@block
def CompressorTree(OUTS, INPUTS, width, stages, lib_names):
    """
    OUTS: Signal(intbv)[Nout:]
    INPUTS: Signal(intbv)[width*width:]
    co_map: dict loaded from gurobi_co_map.json
    lib_names: list of compressor names sorted lexicographically (C++ order)
    """
    id2blk = make_id2block(lib_names)

    # # Collect stages in increasing order: "stage0", "stage1", ...
    # def _stage_key(k):
    #     m = re.match(r"stage(\d+)$", k)
    #     return int(m.group(1)) if m else 10**9
    # stage_keys = sorted([k for k in co_map.keys() if k.startswith("stage")], key=_stage_key)
    # stages = [co_map[k] for k in stage_keys]

    nCols = len(stages[0])  # column count is fixed across stages
    instances_list = []

    # ---------- Build Stage 0 left-bit signals from INPUTS ----------
    # We respect the *order and counts* of stage0[j]['left_bits'].
    cols_cur = []
    bit_idx = 0  # INPUTS is MSB->LSB; we follow your previous practice of linear slicing
    for j in range(nCols):
        L = len(stages[0][j]['left_bits'])  # use exactly the declared list length
        col_sigs = [INPUTS(bit_idx + k) for k in range(L)]
        bit_idx += L
        cols_cur.append(col_sigs)

    # ---------- Walk each stage, instantiate compressors, and build next stage ----------
    for s in range(len(stages)-1):
        stage = stages[s]
        stage_next = stages[s+1]

        # For building next stage: queues for pass-through remain bits, and a lookup for produced outputs
        remain_queue = [[] for _ in range(nCols)]
        produced = {}  # key: (src_stage, src_col, type, inst, out) -> Signal

        # 1) For each column in stage s: wire right_ports via right_to_left, group by (type,inst)
        for j in range(nCols):
            col = stage[j]
            left_sigs = cols_cur[j]

            # (a) Pass-through (remain) ports: first n_remain_ports entries
            nrp = col['n_remain_ports']
            rtl = col['right_to_left']
            for p in range(nrp):
                li = rtl[p]
                remain_queue[j].append(left_sigs[li])  # preserve the declared order

            # (b) Gather compressor inputs by (type,inst)
            groups = {}
            for p in range(nrp, len(col['right_ports'])):
                rp = col['right_ports'][p]
                t = rp['type']; inst = rp['inst']; pin = rp['pin']
                li = rtl[p]
                if t < 0:
                    continue
                groups.setdefault((t, inst), []).append((pin, left_sigs[li]))

            # (c) Instantiate each compressor once, with pins sorted by pin index (0..w-1)
            for (t, inst), plist in sorted(groups.items()):
                plist.sort(key=lambda x: x[0])
                pins = [sig for (_, sig) in plist]

                # Outputs: (carry, sum) positional
                C = Signal(bool(0))
                S = Signal(bool(0))

                # Instantiate the block
                blk = id2blk[t](C, S, *pins)
                instances_list.append(blk)

                # Record outputs for stage s+1, by their declared origin and out index:
                # convention: out=0 -> sum, out=1 -> carry
                # carry lands in the same column iff same_col, otherwise goes to the next-more-significant column.
                produced[(s, j, t, inst, 0)] = S
                produced[(s, j, t, inst, 1)] = C

        # 2) Build cols_next strictly following stage_{s+1}[j]['left_bits'] order
        cols_next = []
        for j in range(nCols):
            lb_list = stage_next[j]['left_bits']
            col_next = []
            # We will pop from remain_queue[j] in order for type==-1 entries
            rq_idx = 0
            for lb in lb_list:
                t = lb['type']
                if t == -1:
                    sig = remain_queue[j][rq_idx]
                    rq_idx += 1
                else:
                    key = (lb['src_stage'], lb['src_col'], t, lb['inst'], lb['out'])
                    sig = produced[key]
                col_next.append(sig)
            cols_next.append(col_next)

        cols_cur = cols_next

    # ---------- Flatten last stage columns (skip the rightmost/LSB column) ----------
    outs_list = []
    last_cols = cols_cur
    for j in range(nCols):           # like your former flow: omit the last column
        outs_list += last_cols[j]
    outs_vec = ConcatSignal(*outs_list)  # bus LSB on the right, like before

    @always_comb
    def comb():
        OUTS.next = outs_vec

    return instances_list + [comb]

@block
def Multiplier(A,B,OUTS,width,stages, lib_names):
    nOutBits_ppg = width*width;
    OutBits_ppg = Signal(intbv(0)[nOutBits_ppg:]);
    PPG_1 = PPG(OutBits_ppg,A,B,width);
    CT_1 = CompressorTree(OUTS,OutBits_ppg, width, stages, lib_names);
    return instances();

@block
def Multiplier_KoggeStone(A,B,OUTS,width,stages, lib_names):
    last_stage = stages[-1]
    inputs_len = 0;
    columns = [];
    for i in range(len(last_stage)):
        if len(last_stage[i]["left_bits"]) != 0:
            columns.append(len(last_stage[i]["left_bits"]))
            inputs_len += len(last_stage[i]["left_bits"])
    columns.reverse();  # Kogge–Stone expects LSB-first columns
    OutBits_ct = Signal(intbv(0)[inputs_len:])
    Multiplier_ins = Multiplier(A,B,OutBits_ct,width,stages, lib_names)
    KoggeStone_ins = KoggeStone(OUTS,OutBits_ct,columns)
    return instances()

def convert_Multiplier_KoggeStone(hdl,co_map, lib_names, width, path):
    # Collect stages in increasing order: "stage0", "stage1", ...
    def _stage_key(k):
        m = re.match(r"stage(\d+)$", k)
        return int(m.group(1)) if m else 10**9
    stage_keys = sorted([k for k in co_map.keys() if k.startswith("stage")], key=_stage_key)
    stages = [co_map[k] for k in stage_keys]
    A= Signal(intbv(0)[width:]);
    B = Signal(intbv(0)[width:]);
    last_stage = stages[-1];
    outputs_len = 0;
    for i in range(len(last_stage)):
        if len(last_stage[i]["left_bits"]) != 0:
            outputs_len += 1;
    outputs_len += 1;
    OUTS = Signal(intbv(0)[outputs_len:]);
    Multiplier_KoggeStone_ins = Multiplier_KoggeStone(A,B,OUTS,width,stages, lib_names);
    Multiplier_KoggeStone_ins.convert(hdl,path = path, name = "approx_mult")
