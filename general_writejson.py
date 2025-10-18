import json
from pathlib import Path
from typing import Sequence, Mapping, Union, List

import numpy as np
import torch

# ------------------------------------------------------------------------
#  bring in the dynamic compressor names  (dummy already included)
# ------------------------------------------------------------------------
from general_parameters import COMP_NAMES              # ← NEW

def dump_column_allocations(
    compressor_counts: Union[np.ndarray, torch.Tensor, Sequence],
    column_bits:       Union[np.ndarray, torch.Tensor, Sequence],
    file_path:         Union[str, Path],
    *,
    compressor_labels: Sequence[str] = COMP_NAMES   # ← NEW default
) -> None:
    """
    Store, for every column in every stage, how many compressors of each
    type were allocated *plus* the current number of bits in that column.

    Parameters
    ----------
    compressor_counts : (S, C, K) int array‑like
        Allocation per stage (S), column (C), compressor type (K).
    column_bits : (C,) or (S, C) int array‑like
        Bit‑count remaining in each column when logged.
    file_path : str or Path
        Destination *.json* filename.
    compressor_labels : list[str], optional
        Human‑readable names; defaults to global COMP_NAMES.
    """
    # ---------- normalise inputs to NumPy --------------------------------
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(int)   # ← NEW .detach()
        return np.asarray(x, dtype=int)

    counts = to_numpy(compressor_counts)
    bits   = to_numpy(column_bits)

    if bits.ndim == 1:                       # broadcast 1‑D → (S,C)
        bits = np.broadcast_to(bits, counts.shape[:2])

    if counts.ndim != 3:
        raise ValueError("compressor_counts must be 3‑D (S,C,K)")
    S, C, K = counts.shape
    if len(compressor_labels) != K:
        raise ValueError("compressor_labels length != K dimension of counts")
    if bits.shape != (S, C):
        raise ValueError("column_bits must be shape (S,C) or (C,)")

    # ---------- build JSON‑serialisable structure ------------------------
    stages: Mapping[str, List[dict]] = {}
    for s in range(S):
        stage_list: List[dict] = []
        for c in range(C):
            alloc = {
                compressor_labels[k]: int(counts[s, c, k]) for k in range(K)
            }
            stage_list.append(
                {"col_idx": c, "bits": int(bits[s, c]), "alloc": alloc}
            )
        stages[f"stage{s}"] = stage_list

    # ---------- write file ----------------------------------------------
    Path(file_path).write_text(json.dumps(stages, indent=2))
