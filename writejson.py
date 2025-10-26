import json
from pathlib import Path
from typing import Sequence, Mapping, Union, List

import numpy as np
import torch


def dump_column_allocations(
    compressor_counts: Union[np.ndarray, torch.Tensor, Sequence],
    column_bits:       Union[np.ndarray, torch.Tensor, Sequence],
    file_path:         Union[str, Path],
    *,
    compressor_labels: Sequence[str] = (
        "exact_3to2",   # k = 0
        "exact_2to2",   # k = 1
        "approx_3to2",  # k = 2
        "approx_4to2",  # k = 3
        "dummy",        # k = 4
    ),
) -> None:
    """
    Store, for **every column in every stage**, how many compressors of each
    type were allocated *plus* the current number of bits in that column.

    Parameters
    ----------
    compressor_counts : (S, C, K) array-like
        Integer tensor/ndarray/list with **S stages**, **C columns** and
        **K compressor types** (order must match `compressor_labels`).
    column_bits : (C,) or (S, C) array-like
        For each column: how many *bits* are still present when the allocation
        is logged.  Pass a 1-D vector if identical for every stage, otherwise
        shape (S, C).
    file_path : str or Path
        Destination *.json filename.
    compressor_labels : iterable[str], optional
        Human-readable names for the K compressor types.  Length **must equal**
        `compressor_counts.shape[-1]`.

    JSON layout
    -----------
    {
      "stage0": [
        {
          "col_idx": 0,
          "bits":    17,
          "alloc": { "exact_3to2": 3, "exact_2to2": 1, ... }
        },
        ...
      ],
      "stage1": [ ... ],
      ...
    }
    """
    # ---------- normalise inputs to numpy for easy indexing ---------------
    counts = np.asarray(
        compressor_counts.detach().cpu() if isinstance(compressor_counts, torch.Tensor)
        else compressor_counts,
        dtype=int,
    )
    bits = np.asarray(
        column_bits.detach().cpu() if isinstance(column_bits, torch.Tensor)
        else column_bits,
        dtype=int,
    )

    if bits.ndim == 1:             # same bit-count for every stage
        bits = np.broadcast_to(bits, counts.shape[:2])

    if counts.ndim != 3:
        raise ValueError("compressor_counts must be shape (S, C, K)")
    S, C, K = counts.shape
    if len(compressor_labels) != K:
        raise ValueError("compressor_labels length must equal last dim of counts")
    if bits.shape != (S, C):
        raise ValueError("column_bits must be shape (S, C) or (C,)")

    # ---------- build JSON-serialisable structure -------------------------
    stages: Mapping[str, List[dict]] = {}
    for s in range(S):
        stage_list: List[dict] = []
        for c in range(C):
            alloc = {
                compressor_labels[k]: int(counts[s, c, k]) for k in range(K)
            }
            stage_list.append(
                {
                    "col_idx": c,
                    "bits":    int(bits[s, c]),
                    "alloc":   alloc,
                }
            )
        stages[f"stage{s}"] = stage_list

    # ---------- write file ------------------------------------------------
    Path(file_path).write_text(json.dumps(stages, indent=2))
