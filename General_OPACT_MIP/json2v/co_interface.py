# co_interface.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List

@dataclass
class LeftBit:
    type: int   # -1 for remain
    inst: int
    out: int    # -1 for remain; else 0=sum, 1=carry
    src_stage: int
    src_col: int

@dataclass
class Port:
    type: int   # -1 for remain port; else compressor type id
    inst: int
    pin: int

@dataclass
class ColumnSol:
    n_remain_left: int
    n_remain_ports: int
    left_bits: List[LeftBit]      # order used by MILP: Remain → Local → Carries
    right_ports: List[Port]       # order: Remain ports → per-type pins
    right_to_left: List[int]      # driver left index per right port

@dataclass
class StageSol:
    columns: List[ColumnSol]

@dataclass
class CO_Solution:
    stages: List[StageSol]

def _lb_from_json(o) -> LeftBit:
    return LeftBit(
        type=int(o["type"]),
        inst=int(o["inst"]),
        out=int(o["out"]),
        src_stage=int(o["src_stage"]),
        src_col=int(o["src_col"])
    )

def _port_from_json(o) -> Port:
    return Port(
        type=int(o["type"]),
        inst=int(o["inst"]),
        pin=int(o["pin"])
    )

def load_co_json(path: str) -> CO_Solution:
    with open(path, "r") as f:
        root = json.load(f)

    stages: List[StageSol] = []
    idx = 0
    while True:
        key = f"stage{idx}"
        if key not in root:
            break
        cols_raw = root[key]
        cols: List[ColumnSol] = []
        for col in cols_raw:
            left_bits = [_lb_from_json(x) for x in col["left_bits"]]
            right_ports = [_port_from_json(x) for x in col["right_ports"]]
            right_to_left = [int(u) for u in col["right_to_left"]]
            cols.append(ColumnSol(
                n_remain_left=int(col["n_remain_left"]),
                n_remain_ports=int(col["n_remain_ports"]),
                left_bits=left_bits,
                right_ports=right_ports,
                right_to_left=right_to_left
            ))
        stages.append(StageSol(columns=cols))
        idx += 1

    return CO_Solution(stages=stages)
