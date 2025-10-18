# comp_lut.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List
from myhdl import block, always_comb, Signal, intbv

@dataclass
class CompType:
    name: str
    width: int
    s_tt: List[int]     # truth_s (len = 2^width)
    c_tt: List[int]     # truth_c (len = 2^width)
    same_col: bool      # True if name contains 'ew'
    area: float = 0.0   # optional (not required for gen)

class CompLib:
    def __init__(self, types: List[CompType]):
        self.types = types

    @staticmethod
    def load_from_json(path: str) -> "CompLib":
        with open(path, "r") as f:
            data = json.load(f)

        types: List[CompType] = []

        if isinstance(data, dict):
            # dict-of-dicts keyed by name (your current format)
            items = list(data.items())  # preserves insertion order (Py3.7+)
        elif isinstance(data, list):
            # fallback: list of entries each with a "name"
            items = [(ent.get("name", f"type{i}"), ent) for i, ent in enumerate(data)]
        else:
            raise ValueError("ac_lib.json must be an object or array")

         # ---- Sort by name to match C++ ordering (std::string lexicographic) ----
        items.sort(key=lambda kv: kv[0])  # case-sensitive ASCII order

        for i, (name, ent) in enumerate(items):
            if not isinstance(ent, dict):
                raise ValueError(f"Entry {i} for '{name}' must be an object")
            w = int(ent["width"])
            s_tt = [int(x) for x in ent["truth_s"]]
            c_tt = [int(x) for x in ent["truth_c"]]
            if len(s_tt) != (1 << w) or len(c_tt) != (1 << w):
                raise ValueError(f"Truth table length mismatch for '{name}' (width={w})")
            same_col = ("ew" in str(name))
            area = float(ent.get("area", 0.0))
            types.append(CompType(name=str(name), width=w, s_tt=s_tt, c_tt=c_tt,
                                  same_col=same_col, area=area))
        return CompLib(types)

    def by_id(self, type_id: int) -> CompType:
        return self.types[type_id]

@block
def LUTCompressor(S, C, X, s_tt, c_tt):
    """
    Truth-table compressor with a *vector* input X[W:].
    Index convention: idx = int(X) = Σ X[i] << i  (pin 0 is LSB).
    Converts to a simple combinational ladder in Verilog.
    """
    W = int(X._nrbits)

    @always_comb
    def lut():
        # defaults
        S.next = 0
        C.next = 0
        x = int(X)                  # read the vector (establishes sensitivity)
        # MyHDL convertible: range-loop + if/elif ladder
        for idx in range(1 << W):
            if x == idx:
                S.next = int(s_tt[idx])
                C.next = int(c_tt[idx])

    return lut

def convert_LUTCompressor(hdl, Comp:CompType):
    
    CT1 = LUTCompressor(
        S=Signal(intbv(0)[1:]),
        C=Signal(intbv(0)[1:]),
        X=Signal(intbv(0)[Comp.width:]),
        s_tt=Comp.s_tt,
        c_tt=Comp.c_tt
    )
    CT1.convert(hdl);

# lib = CompLib.load_from_json("/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json")
# for i, t in enumerate(lib.types):
#     print(i, t.name)