import json, subprocess, re, itertools, textwrap
from pathlib import Path
import sympy as sp
import pandas as pd
import aigverse
import os

def verify_Compressor(verilog: str, input_ports=["x1","x2","x3","x4"], output_ports=["C","S"], output_tts=None):
    # design = ys.Design()
    # module = verilog.split("/")[-1]
    # ys.run_pass(f"read_verilog {verilog}.v", design)
    # ys.run_pass("aigmap", design)
    # ys.run_pass(f"write_aiger {verilog}.aig", design)
    aig = aigverse.read_aiger_into_aig(verilog)
    tts = aigverse.simulate(aig)
    real_output_tts = [[] for _ in range(len(output_ports))]
    for i in range(1<< len(input_ports)):
        for j, tt in enumerate(tts):
            real_output_tts[j].append(int(tt.get_bit(i)))
            
    for i, real_tt in enumerate(real_output_tts):
        if real_tt != output_tts[i]:
            print(f"Mismatch in output for {output_ports[i]}: {real_tt} != {output_tts[i]}")
            return False
    return True


def _load_ac_lib(path):
    with open(path, "r") as f:
        data = json.load(f)
    lib = {}
    for name, ent in data.items():
        if not isinstance(ent, dict): continue
        w = int(ent["width"])
        s_tt = [int(x) for x in ent["truth_s"]]
        c_tt = [int(x) for x in ent["truth_c"]]
        if len(s_tt) != (1 << w) or len(c_tt) != (1 << w):
            raise ValueError(f"[ac_lib] len(truth) != 2^width for '{name}'")
        lib[name] = {"width": w, "s_tt": s_tt, "c_tt": c_tt, "input_ports": ent["input_ports"], "output_ports": ent["output_ports"]}
    return lib



libs = _load_ac_lib("/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json")
for lib in libs:
    print(f"Compressor {lib}: width={libs[lib]['width']}, s_tt={libs[lib]['s_tt']}, c_tt={libs[lib]['c_tt']}")
    # Verify the compressor
    if not verify_Compressor(f"/Users/fch/Python/OPACT-Extension/General_OPACT_MIP/json2v/ACs/myhdl_{lib}.aig",input_ports=libs[lib]['input_ports'], output_ports=libs[lib]['output_ports'], output_tts=[libs[lib]['c_tt'], libs[lib]['s_tt']]):
        print(f"Verification failed for {lib}")
    else:
        print(f"Verification passed for {lib}")