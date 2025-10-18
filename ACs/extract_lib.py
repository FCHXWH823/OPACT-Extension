#!/usr/bin/env python
import json, subprocess, re, itertools, textwrap
from pathlib import Path
import sympy as sp
import pandas as pd
import aigverse
import os

AC_DIR   = Path("/Users/fch/Python/OPACT-Extension/ACs")
PPA_FILE = Path("/Users/fch/Python/OPACT-Extension/ACs/ppa.json")
OUT_META = Path("/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json")
OUT_PY   = Path("/Users/fch/Python/OPACT-Extension/ACs/pairs.py")

device   = "cpu"   # only used for torch later

# ---------------------------------------------------- helpers -------------
def yosys_tt(verilog: Path, top: str, inputs: list[str], outputs: list[str]):
    """
    Run `yosys` + `abc -tt` to obtain truth-table lines "inp_bits output_bits".
    Returns list[str] of 2**n lines.
    """
    script = textwrap.dedent(f"""
        read_verilog {verilog}
        hierarchy -check -top {top}
        proc; opt; techmap; opt
        write_blif -gates mytmp.blif
    """)
    (Path("myy.ys")).write_text(script)
    subprocess.run(["yosys", "-q", "-s", "myy.ys"], check=True)

    # call abc to dump truth table
    res = subprocess.run(
        ["abc", "-c", "read_blif mytmp.blif; strash; tt -x"],
        capture_output=True, text=True, check=True
    ).stdout.splitlines()

    # lines look like  "0000 01" (inp_bits LSB-first, outputs MSB-first)
    tt = [ln.strip() for ln in res if re.fullmatch(r"[01]+ [01]+", ln)]
    assert len(tt) == 2 ** len(inputs), "Incomplete truth table extracted"
    return tt

def print_tt(tt_vecs, input_ports, output_ports):
    port_str = ""
    for port in input_ports:
        port_str += port+" "
    port_str +="|"
    for port in output_ports:
        port_str += " "+port
    print(port_str)
    for tt in tt_vecs:
        tt_str = ""
        for i in range(len(input_ports)):
            tt_str += str(tt[i])+" "
        tt_str += "|"
        for i in range(len(output_ports)):
            tt_str += " "+str(tt[i+len(input_ports)])
        print(tt_str)

# bits_str: x1x2x3x4 CS
def simulate_AC(verilog, input_ports=["x1","x2","x3","x4"], output_ports=["C","S"]):
    # design = ys.Design()
    # module = verilog.split("/")[-1]
    # ys.run_pass(f"read_verilog {verilog}.v", design)
    # ys.run_pass("aigmap", design)
    # ys.run_pass(f"write_aiger {verilog}.aig", design)
    aig = aigverse.read_pla_into_aig(verilog)
    tts = aigverse.simulate(aig)
    bits_str = [''] * (1<< len(input_ports))
    # bits_vec = [[int((i & (1<<j))>>j) for j in range(len(input_ports))] for i in range(1<< len(input_ports))]
    for i in range(1<< len(input_ports)):
        for j in range(len(input_ports)):
            bits_str[i] += str((i & (1<<j))>>j)
    for i in range(1<< len(input_ports)):
        bits_str[i] += ' '
    # bits_str = [bits_str[i]+' ' for i in range(1<< len(input_ports))]
    for i in range(1<< len(input_ports)):
        for j, tt in enumerate(tts):
            # bits_vec[i].append(int(tt.get_bit(i)))
            bits_str[i] += str(int(tt.get_bit(i)))
    # print("Truth Table of", verilog)
    # print_tt(bits_vec,input_ports,output_ports)
    return bits_str

def expectation_poly(tt: list[str], inp_bits: int, out_idx: int):
    """
    Fit E[out | p] for a single output bit (sum or carry).
    """
    p = sp.symbols('p')
    prob = 0
    for line in tt:
        inp, out = line.split()
        k = inp.count('1')
        if out[out_idx] == '1':
            prob += p**k * (1-p)**(inp_bits-k)
    prob = sp.expand(prob)
    return str(prob)

def error_poly0(tt_lines, n_in):
    """
    Return a string with the expanded polynomial
        E[ | (S + 2*C)_approx  -  popcount | ]  as a function of p.

    *LSB weight = 1* is assumed; in Stage‑1 you still multiply by
    `bit_weight[j]` to account for column significance.
    """
    p = sp.symbols('p')
    expr = 0
    for line in tt_lines:
        inp, out = line.split()
        k   = inp.count('1')
        Sau = int(out[-1])               # S is last output bit
        Cau = int(out[-2])               # C is penultimate output bit
        approx_val = Sau + 2*Cau
        exact_val  = k                   # because popcount literally = sum of inputs
        err = exact_val - approx_val
        if err:
            expr += err * p**k * (1-p)**(n_in-k)
    return str(sp.expand(expr)), sp.expand(expr).subs(p, 0.25)

def error_poly1(tt_lines, n_in):
    """
    Return a string with the expanded polynomial
        E[ | (S + 2*C)_approx  -  popcount | ]  as a function of p.

    *LSB weight = 1* is assumed; in Stage‑1 you still multiply by
    `bit_weight[j]` to account for column significance.
    """
    p = sp.symbols('p')
    expr = 0
    for line in tt_lines:
        inp, out = line.split()
        k   = inp.count('1')
        approx_val = out.count('1') 
        exact_val  = k                   # because popcount literally = sum of inputs
        err = exact_val - approx_val
        if err:
            expr += err * p**k * (1-p)**(n_in-k)
    return str(sp.expand(expr)), sp.expand(expr).subs(p, 0.25)


# def expectation_poly(tt: list[str], inp_bits: int, out_idx: int):
#     """
#     Fit E[out | p] for a single output bit (sum or carry).
#     """
#     # p = sp.symbols('p')
#     ps = [sp.symbols(f'p{i}') for i in range(inp_bits)]
#     prob = 0
#     for line in tt:
#         inp, out = line.split()
#         # k = inp.count('1')
#         if out[out_idx] == '1':
#             pd = 1
#             for i in range(inp_bits):
#                 if inp[i] == '1':
#                     pd *= ps[i]
#                 else:
#                     pd *= (1 - ps[i])
#             prob += pd
#     prob = sp.expand(prob)
#     return str(prob)

# ---------------------------------------------------- main ----------------
ppa = json.loads(PPA_FILE.read_text())
records = {}

for pla in AC_DIR.glob("*.pla"):
    name = pla.stem
    area = ppa.get(name, {}).get("area", None)
    assert area is not None, f"{name} missing in ppa.json"

    # heuristic: top module = file stem, inputs = x1..xn, outputs = [C,S0,S1?]
    inputs = ppa.get(name, {}).get("inputs", [])
    outputs = ppa.get(name, {}).get("outputs", [])
    n_in = len(inputs)
    tt = simulate_AC(str(pla.absolute()), inputs, outputs)
    # tt = yosys_tt(pla, name, inputs, outputs)

    carry_poly = expectation_poly(tt, n_in, 0)
    sum_poly   = expectation_poly(tt, n_in, 1)

    # get the truth table for each output bit and store it as a list of int
    carry_tt = [int(line.split(' ')[-1][0]) for line in tt]
    sum_tt   = [int(line.split(' ')[-1][1]) for line in tt]

    if "ew" in name:
        err_poly, err_val = error_poly1(tt, n_in)
    else:
        err_poly, err_val = error_poly0(tt, n_in)

    records[name] = {
        "width":     n_in,
        "area":      area,
        "med":       err_poly,
        "poly_C":    carry_poly, 
        "poly_S":    sum_poly,
        "err_val":   f"{err_val:.3f}",
        "truth_s": sum_tt,
        "truth_c": carry_tt,
        "input_ports": inputs,
        "output_ports": outputs
    }

# dump metadata
OUT_META.write_text(json.dumps(records, indent=2))
print(f"wrote {OUT_META}")

# auto‑generate Python lambdas for import
with OUT_PY.open("w") as f:
    f.write("import torch\n\n")
    f.write("if torch.cuda.is_available():\n    device = \"cuda\"\n\n")
    f.write("elif torch.backends.mps.is_available():\n    device = \"mps\"\n\n")
    f.write("else:\n    device = \"cpu\"\n\n")

    ac_names = sorted(records)

    for ac_name in ac_names:
        rec = records[ac_name]
        if rec['poly_S'] in ("0", "0.0"):
            f.write(f"def {ac_name}_S(p):\n    return torch.zeros_like(p, device = device)\n\n")
        elif rec['poly_S'] in ("1", "1.0"):
            f.write(f"def {ac_name}_S(p):\n    return torch.ones_like(p, device = device)\n\n")
        else:
            f.write(f"def {ac_name}_S(p):\n    return ({rec['poly_S']})\n\n")

        if rec['poly_C'] in ("0", "0.0"):
            f.write(f"def {ac_name}_C(p):\n    return torch.zeros_like(p, device = device)\n\n")
        elif rec['poly_C'] in ("1", "1.0"):
            f.write(f"def {ac_name}_C(p):\n    return torch.ones_like(p, device = device)\n\n")
        else:
            f.write(f"def {ac_name}_C(p):\n    return ({rec['poly_C']})\n\n")
        
        if rec['med'] in ("0", "0.0"):
            f.write(f"def {ac_name}_err(p):\n    return torch.zeros_like(p, device = device)\n\n")
        elif rec['med'] in ("1", "1.0"):
            f.write(f"def {ac_name}_err(p):\n    return torch.ones_like(p, device = device)\n\n")
        else:
            f.write(f"def {ac_name}_err(p):\n    return ({rec['med']})\n\n")

        # f.write(f"def {ac_name}_S(p):\n")
        # f.write(f"    return ({rec['poly_S']})\n\n")
        # f.write(f"def {ac_name}_{rec['']}(p):\n")
        # f.write(f"    return ({rec['poly_C']})\n\n")
        # f.write(f"def {ac_name}_err(p):\n")
        # f.write(f"    return ({rec['med']})\n\n")

    fn_list_S = ", ".join([f"{r}_S" for r in ac_names] + ["lambda p: p"])
    fn_list_C = ", ".join([f"{r}_C" for r in ac_names] + ["lambda p: 0*p"])
    fn_list_err = ", ".join([f"{r}_err" for r in ac_names] + ["lambda p: 0*p"])
    f.write(f"F_S_LIST = [{fn_list_S}]\n")
    f.write(f"F_C_LIST = [{fn_list_C}]\n")
    f.write(f"F_ERR_LIST = [{fn_list_err}]\n")
print(f"wrote {OUT_PY}")
