# gen_myhdl_compressors_sop.py
# Generate MyHDL compressors with ports from ppa.json and SOP assigns from ac_lib.json

import json
from pathlib import Path
import re

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
        lib[name] = {"width": w, "s_tt": s_tt, "c_tt": c_tt}
    return lib

def _load_ppa(path):
    with open(path, "r") as f:
        data = json.load(f)
    ppa = {}
    for name, ent in data.items():
        if not isinstance(ent, dict): continue
        ins  = list(ent["inputs"])
        outs = list(ent["outputs"])
        if len(set(ins)) != len(ins) or len(set(outs)) != len(outs):
            raise ValueError(f"[ppa] duplicate port names for '{name}'")
        ppa[name] = {"inputs": ins, "outputs": outs}
    return ppa

def _py_ident(s: str) -> str:
    s2 = re.sub(r'[^0-9a-zA-Z_]', '_', s)
    if re.match(r'^[0-9]', s2): s2 = "comp_" + s2
    return s2

def _map_outputs_to_sc(outputs):
    """
    Decide which output is sum vs carry. Returns [('name','s'|'c'), ...]
    Rules: 'S' -> sum, 'C' -> carry; names ending '1' -> sum, ending '2' -> carry;
    fallback (2 outs): first=carry, second=sum.
    """
    mapped = []
    for on in outputs:
        u = on.upper()
        if u == "S": mapped.append((on,'s'))
        elif u == "C": mapped.append((on,'c'))
        elif on.endswith('1'): mapped.append((on,'s'))
        elif on.endswith('2'): mapped.append((on,'c'))
        else: mapped.append((on,None))
    if any(k is None for _,k in mapped):
        tmp=[]; 
        for i,(on,k) in enumerate(mapped):
            if k is None: k = 'c' if i==0 else 's'
            tmp.append((on,k))
        mapped = tmp
    return mapped

def _prod_term(idx, inputs):
    """
    Return a product term string using bit-wise ops.
        pin = 1 →  pin
        pin = 0 → ~pin
    products are ANDed with '&'.
    """
    parts = []
    for i, name in enumerate(inputs):          # inputs[0] = pin0 (LSB)
        if (idx >> i) & 1:
            parts.append(f"{name}")            #  literal
        else:
            parts.append(f"~{name}")           #  negated
    # Join with '&' (bit-wise AND).  Parenthesize the whole product.
    return " & ".join(parts) if parts else "0"

def _sop_expr(tt, inputs):
    """OR (|) together every product term where tt[idx]==1."""
    on_idxs = [i for i, b in enumerate(tt) if b]
    if not on_idxs:
        return "0"
    products = [f"({ _prod_term(i, inputs) })" for i in on_idxs]
    return " | ".join(products)

def yosys_scripts(cmp_name):
    return f'''
read_verilog -sv /Users/fch/Python/OPACT-Extension/ACs/{cmp_name}.v
read_verilog -sv /Users/fch/Python/OPACT-Extension/General_OPACT_MIP/json2v/ACs/myhdl_{cmp_name}.v
prep; proc; opt; memory;
clk2fflogic;
miter -equiv -flatten {cmp_name} myhdl_{cmp_name} miter
sat -seq 20 -verify -prove trigger 0 -show-inputs -show-outputs -set-init-zero miter
    '''

def generate_myhdl_compressors_sop(ac_lib_path, ppa_path,
                                   out_py="myhdl_compressors_sop.py",
                                   only_names=None):
    ac  = _load_ac_lib(ac_lib_path)
    ppa = _load_ppa(ppa_path)
    names = sorted(set(ac.keys()) & set(ppa.keys()))
    if only_names: 
        only = set(only_names)
        names = [n for n in names if n in only]
    if not names:
        raise SystemExit("No overlapping compressor names in ac_lib.json & ppa.json")

    out = []
    out.append("# Auto-generated MyHDL compressors (SOP form)")
    out.append("# Ports (names/order) from ppa.json; behavior from ac_lib.json")
    out.append("from myhdl import block, always_comb, Signal, intbv")
    out.append("import os")
    out.append("")
    
    for n in names:
        w    = ac[n]["width"]
        s_tt = ac[n]["s_tt"]
        c_tt = ac[n]["c_tt"]
        ins  = ppa[n]["inputs"]
        outs = ppa[n]["outputs"]
        if len(ins) != w:
            raise ValueError(f"{n}: ppa inputs({len(ins)}) != width({w})")
        fn   = _py_ident(n)
        out_map = _map_outputs_to_sc(outs)

        out.append(f"@block")
        args = ", ".join(outs + ins)  # outputs first, then inputs
        out.append(f"def {fn}({args}):")
        out.append(f"    \"\"\"Compressor '{n}' (width={w}) — SOP form.")
        out.append( "    idx mapping: inputs[0] is LSB (pin0) for truth tables.")
        out.append( "    \"\"\"")
        out.append(f"    @always_comb")
        out.append(f"    def logic():")
        # defaults
        # for on, _k in out_map:
        #     out.append(f"        {on}.next = 0")
        # SOP expressions for each output
        sop_s = _sop_expr(s_tt, ins)
        sop_c = _sop_expr(c_tt, ins)
        # drive by mapping (keep output order as in ppa.json)
        for on, kind in out_map:
            expr = sop_s if kind=='s' else sop_c
            out.append(f"        {on}.next = ({expr})")
        out.append(f"    return logic")
        out.append("")

        # --- convert helper ---
        helplines=[]
        # helplines.append(f"def convert_{fn}(hdl='Verilog', fname=None, path=None):")
        # helplines.append(f"    \"\"\"Convert {fn} to HDL (default Verilog).")
        # helplines.append( "    fname overrides output filename.\"\"\"")
        # # create fresh Signals
        # for o in outs: helplines.append(f"    {o} = Signal(bool(0))")
        # for i in ins : helplines.append(f"    {i} = Signal(bool(0))")
        # args=", ".join(outs+ins)
        # helplines.append(f"    dut = {fn}({args})")
        # helplines.append(f"    fname = fname or '{fn}'")
        # helplines.append(f"    dut.convert(hdl=hdl, path=path, name=fname)")
        # helplines.append(f"\nconvert_{fn}('verilog',fname='myhdl_{fn}', path='/Users/fch/Python/OPACT-Extension/General_OPACT_MIP/json2v/ACs')")
        # helplines.append(f"os.system(\"yosys -p 'read_verilog /Users/fch/Python/OPACT-Extension/General_OPACT_MIP/json2v/ACs/myhdl_{fn}.v; aigmap; write_aiger /Users/fch/Python/OPACT-Extension/General_OPACT_MIP/json2v/ACs/myhdl_{fn}.aig;'\")")
        out.append("\n".join(helplines))
        out.append("")

        # --- yosys script for verification ---
        yosys_script = yosys_scripts(fn)
        with open(f"/Users/fch/Python/OPACT-Extension/General_OPACT_MIP/json2v/ACs/{fn}.ys", "w") as f:
            f.write(yosys_script)

    Path(out_py).write_text("\n".join(out))
    print(f"Wrote {out_py} with {len(names)} SOP compressor blocks.")
    return out_py

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate SOP-form MyHDL compressors")
    ap.add_argument("--lib", help="ac_lib.json path", default="/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json")
    ap.add_argument("--ppa", help="ppa.json path (port names)", default="/Users/fch/Python/OPACT-Extension/ACs/ppa.json")
    ap.add_argument("--out", default="/Users/fch/Python/OPACT-Extension/General_OPACT_MIP/json2v/myhdl_compressors_sop.py", help="output .py")
    ap.add_argument("--only", nargs="*", help="subset of names to emit")
    args = ap.parse_args()
    generate_myhdl_compressors_sop(args.lib, args.ppa, args.out, args.only)
