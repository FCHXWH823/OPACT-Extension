# convert_from_co.py
from __future__ import annotations
import sys
from typing import List
from myhdl import block, Signal, intbv
from comp_lut import CompLib
from co_tree_ks import *
import json
import os

def main():
    # if len(sys.argv) < 6:
    #     print("Usage: python convert_from_co.py <co_json> <ac_lib.json> <width> <stages_num> <out_prefix>")
    #     sys.exit(1)

    co_json = "/Users/fch/Python/OPACT-Extension/General_OPACT_MIP/gurobi_co_map.json"
    lib_json = "/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json"
    width    = 16
    if (2 * width > 20):
        NTEST = (1 << 20)
    else:
        NTEST = (1 << (2 * width))
    prefix   = "/Users/fch/Python/OPACT-Extension/General_OPACT_MIP/json2v/"
    simulation_path = "/Users/fch/Python/ApproximateMult/ApproxMULT_MyHDL/ApproxMULT_MyHDL/simulation/"
    with open(co_json, "r") as f:
        co_map = json.load(f)
    lib = CompLib.load_from_json(lib_json)
    lib_names = [t.name for t in lib.types]
    convert_Multiplier_KoggeStone("Verilog", co_map, lib_names, width, simulation_path)

    os.system(("cd " + simulation_path + "; ./run.sh -w " + str(width) + " -n " + str(NTEST) + " -m approx_mult;"));
    os.system(("cd " + simulation_path + "; ./sim"));


if __name__ == "__main__":
    main()