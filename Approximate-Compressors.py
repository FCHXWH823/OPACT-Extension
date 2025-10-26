from myhdl import block,always_comb,Signal
from myhdl import instances, Signal, intbv, delay
from myhdl import *
import random
from pyosys import libyosys as ys 
import matplotlib.pyplot as plt
import numpy as np
import aigverse
import os

# module : no need of width specification for input ports;
#          need of width specification for internal output ports;
#          better to specify the width in paramters of function (especially, hierarchical designs).
# convert :need of width specification for input ports;
# simulate : need of width specification for input ports;
def convert_AC(hdl,AC):
    x1 = Signal(bool(0))
    x2 = Signal(bool(0))
    x3 = Signal(bool(0))
    x4 = Signal(bool(0))
    S = Signal(bool(0))
    C = Signal(bool(0))
    AC(C,S,x1,x2,x3,x4).convert(hdl)

@block
def AC42_uw1(C,S,x1,x2,x3,x4):
    @always_comb
    def comb():
        C.next = ~((~(x1 | x2)) | (~(x3 | x4)))
        S.next = ((~(x1 ^ x2)) | (~(x3 ^ x4)))
    return comb



@block
def AC42_uw2(C,S,x1,x2,x3,x4):
    @always_comb
    def comb():
        C.next = (x1 & x2) | (x3 & x4)
        S.next = (x1 ^ x2) | ((x1 & x2) & (x3 & x4)) | (x3 ^ x4)
    return comb

@block
def AC42_uw3(C,S,x1,x2,x3,x4):
    @always_comb
    def comb():
        C.next = (x1 & x2) | (x3 & x4)
        S.next = (x1 ^ x2) | ((x1 & x2) & (x3 & x4)) | (x3 ^ x4)
    return comb

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

def simulate_AC(verilog, input_ports=["x1","x2","x3","x4"], output_ports=["C","S"]):
    design = ys.Design()
    ys.run_pass(f"read_verilog {verilog}.v", design)
    ys.run_pass("aigmap", design)
    ys.run_pass(f"write_aiger {verilog}.aig", design)
    os.system(f"yosys -p 'read_aiger {verilog}.aig; synth; write_verilog {verilog}.v'")
    aig = aigverse.read_aiger_into_aig(f"{verilog}.aig")
    tts = aigverse.simulate(aig)

    bits_vec = [[int((i & (1<<j))>>j) for j in range(len(input_ports))] for i in range(1<< len(input_ports))]
    for i in range(1<< len(input_ports)):
        for j, tt in enumerate(tts):
            bits_vec[i].append(int(tt.get_bit(i)))
    print("Truth Table of", verilog)
    print_tt(bits_vec,input_ports,output_ports)

def simulate_AC(verilog, input_ports=["x1","x2","x3","x4"], output_ports=["C","S"]):
    # design = ys.Design()
    module = verilog.split("/")[-1]
    # ys.run_pass(f"read_verilog {verilog}.v", design)
    # ys.run_pass("aigmap", design)
    # ys.run_pass(f"write_aiger {verilog}.aig", design)
    aig = aigverse.read_pla_into_aig(f"{verilog}.pla")
    aigverse.write_aiger(aig, f"{verilog}.aig")
    os.system(f"yosys -p 'read_aiger {verilog}.aig; rename -top {module}; add -input clk; synth; write_verilog {verilog}.v' 1>/dev/null")
    # aigverse.write_verilog(aig, f"{verilog}.aig")
    tts = aigverse.simulate(aig)

    bits_vec = [[int((i & (1<<j))>>j) for j in range(len(input_ports))] for i in range(1<< len(input_ports))]
    for i in range(1<< len(input_ports)):
        for j, tt in enumerate(tts):
            bits_vec[i].append(int(tt.get_bit(i)))
    print("Truth Table of", verilog)
    print_tt(bits_vec,input_ports,output_ports)

# convert_AC("verilog",AC42_uw2)
simulate_AC("./ACs/AC42_uw1")
simulate_AC("./ACs/AC42_uw2")
simulate_AC("./ACs/AC42_uw3")
simulate_AC("./ACs/AC42_uw4")
simulate_AC("./ACs/AC42_uw5")
simulate_AC("./ACs/AC42_uw6")
simulate_AC("./ACs/AC42_uw7")
simulate_AC("./ACs/AC42_uw8")
simulate_AC("./ACs/AC42_uw9")
simulate_AC("./ACs/AC42_uw10")
simulate_AC("./ACs/AC42_uw11")
simulate_AC("./ACs/AC42_uw12")
simulate_AC("./ACs/AC42_ew1",input_ports=["x1","x2","x3","x4"], output_ports=["w2","w1"])
simulate_AC("./ACs/AC32_ew1",input_ports=["x1","x2","x3"], output_ports=["w2","w1"])
simulate_AC("./ACs/EC22",input_ports=["x1","x2"], output_ports=["C","S"])
simulate_AC("./ACs/EC32",input_ports=["x1","x2","x3"], output_ports=["C","S"])
