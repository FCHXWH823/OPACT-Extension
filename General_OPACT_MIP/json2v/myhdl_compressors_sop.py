# Auto-generated MyHDL compressors (SOP form)
# Ports (names/order) from ppa.json; behavior from ac_lib.json
from myhdl import block, always_comb, Signal, intbv
import os

@block
def AC32_ew1(w2, w1, x1, x2, x3):
    """Compressor 'AC32_ew1' (width=3) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        w2.next = ((x1 & x2 & ~x3) | (~x1 & ~x2 & x3) | (x1 & ~x2 & x3) | (~x1 & x2 & x3) | (x1 & x2 & x3))
        w1.next = ((x1 & ~x2 & ~x3) | (~x1 & x2 & ~x3) | (x1 & x2 & ~x3) | (x1 & ~x2 & x3) | (~x1 & x2 & x3) | (x1 & x2 & x3))
    return logic



@block
def AC42_ew1(w2, w1, x1, x2, x3, x4):
    """Compressor 'AC42_ew1' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        w2.next = ((x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        w1.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (x1 & x2 & ~x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw1(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw1' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((~x1 & ~x2 & ~x3 & ~x4) | (x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw10(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw10' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & ~x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((~x1 & ~x2 & ~x3 & ~x4) | (x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw11(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw11' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw12(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw12' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw2(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw2' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw3(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw3' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw4(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw4' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw5(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw5' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw6(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw6' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw7(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw7' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw8(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw8' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((~x1 & ~x2 & ~x3 & x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4))
    return logic



@block
def AC42_uw9(C, S, x1, x2, x3, x4):
    """Compressor 'AC42_uw9' (width=4) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (x1 & x2 & ~x3 & x4) | (~x1 & ~x2 & x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4) | (x1 & x2 & x3 & x4))
        S.next = ((x1 & ~x2 & ~x3 & ~x4) | (~x1 & x2 & ~x3 & ~x4) | (~x1 & ~x2 & x3 & ~x4) | (x1 & ~x2 & x3 & ~x4) | (~x1 & x2 & x3 & ~x4) | (x1 & x2 & x3 & ~x4) | (~x1 & ~x2 & ~x3 & x4) | (x1 & ~x2 & ~x3 & x4) | (~x1 & x2 & ~x3 & x4) | (x1 & x2 & ~x3 & x4) | (x1 & ~x2 & x3 & x4) | (~x1 & x2 & x3 & x4))
    return logic



@block
def EC22(C, S, x1, x2):
    """Compressor 'EC22' (width=2) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2))
        S.next = ((x1 & ~x2) | (~x1 & x2))
    return logic



@block
def EC32(C, S, x1, x2, x3):
    """Compressor 'EC32' (width=3) — SOP form.
    idx mapping: inputs[0] is LSB (pin0) for truth tables.
    """
    @always_comb
    def logic():
        C.next = ((x1 & x2 & ~x3) | (x1 & ~x2 & x3) | (~x1 & x2 & x3) | (x1 & x2 & x3))
        S.next = ((x1 & ~x2 & ~x3) | (~x1 & x2 & ~x3) | (~x1 & ~x2 & x3) | (x1 & x2 & x3))
    return logic


