# Auto-generated MyHDL compressor blocks
# Ports (names/order) come from ppa.json; behavior from ac_lib.json

from myhdl import block, always_comb, Signal, intbv

@block
def AC32_ew1(w2, w1, x1, x2, x3):
    """Auto-generated compressor 'AC32_ew1' (width=3).
    Ports are named exactly as in ppa.json. Pin mapping:
      idx = Σ inputs[i] << i  (inputs[0] is LSB/pin0).
    """
    # sanity checks (optional at sim time)
    # assert all(isinstance(x, Signal) for x in [x1, x2, x3])

    @always_comb
    def logic():
        w2.next = 0
        w1.next = 0
        x = 0
        x = x | ((int(x1) & 1) << 0)
        x = x | ((int(x2) & 1) << 1)
        x = x | ((int(x3) & 1) << 2)
        if x == 0:
            w2.next = 0
            w1.next = 0
        elif x == 1:
            w2.next = 0
            w1.next = 1
        elif x == 2:
            w2.next = 0
            w1.next = 1
        elif x == 3:
            w2.next = 1
            w1.next = 1
        elif x == 4:
            w2.next = 1
            w1.next = 0
        elif x == 5:
            w2.next = 1
            w1.next = 1
        elif x == 6:
            w2.next = 1
            w1.next = 1
        elif x == 7:
            w2.next = 1
            w1.next = 1
    return logic

from myhdl import block, always_comb, Signal, intbv

@block
def AC42_ew1(w2, w1, x1, x2, x3, x4):
    """Auto-generated compressor 'AC42_ew1' (width=4).
    Ports are named exactly as in ppa.json. Pin mapping:
      idx = Σ inputs[i] << i  (inputs[0] is LSB/pin0).
    """
    # sanity checks (optional at sim time)
    # assert all(isinstance(x, Signal) for x in [x1, x2, x3, x4])

    @always_comb
    def logic():
        w2.next = 0
        w1.next = 0
        x = 0
        x = x | ((int(x1) & 1) << 0)
        x = x | ((int(x2) & 1) << 1)
        x = x | ((int(x3) & 1) << 2)
        x = x | ((int(x4) & 1) << 3)
        if x == 0:
            w2.next = 0
            w1.next = 0
        elif x == 1:
            w2.next = 0
            w1.next = 1
        elif x == 2:
            w2.next = 0
            w1.next = 1
        elif x == 3:
            w2.next = 1
            w1.next = 1
        elif x == 4:
            w2.next = 1
            w1.next = 0
        elif x == 5:
            w2.next = 1
            w1.next = 1
        elif x == 6:
            w2.next = 1
            w1.next = 1
        elif x == 7:
            w2.next = 1
            w1.next = 1
        elif x == 8:
            w2.next = 1
            w1.next = 0
        elif x == 9:
            w2.next = 1
            w1.next = 1
        elif x == 10:
            w2.next = 1
            w1.next = 1
        elif x == 11:
            w2.next = 1
            w1.next = 1
        elif x == 12:
            w2.next = 1
            w1.next = 1
        elif x == 13:
            w2.next = 1
            w1.next = 1
        elif x == 14:
            w2.next = 1
            w1.next = 1
        elif x == 15:
            w2.next = 1
            w1.next = 1
    return logic

from myhdl import block, always_comb, Signal, intbv

@block
def EC22(C, S, x1, x2):
    """Auto-generated compressor 'EC22' (width=2).
    Ports are named exactly as in ppa.json. Pin mapping:
      idx = Σ inputs[i] << i  (inputs[0] is LSB/pin0).
    """
    # sanity checks (optional at sim time)
    # assert all(isinstance(x, Signal) for x in [x1, x2])

    @always_comb
    def logic():
        C.next = 0
        S.next = 0
        x = 0
        x = x | ((int(x1) & 1) << 0)
        x = x | ((int(x2) & 1) << 1)
        if x == 0:
            C.next = 0
            S.next = 0
        elif x == 1:
            C.next = 0
            S.next = 1
        elif x == 2:
            C.next = 0
            S.next = 1
        elif x == 3:
            C.next = 1
            S.next = 0
    return logic

from myhdl import block, always_comb, Signal, intbv

@block
def EC32(C, S, x1, x2, x3):
    """Auto-generated compressor 'EC32' (width=3).
    Ports are named exactly as in ppa.json. Pin mapping:
      idx = Σ inputs[i] << i  (inputs[0] is LSB/pin0).
    """
    # sanity checks (optional at sim time)
    # assert all(isinstance(x, Signal) for x in [x1, x2, x3])

    @always_comb
    def logic():
        C.next = 0
        S.next = 0
        x = 0
        x = x | ((int(x1) & 1) << 0)
        x = x | ((int(x2) & 1) << 1)
        x = x | ((int(x3) & 1) << 2)
        if x == 0:
            C.next = 0
            S.next = 0
        elif x == 1:
            C.next = 0
            S.next = 1
        elif x == 2:
            C.next = 0
            S.next = 1
        elif x == 3:
            C.next = 1
            S.next = 0
        elif x == 4:
            C.next = 0
            S.next = 1
        elif x == 5:
            C.next = 1
            S.next = 0
        elif x == 6:
            C.next = 1
            S.next = 0
        elif x == 7:
            C.next = 1
            S.next = 1
    return logic
