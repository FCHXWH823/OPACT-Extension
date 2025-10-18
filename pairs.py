import numpy as np

def AC32_ew1_S(p):
    return (-p0*p1 + p0 + p1)

def AC32_ew1_C(p):
    return (-p0*p1*p2 + p0*p1 + p2)

F_S_LIST = [AC32_ew1_S, lambda p: p]
F_C_LIST = [AC32_ew1_C, lambda p: 0*p]
