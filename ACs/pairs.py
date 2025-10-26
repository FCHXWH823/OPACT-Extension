import torch

if torch.cuda.is_available():
    device = "cuda"

elif torch.backends.mps.is_available():
    device = "mps"

else:
    device = "cpu"

def AC32_ew1_S(p):
    return (-p**2 + 2*p)

def AC32_ew1_C(p):
    return (-p**3 + p**2 + p)

def AC32_ew1_err(p):
    return (p**3)

def AC42_ew1_S(p):
    return (p**4 - 2*p**3 + 2*p)

def AC42_ew1_C(p):
    return (p**4 - 2*p**3 + 2*p)

def AC42_ew1_err(p):
    return (-2*p**4 + 4*p**3)

def AC42_uw1_S(p):
    return (-4*p**4 + 8*p**3 - 4*p**2 + 1)

def AC42_uw1_C(p):
    return (p**4 - 4*p**3 + 4*p**2)

def AC42_uw1_err(p):
    return (2*p**4 - 4*p**2 + 4*p - 1)

def AC42_uw10_S(p):
    return torch.ones_like(p, device = device)

def AC42_uw10_C(p):
    return (-2*p**3 + 3*p**2)

def AC42_uw10_err(p):
    return (4*p**3 - 6*p**2 + 4*p - 1)

def AC42_uw11_S(p):
    return (-p**4 + 4*p**3 - 6*p**2 + 4*p)

def AC42_uw11_C(p):
    return (p**4 - 4*p**3 + 4*p**2)

def AC42_uw11_err(p):
    return (-p**4 + 4*p**3 - 2*p**2)

def AC42_uw12_S(p):
    return (-6*p**4 + 15*p**3 - 12*p**2 + 4*p)

def AC42_uw12_C(p):
    return (3*p**4 - 8*p**3 + 6*p**2)

def AC42_uw12_err(p):
    return (p**3)

def AC42_uw2_S(p):
    return (-3*p**4 + 8*p**3 - 8*p**2 + 4*p)

def AC42_uw2_C(p):
    return (-p**4 + 2*p**2)

def AC42_uw2_err(p):
    return (5*p**4 - 8*p**3 + 4*p**2)

def AC42_uw3_S(p):
    return (-7*p**4 + 16*p**3 - 12*p**2 + 4*p)

def AC42_uw3_C(p):
    return (3*p**4 - 8*p**3 + 6*p**2)

def AC42_uw3_err(p):
    return (p**4)

def AC42_uw4_S(p):
    return (-6*p**4 + 14*p**3 - 11*p**2 + 4*p)

def AC42_uw4_C(p):
    return (3*p**4 - 8*p**3 + 6*p**2)

def AC42_uw4_err(p):
    return (2*p**3 - p**2)

def AC42_uw5_S(p):
    return (-4*p**4 + 12*p**3 - 11*p**2 + 4*p)

def AC42_uw5_C(p):
    return (3*p**4 - 8*p**3 + 6*p**2)

def AC42_uw5_err(p):
    return (-2*p**4 + 4*p**3 - p**2)

def AC42_uw6_S(p):
    return (-8*p**4 + 16*p**3 - 12*p**2 + 4*p)

def AC42_uw6_C(p):
    return (3*p**4 - 8*p**3 + 6*p**2)

def AC42_uw6_err(p):
    return (2*p**4)

def AC42_uw7_S(p):
    return (-4*p**4 + 12*p**3 - 11*p**2 + 4*p)

def AC42_uw7_C(p):
    return (2*p**4 - 6*p**3 + 5*p**2)

def AC42_uw7_err(p):
    return (p**2)

def AC42_uw8_S(p):
    return (-4*p**4 + 8*p**3 - 8*p**2 + 4*p)

def AC42_uw8_C(p):
    return (p)

def AC42_uw8_err(p):
    return (4*p**4 - 8*p**3 + 8*p**2 - 2*p)

def AC42_uw9_S(p):
    return (-4*p**4 + 8*p**3 - 8*p**2 + 4*p)

def AC42_uw9_C(p):
    return (-p**4 + 2*p**2)

def AC42_uw9_err(p):
    return (6*p**4 - 8*p**3 + 4*p**2)

def EC22_S(p):
    return (-2*p**2 + 2*p)

def EC22_C(p):
    return (p**2)

def EC22_err(p):
    return torch.zeros_like(p, device = device)

def EC32_S(p):
    return (4*p**3 - 6*p**2 + 3*p)

def EC32_C(p):
    return (-2*p**3 + 3*p**2)

def EC32_err(p):
    return torch.zeros_like(p, device = device)

F_S_LIST = [AC32_ew1_S, AC42_ew1_S, AC42_uw1_S, AC42_uw10_S, AC42_uw11_S, AC42_uw12_S, AC42_uw2_S, AC42_uw3_S, AC42_uw4_S, AC42_uw5_S, AC42_uw6_S, AC42_uw7_S, AC42_uw8_S, AC42_uw9_S, EC22_S, EC32_S, lambda p: p]
F_C_LIST = [AC32_ew1_C, AC42_ew1_C, AC42_uw1_C, AC42_uw10_C, AC42_uw11_C, AC42_uw12_C, AC42_uw2_C, AC42_uw3_C, AC42_uw4_C, AC42_uw5_C, AC42_uw6_C, AC42_uw7_C, AC42_uw8_C, AC42_uw9_C, EC22_C, EC32_C, lambda p: 0*p]
F_ERR_LIST = [AC32_ew1_err, AC42_ew1_err, AC42_uw1_err, AC42_uw10_err, AC42_uw11_err, AC42_uw12_err, AC42_uw2_err, AC42_uw3_err, AC42_uw4_err, AC42_uw5_err, AC42_uw6_err, AC42_uw7_err, AC42_uw8_err, AC42_uw9_err, EC22_err, EC32_err, lambda p: 0*p]
