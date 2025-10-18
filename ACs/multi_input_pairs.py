import torch

if torch.cuda.is_available():
    device = "cuda"

elif torch.backends.mps.is_available():
    device = "mps"

else:
    device = "cpu"

def AC32_ew1_S(p1, p2, p3):
    return p1*(1 - p2)*(1 - p3) + (1 - p1)*p2*(1 - p3) + p1*p2*(1 - p3) + p1*(1 - p2)*p3 + (1 - p1)*p2*p3 + p1*p2*p3

def AC32_ew1_C(p1, p2, p3):
    return p1*p2*(1 - p3) + (1 - p1)*(1 - p2)*p3 + p1*(1 - p2)*p3 + (1 - p1)*p2*p3 + p1*p2*p3

def AC32_ew1_err(p1, p2, p3):
    return (-1)*p1*p2*(1 - p3) + (-1)*(1 - p1)*(1 - p2)*p3 + (-1)*p1*(1 - p2)*p3 + (-1)*(1 - p1)*p2*p3

def AC42_ew1_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + p1*p2*(1 - p3)*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_ew1_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_ew1_err(p1, p2, p3, p4):
    return (-1)*p1*p2*(1 - p3)*(1 - p4) + (-1)*(1 - p1)*(1 - p2)*p3*(1 - p4) + (-1)*p1*(1 - p2)*p3*(1 - p4) + (-1)*(1 - p1)*p2*p3*(1 - p4) + (-1)*(1 - p1)*(1 - p2)*(1 - p3)*p4 + (-1)*p1*(1 - p2)*(1 - p3)*p4 + (-1)*(1 - p1)*p2*(1 - p3)*p4 + (-1)*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw1_S(p1, p2, p3, p4):
    return (1 - p1)*(1 - p2)*(1 - p3)*(1 - p4) + p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + p1*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw1_C(p1, p2, p3, p4):
    return p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw1_err(p1, p2, p3, p4):
    return (-1)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - p4) + 1*p1*p2*(1 - p3)*(1 - p4) + 1*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw10_S(p1, p2, p3, p4):
    return (1 - p1)*(1 - p2)*(1 - p3)*(1 - p4) + p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + p1*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw10_C(p1, p2, p3, p4):
    return p1*(1 - p2)*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw10_err(p1, p2, p3, p4):
    return (-1)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - p4) + 1*p1*p2*(1 - p3)*(1 - p4) + (-1)*p1*(1 - p2)*p3*(1 - p4) + 1*(1 - p1)*p2*p3*(1 - p4) + (-1)*p1*(1 - p2)*(1 - p3)*p4 + 1*(1 - p1)*p2*(1 - p3)*p4 + (-1)*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw11_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + p1*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw11_C(p1, p2, p3, p4):
    return p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw11_err(p1, p2, p3, p4):
    return 1*p1*p2*(1 - p3)*(1 - p4) + (-1)*p1*(1 - p2)*p3*(1 - p4) + (-1)*(1 - p1)*p2*p3*(1 - p4) + (-1)*p1*(1 - p2)*(1 - p3)*p4 + (-1)*(1 - p1)*p2*(1 - p3)*p4 + 1*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw12_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw12_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw12_err(p1, p2, p3, p4):
    return 1*p1*p2*p3*(1 - p4) + 1*p1*p2*p3*p4

def AC42_uw2_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw2_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + p1*p2*p3*(1 - p4) + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw2_err(p1, p2, p3, p4):
    return 1*p1*(1 - p2)*p3*(1 - p4) + 1*(1 - p1)*p2*p3*(1 - p4) + 1*p1*(1 - p2)*(1 - p3)*p4 + 1*(1 - p1)*p2*(1 - p3)*p4 + 1*p1*p2*p3*p4

def AC42_uw3_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw3_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw3_err(p1, p2, p3, p4):
    return 1*p1*p2*p3*p4

def AC42_uw4_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw4_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw4_err(p1, p2, p3, p4):
    return (-1)*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw5_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*p2*p3*p4

def AC42_uw5_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw5_err(p1, p2, p3, p4):
    return (-1)*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*(1 - p2)*p3*p4 + 1*(1 - p1)*p2*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw6_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4

def AC42_uw6_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw6_err(p1, p2, p3, p4):
    return 2*p1*p2*p3*p4

def AC42_uw7_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*p2*p3*p4

def AC42_uw7_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw7_err(p1, p2, p3, p4):
    return 1*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*(1 - p2)*p3*p4 + 1*(1 - p1)*p2*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw8_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4

def AC42_uw8_C(p1, p2, p3, p4):
    return (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw8_err(p1, p2, p3, p4):
    return 2*p1*p2*(1 - p3)*(1 - p4) + 1*p1*(1 - p2)*p3*(1 - p4) + 1*(1 - p1)*p2*p3*(1 - p4) + 2*p1*p2*p3*(1 - p4) + (-2)*(1 - p1)*(1 - p2)*(1 - p3)*p4 + (-1)*p1*(1 - p2)*(1 - p3)*p4 + (-1)*(1 - p1)*p2*(1 - p3)*p4 + 2*p1*p2*p3*p4

def AC42_uw9_S(p1, p2, p3, p4):
    return p1*(1 - p2)*(1 - p3)*(1 - p4) + (1 - p1)*p2*(1 - p3)*(1 - p4) + (1 - p1)*(1 - p2)*p3*(1 - p4) + p1*(1 - p2)*p3*(1 - p4) + (1 - p1)*p2*p3*(1 - p4) + p1*p2*p3*(1 - p4) + (1 - p1)*(1 - p2)*(1 - p3)*p4 + p1*(1 - p2)*(1 - p3)*p4 + (1 - p1)*p2*(1 - p3)*p4 + p1*p2*(1 - p3)*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4

def AC42_uw9_C(p1, p2, p3, p4):
    return p1*p2*(1 - p3)*(1 - p4) + p1*p2*p3*(1 - p4) + p1*p2*(1 - p3)*p4 + (1 - p1)*(1 - p2)*p3*p4 + p1*(1 - p2)*p3*p4 + (1 - p1)*p2*p3*p4 + p1*p2*p3*p4

def AC42_uw9_err(p1, p2, p3, p4):
    return 1*p1*(1 - p2)*p3*(1 - p4) + 1*(1 - p1)*p2*p3*(1 - p4) + 1*p1*(1 - p2)*(1 - p3)*p4 + 1*(1 - p1)*p2*(1 - p3)*p4 + 2*p1*p2*p3*p4

def EC22_S(p1, p2):
    return p1*(1 - p2) + (1 - p1)*p2

def EC22_C(p1, p2):
    return p1*p2

def EC22_err(p1, p2):
    return torch.zeros_like(p1, device=device)

def EC32_S(p1, p2, p3):
    return p1*(1 - p2)*(1 - p3) + (1 - p1)*p2*(1 - p3) + (1 - p1)*(1 - p2)*p3 + p1*p2*p3

def EC32_C(p1, p2, p3):
    return p1*p2*(1 - p3) + p1*(1 - p2)*p3 + (1 - p1)*p2*p3 + p1*p2*p3

def EC32_err(p1, p2, p3):
    return torch.zeros_like(p1, device=device)

F_S_LIST = [AC32_ew1_S, AC42_ew1_S, AC42_uw1_S, AC42_uw10_S, AC42_uw11_S, AC42_uw12_S, AC42_uw2_S, AC42_uw3_S, AC42_uw4_S, AC42_uw5_S, AC42_uw6_S, AC42_uw7_S, AC42_uw8_S, AC42_uw9_S, EC22_S, EC32_S, lambda p1: p1]
F_C_LIST = [AC32_ew1_C, AC42_ew1_C, AC42_uw1_C, AC42_uw10_C, AC42_uw11_C, AC42_uw12_C, AC42_uw2_C, AC42_uw3_C, AC42_uw4_C, AC42_uw5_C, AC42_uw6_C, AC42_uw7_C, AC42_uw8_C, AC42_uw9_C, EC22_C, EC32_C, lambda p1: 0*p1]
F_ERR_LIST = [AC32_ew1_err, AC42_ew1_err, AC42_uw1_err, AC42_uw10_err, AC42_uw11_err, AC42_uw12_err, AC42_uw2_err, AC42_uw3_err, AC42_uw4_err, AC42_uw5_err, AC42_uw6_err, AC42_uw7_err, AC42_uw8_err, AC42_uw9_err, EC22_err, EC32_err, lambda p1: 0*p1]
