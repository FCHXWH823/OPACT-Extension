import torch

if torch.cuda.is_available():
    device = "cuda"

elif torch.backends.mps.is_available():
    device = "mps"

else:
    device = "cpu"

def AC32_ew1_S(inputs):
    p1, p2, p3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    return -p1*p2 + p1 + p2

def AC32_ew1_C(inputs):
    p1, p2, p3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    return -p1*p2*p3 + p1*p2 + p3

def AC32_ew1_err(inputs):
    p1, p2, p3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    return p1*p2*p3

def AC42_ew1_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return p1*p2*p3*p4 - p1*p2 - p1*p3*p4 + p1 - p2*p3*p4 + p2 + p3*p4

def AC42_ew1_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return p1*p2*p3*p4 - p1*p2*p3 - p1*p2*p4 + p1*p2 - p3*p4 + p3 + p4

def AC42_ew1_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 1*p1*p2*p3*(1 - p4) + 1*p1*p2*(1 - p3)*p4 + 1*p1*(1 - p2)*p3*p4 + 1*(1 - p1)*p2*p3*p4 + 2*p1*p2*p3*p4

def AC42_uw1_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -4*p1*p2*p3*p4 + 2*p1*p2*p3 + 2*p1*p2*p4 + 2*p1*p3*p4 - p1*p3 - p1*p4 + 2*p2*p3*p4 - p2*p3 - p2*p4 + 1

def AC42_uw1_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return p1*p2*p3*p4 - p1*p2*p3 - p1*p2*p4 - p1*p3*p4 + p1*p3 + p1*p4 - p2*p3*p4 + p2*p3 + p2*p4

def AC42_uw1_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return (-1)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - p4) + 1*p1*p2*(1 - p3)*(1 - p4) + 1*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw10_S(inputs):
    return 1

def AC42_uw10_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -2*p1*p3*p4 + p1*p3 + p1*p4 + p3*p4

def AC42_uw10_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return (-1)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - p4) + 1*p1*p2*(1 - p3)*(1 - p4) + (-1)*p1*(1 - p2)*p3*(1 - p4) + 1*(1 - p1)*p2*p3*(1 - p4) + (-1)*p1*(1 - p2)*(1 - p3)*p4 + 1*(1 - p1)*p2*(1 - p3)*p4 + (-1)*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw11_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -p1*p2*p3*p4 + p1*p2*p3 + p1*p2*p4 - p1*p2 + p1*p3*p4 - p1*p3 - p1*p4 + p1 + p2*p3*p4 - p2*p3 - p2*p4 + p2 - p3*p4 + p3 + p4

def AC42_uw11_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return p1*p2*p3*p4 - p1*p2*p3 - p1*p2*p4 - p1*p3*p4 + p1*p3 + p1*p4 - p2*p3*p4 + p2*p3 + p2*p4

def AC42_uw11_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 1*p1*p2*(1 - p3)*(1 - p4) + (-1)*p1*(1 - p2)*p3*(1 - p4) + (-1)*(1 - p1)*p2*p3*(1 - p4) + (-1)*p1*(1 - p2)*(1 - p3)*p4 + (-1)*(1 - p1)*p2*(1 - p3)*p4 + 1*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw12_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -6*p1*p2*p3*p4 + 3*p1*p2*p3 + 4*p1*p2*p4 - 2*p1*p2 + 4*p1*p3*p4 - 2*p1*p3 - 2*p1*p4 + p1 + 4*p2*p3*p4 - 2*p2*p3 - 2*p2*p4 + p2 - 2*p3*p4 + p3 + p4

def AC42_uw12_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 3*p1*p2*p3*p4 - 2*p1*p2*p3 - 2*p1*p2*p4 + p1*p2 - 2*p1*p3*p4 + p1*p3 + p1*p4 - 2*p2*p3*p4 + p2*p3 + p2*p4 + p3*p4

def AC42_uw12_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 1*p1*p2*p3*(1 - p4) + 1*p1*p2*p3*p4

def AC42_uw2_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -3*p1*p2*p3*p4 + 2*p1*p2*p3 + 2*p1*p2*p4 - 2*p1*p2 + 2*p1*p3*p4 - p1*p3 - p1*p4 + p1 + 2*p2*p3*p4 - p2*p3 - p2*p4 + p2 - 2*p3*p4 + p3 + p4

def AC42_uw2_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -p1*p2*p3*p4 + p1*p2 + p3*p4

def AC42_uw2_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 1*p1*(1 - p2)*p3*(1 - p4) + 1*(1 - p1)*p2*p3*(1 - p4) + 1*p1*(1 - p2)*(1 - p3)*p4 + 1*(1 - p1)*p2*(1 - p3)*p4 + 1*p1*p2*p3*p4

def AC42_uw3_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -7*p1*p2*p3*p4 + 4*p1*p2*p3 + 4*p1*p2*p4 - 2*p1*p2 + 4*p1*p3*p4 - 2*p1*p3 - 2*p1*p4 + p1 + 4*p2*p3*p4 - 2*p2*p3 - 2*p2*p4 + p2 - 2*p3*p4 + p3 + p4

def AC42_uw3_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 3*p1*p2*p3*p4 - 2*p1*p2*p3 - 2*p1*p2*p4 + p1*p2 - 2*p1*p3*p4 + p1*p3 + p1*p4 - 2*p2*p3*p4 + p2*p3 + p2*p4 + p3*p4

def AC42_uw3_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 1*p1*p2*p3*p4

def AC42_uw4_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -6*p1*p2*p3*p4 + 4*p1*p2*p3 + 4*p1*p2*p4 - 2*p1*p2 + 3*p1*p3*p4 - 2*p1*p3 - 2*p1*p4 + p1 + 3*p2*p3*p4 - 2*p2*p3 - 2*p2*p4 + p2 - p3*p4 + p3 + p4

def AC42_uw4_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 3*p1*p2*p3*p4 - 2*p1*p2*p3 - 2*p1*p2*p4 + p1*p2 - 2*p1*p3*p4 + p1*p3 + p1*p4 - 2*p2*p3*p4 + p2*p3 + p2*p4 + p3*p4

def AC42_uw4_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return (-1)*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw5_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -4*p1*p2*p3*p4 + 4*p1*p2*p3 + 4*p1*p2*p4 - 2*p1*p2 + 2*p1*p3*p4 - 2*p1*p3 - 2*p1*p4 + p1 + 2*p2*p3*p4 - 2*p2*p3 - 2*p2*p4 + p2 - p3*p4 + p3 + p4

def AC42_uw5_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 3*p1*p2*p3*p4 - 2*p1*p2*p3 - 2*p1*p2*p4 + p1*p2 - 2*p1*p3*p4 + p1*p3 + p1*p4 - 2*p2*p3*p4 + p2*p3 + p2*p4 + p3*p4

def AC42_uw5_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return (-1)*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*(1 - p2)*p3*p4 + 1*(1 - p1)*p2*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw6_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -8*p1*p2*p3*p4 + 4*p1*p2*p3 + 4*p1*p2*p4 - 2*p1*p2 + 4*p1*p3*p4 - 2*p1*p3 - 2*p1*p4 + p1 + 4*p2*p3*p4 - 2*p2*p3 - 2*p2*p4 + p2 - 2*p3*p4 + p3 + p4

def AC42_uw6_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 3*p1*p2*p3*p4 - 2*p1*p2*p3 - 2*p1*p2*p4 + p1*p2 - 2*p1*p3*p4 + p1*p3 + p1*p4 - 2*p2*p3*p4 + p2*p3 + p2*p4 + p3*p4

def AC42_uw6_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 2*p1*p2*p3*p4

def AC42_uw7_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -4*p1*p2*p3*p4 + 4*p1*p2*p3 + 4*p1*p2*p4 - 2*p1*p2 + 2*p1*p3*p4 - 2*p1*p3 - 2*p1*p4 + p1 + 2*p2*p3*p4 - 2*p2*p3 - 2*p2*p4 + p2 - p3*p4 + p3 + p4

def AC42_uw7_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 2*p1*p2*p3*p4 - 2*p1*p2*p3 - 2*p1*p2*p4 + p1*p2 - p1*p3*p4 + p1*p3 + p1*p4 - p2*p3*p4 + p2*p3 + p2*p4

def AC42_uw7_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 1*(1 - p1)*(1 - p2)*p3*p4 + 1*p1*(1 - p2)*p3*p4 + 1*(1 - p1)*p2*p3*p4 + 1*p1*p2*p3*p4

def AC42_uw8_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -4*p1*p2*p3*p4 + 2*p1*p2*p3 + 2*p1*p2*p4 - 2*p1*p2 + 2*p1*p3*p4 - p1*p3 - p1*p4 + p1 + 2*p2*p3*p4 - p2*p3 - p2*p4 + p2 - 2*p3*p4 + p3 + p4

def AC42_uw8_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return p4

def AC42_uw8_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 2*p1*p2*(1 - p3)*(1 - p4) + 1*p1*(1 - p2)*p3*(1 - p4) + 1*(1 - p1)*p2*p3*(1 - p4) + 2*p1*p2*p3*(1 - p4) + (-2)*(1 - p1)*(1 - p2)*(1 - p3)*p4 + (-1)*p1*(1 - p2)*(1 - p3)*p4 + (-1)*(1 - p1)*p2*(1 - p3)*p4 + 2*p1*p2*p3*p4

def AC42_uw9_S(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -4*p1*p2*p3*p4 + 2*p1*p2*p3 + 2*p1*p2*p4 - 2*p1*p2 + 2*p1*p3*p4 - p1*p3 - p1*p4 + p1 + 2*p2*p3*p4 - p2*p3 - p2*p4 + p2 - 2*p3*p4 + p3 + p4

def AC42_uw9_C(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return -p1*p2*p3*p4 + p1*p2 + p3*p4

def AC42_uw9_err(inputs):
    p1, p2, p3, p4 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    return 1*p1*(1 - p2)*p3*(1 - p4) + 1*(1 - p1)*p2*p3*(1 - p4) + 1*p1*(1 - p2)*(1 - p3)*p4 + 1*(1 - p1)*p2*(1 - p3)*p4 + 2*p1*p2*p3*p4

def EC22_S(inputs):
    p1, p2 = inputs[:, 0], inputs[:, 1]
    return -2*p1*p2 + p1 + p2

def EC22_C(inputs):
    p1, p2 = inputs[:, 0], inputs[:, 1]
    return p1*p2

def EC22_err(inputs):
    p1, _ = inputs[:, 0], inputs[:, 1]
    return torch.zeros_like(p1)

def EC32_S(inputs):
    p1, p2, p3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    return 4*p1*p2*p3 - 2*p1*p2 - 2*p1*p3 + p1 - 2*p2*p3 + p2 + p3

def EC32_C(inputs):
    p1, p2, p3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    return -2*p1*p2*p3 + p1*p2 + p1*p3 + p2*p3

def EC32_err(inputs):
    p1, _, _ = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    return torch.zeros_like(p1)

F_S_LIST = [AC32_ew1_S, AC42_ew1_S, AC42_uw1_S, AC42_uw10_S, AC42_uw11_S, AC42_uw12_S, AC42_uw2_S, AC42_uw3_S, AC42_uw4_S, AC42_uw5_S, AC42_uw6_S, AC42_uw7_S, AC42_uw8_S, AC42_uw9_S, EC22_S, EC32_S, lambda p1: p1.squeeze(1)]
F_C_LIST = [AC32_ew1_C, AC42_ew1_C, AC42_uw1_C, AC42_uw10_C, AC42_uw11_C, AC42_uw12_C, AC42_uw2_C, AC42_uw3_C, AC42_uw4_C, AC42_uw5_C, AC42_uw6_C, AC42_uw7_C, AC42_uw8_C, AC42_uw9_C, EC22_C, EC32_C, lambda p1: 0*p1]
F_ERR_LIST = [AC32_ew1_err, AC42_ew1_err, AC42_uw1_err, AC42_uw10_err, AC42_uw11_err, AC42_uw12_err, AC42_uw2_err, AC42_uw3_err, AC42_uw4_err, AC42_uw5_err, AC42_uw6_err, AC42_uw7_err, AC42_uw8_err, AC42_uw9_err, EC22_err, EC32_err, lambda p1: 0*p1]
