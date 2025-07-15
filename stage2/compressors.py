import torch
def ex22_fn(p0, p1):
    S = p0 + p1 - 2 * p0 * p1
    C = p0 * p1
    err = torch.zeros_like(p0)
    return (S, C), err

def ex32_fn(p0, p1, p2):
    S = (p0 + p1 + p2) - 2 * (p0*p1 + p0*p2 + p1*p2) + 4 * p0*p1*p2
    C = p0*p1 + p0*p2 + p1*p2 - 2 * p0*p1*p2
    err = torch.zeros_like(p0)
    return (S, C), err

def ap32_fn(p0, p1, p2):
    S = p0 + p1 - p0*p1
    C = p2 + p0*p1 - p0*p1*p2
    err = p0 * p1 * p2
    return (S, C), err

def ap42_fn(p0, p1, p2, p3):
    S = (p1 + p0 + p3*p2 + 2*p3*p0 + p1*p0
         - 2*p3*p1*p0 - p3*p2*p1 - p3*p2*p0 + p3*p2*p1*p0)
    C = (p2 + p3 + p1*p0 - p3*p2 - p2*p1*p0
         - p3*p1*p0 + p3*p2*p1*p0)
    err = 4 * p0*p1*p2 - 2 * (p0*p1*p2*p3)
    return (S, C), err

def dummy_fn(p0):
    return (p0, torch.tensor(0.0, device=p0.device)), torch.tensor(0.0, device=p0.device)

F_COMPRESSOR = [
    lambda inputs: ex32_fn(*inputs),
    lambda inputs: ex22_fn(*inputs),
    lambda inputs: ap32_fn(*inputs),
    lambda inputs: ap42_fn(*inputs),
    lambda inputs: dummy_fn(*inputs),
]
