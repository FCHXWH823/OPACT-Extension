import torch
def sinkhorn_log(log_alpha, n_iters=20):
    """
    log_alpha: [B, N, N] or [N, N]
    return: doubly stochastic matrix [B, N, N]
    """
    if log_alpha.dim() == 2:
        log_alpha = log_alpha.unsqueeze(0)

    for _ in range(n_iters):
        # normalize rows
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        # normalize cols
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

    return torch.exp(log_alpha)

def gumbel_sinkhorn(
    logits,
    tau=1.0,
    noise=True,
    noise_factor=0.1,
    n_iters=20
):
    """
    logits: [B, N, N] or [N, N]
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)

    if noise:
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(logits) + 1e-9) + 1e-9
        )
        logits = logits + noise_factor * gumbel_noise

    log_alpha = logits / tau
    P = sinkhorn_log(log_alpha, n_iters)
    return P
