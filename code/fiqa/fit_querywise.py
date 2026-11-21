"""Independent per-query Adam updates, vectorized across queries/arms."""
import torch
import torch.nn.functional as F

def objective(x, z, rewards, v):
    target_log = F.log_softmax(rewards / .02, dim=-1)
    target = target_log.exp()
    student_log = F.log_softmax(torch.einsum('bd,bkd->bk', F.normalize(x + v, dim=-1), z) / .05, dim=-1)
    return ((target * (target_log - student_log)).sum(dim=-1) + .3 * v.square().sum(dim=-1)).mean()

def fit(x, z, rewards, steps=300):
    if x.ndim != 2 or z.ndim != 3 or rewards.shape != z.shape[:2] or x.shape != (z.shape[0], z.shape[2]):
        raise ValueError('incompatible query, feedback and document shapes')
    if not all(torch.isfinite(a).all() for a in (x,z,rewards)):
        raise ValueError('nonfinite query, feedback or document')
    v = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(x,z,rewards,v)
        # Restore each row's independent gradient before Adam's epsilon is applied.
        (loss * x.shape[0]).backward()
        optimizer.step()
    if not torch.isfinite(v).all(): raise ValueError('nonfinite adapted vectors')
    return v.detach()
