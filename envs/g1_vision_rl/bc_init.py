from __future__ import annotations

from typing import Dict

import torch


def load_bc_actor_weights(actor_critic, bc_checkpoint_path: str, device: torch.device) -> Dict[str, int]:
    """
    Loads BC MLP weights into PPO actor when layer shapes match.

    BC key format (HighLevelMLP):  net.<idx>.weight / net.<idx>.bias
    PPO actor format (ActorCritic): actor.<idx>.weight / actor.<idx>.bias

    Returns load summary dict.
    """
    try:
        ckpt = torch.load(bc_checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(bc_checkpoint_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"BC checkpoint missing 'model_state_dict': {bc_checkpoint_path}")

    bc_sd = ckpt["model_state_dict"]
    actor_sd = actor_critic.actor.state_dict()

    mapped = {}
    loaded = 0
    skipped = 0

    for k, v in bc_sd.items():
        if not k.startswith("net."):
            continue
        kk = k[len("net.") :]
        if kk not in actor_sd:
            skipped += 1
            continue
        if tuple(actor_sd[kk].shape) != tuple(v.shape):
            skipped += 1
            continue
        mapped[kk] = v
        loaded += 1

    actor_critic.actor.load_state_dict(mapped, strict=False)

    return {
        "loaded": int(loaded),
        "skipped": int(skipped),
        "actor_params": int(len(actor_sd)),
    }
