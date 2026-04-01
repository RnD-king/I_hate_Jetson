# G1 Vision Script Guide

## Active Pipeline Scripts

- `g1_pid_module.py`
  - Teacher/perception/controller module.
- `g1_pid_play.py`
  - Teacher rollout debug player.
- `g1_pid_collect_dataset.py`
  - Teacher dataset collection.
- `train_g1_highlevel_bc.py`
  - Behavior cloning trainer.
- `g1_highlevel_bc_play.py`
  - BC inference player with frozen low-level walking policy.

## Other Core Scripts

- `train.py`
- `play.py`
- `export_g1_jit.py`

## Legacy Scripts

Older heuristic/experimental scripts were moved to:

- `legged_gym/scripts/legacy/`

They are kept for reference but are not part of the current G1 vision BC pipeline.
