# AGENTS.md

## Repository Focus

This repository is a nano-drone system identification benchmark centered on 50-step open-loop rollout prediction for a Crazyflie-class platform. The main ablation work happens around Physics + Residual, lag, GRU, and force-aware variants. Preserve the official benchmark protocol and keep edits easy to isolate, compare, and ablate.

## Important Files

- `models/models.py`: core physics backbone and base residual / hybrid models.
- `models/models_lag.py`: lag, GRU, and force-oriented Physics + Residual variants.
- `train/train_physres_ablation.py`: main ablation training entry point and variant wiring.
- `dataset/`: dataset loading and trajectory-window construction.
- `train/losses.py`, `train/losses_ext.py`: benchmark loss definitions.
- `test/`: evaluation scripts; treat protocol changes as high risk.
- `data/`, `scalers/`: benchmark inputs and normalization artifacts; do not change casually.
- `run_full12.sh`: batch experiment runner for the full benchmark suite.

## Working Conventions

- Keep changes small, reviewable, and ablation-friendly.
- Do not change data files, scalers, evaluation protocol, or trajectory split logic unless explicitly asked.
- Do not rename existing variants, checkpoint-facing module names, or output conventions unless explicitly asked.
- Preserve checkpoint compatibility whenever possible.
- Physics should remain the backbone; neural modules should learn corrections rather than replace the dynamics model.
- New residual / force / torque heads should start from zero effect when possible so the physics path remains the default behavior at initialization.
- Avoid silent changes to rollout horizon, loss semantics, normalization, or state conventions.
- Be careful with state representation: model code uses a 12D state `[pos, vel, so3_log, omega]` internally even though the dataset/benchmark description often references quaternion outputs.
- Prefer targeted edits over broad refactors. If adding a new variant, keep wiring localized and comparable to existing ablations.

## Validation

Avoid full training runs during code-edit tasks unless explicitly asked. Prefer lightweight checks such as:

```bash
python -m py_compile models/models.py models/models_lag.py train/train_physres_ablation.py
```

Useful follow-up checks when relevant:

```bash
git diff --name-only
python -m pytest test -q
```
