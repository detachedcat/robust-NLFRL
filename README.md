# robust-NLFRL

[Project Page](https://detachedcat.github.io/robust-NLFRL/) | [中文说明](README_zh.md)

Simulation code for **certifiably robust legged locomotion** on Unitree G1: probabilistic neural Lyapunov functions integrated into an Actor–Critic RL pipeline (AMP + `LyaPPO`).

![Framework overview](docs/assets/framework.png)

## Fork & Scope

This project extends the open-source repository **[ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab)** (mjlab + rsl_rl AMP stack for Unitree G1). We thank the upstream authors for releasing their codebase.

| | Upstream (`ccrpRepo/AMP_mjlab`) | This repo (`robust-NLFRL`) |
|---|---|---|
| Focus | AMP-based G1 locomotion infrastructure | **Probabilistic neural Lyapunov + RL** for robust velocity tracking |
| Main task | `Unitree-G1-AMP-Flat` / `Rough` | **`Unitree-G1-LYA-Flat`** |
| Added module | — | TCLF co-training, probabilistic stability regularization, certified RoA |

**Note:** The upstream repo may include broader locomotion features. **This release focuses on robust velocity-tracking locomotion**, not fall-and-get-up recovery.

Deployment integration from the upstream project remains in [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm) (`MJAmp State`).

## Highlights

- Probabilistic Lyapunov conditions under sub-Gaussian uncertainty (dynamics + state estimation)
- Twin Control Lyapunov Function (`TCLF`) co-trained with policy via `LyaPPO`
- State-aware gating between task reward and Lyapunov robustness penalty
- Built on mjlab + vendored `rsl_rl`; ONNX export supported in train/play workflows

## Requirements

- Linux
- Python 3.11 (recommended)
- MuJoCo-compatible environment with GPU support

```bash
pip install "warp-lang>=1.12.0,<1.13"
```

## Quick Start

### 1. Install

```bash
conda activate mjlab
cd robust-NLFRL
python -m pip install -e .
pip install -e ./rsl_rl
```

References: [mjlab](https://github.com/mujocolab/mjlab), [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab).

### 2. Apply mjlab patch (optional)

If you skip this patch, remove `history_ordering` from observation configs.

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  $(python -c "import mjlab, os; print(os.path.dirname(mjlab.__file__))")/managers/observation_manager.py
```

### 3. List tasks

```bash
python scripts/list_envs.py --keyword LYA
```

| Task ID | Description |
|---|---|
| `Unitree-G1-LYA-Flat` | **Ours** — AMP + probabilistic neural Lyapunov (flat terrain) |
| `Unitree-G1-AMP-Flat` | AMP baseline (flat) |
| `Unitree-G1-Flat` | PPO velocity tracking baseline (no AMP) |

## Training

```bash
python scripts/train.py Unitree-G1-LYA-Flat --env.scene.num-envs=4096
```

Logs: `logs/rsl_rl/g1_lya_locomotion/<timestamp_run>/`

See [scripts/README_trainArgs.md](scripts/README_trainArgs.md) for full CLI options.

## Evaluation

```bash
python scripts/play.py Unitree-G1-LYA-Flat \
  --checkpoint-file logs/rsl_rl/g1_lya_locomotion/<run_dir>/model_<iter>.pt
```

## Motion Data

AMP/LYA tasks require motion NPZ files under `src/assets/motions/g1/amp/WalkandRun`.

```bash
python scripts/csv_to_npz.py --help
```

## Repository Structure

- `src/tasks/amp_lya/` — Lyapunov-augmented AMP task (`Unitree-G1-LYA-Flat`)
- `src/tasks/amp_loco/` — AMP locomotion environment (shared with upstream)
- `rsl_rl/algorithms/lya_ppo.py` — `LyaPPO` with TCLF co-training
- `rsl_rl/algorithms/neural_lyapunov/` — Twin Control Lyapunov Function
- `docs/` — project page assets (GitHub Pages)

## Acknowledgements

- [ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) — base G1 AMP mjlab codebase this work extends
- [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) — AMP implementation reference in `rsl_rl`
