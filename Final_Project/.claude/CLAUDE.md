# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Course project for "Frontiers of Geometric Computation" (PKU). Testing whether JiT's manifold hypothesis — that x-prediction is superior to ε/v-prediction when ambient dimension ≫ intrinsic dimension — holds for 3D shape generation.

**Plan:** Compare x/ε/v prediction targets under a flow matching framework on ShapeNet point clouds, using a deliberately lightweight backbone to amplify the effect of prediction target choice.

## Build / compile / test

```bash
# Compile milestone report
cd milestone && typst compile milestone.typ milestone.pdf
```

Typst fonts configured for macOS: Songti SC (serif), Heiti SC (sans-serif), Menlo (monospace).

There is no runnable code at the project root yet — implementation is planned but not started.

## Architecture

```
Final_Project/
├── INIT.md              # Project proposal / research plan (Chinese)
├── milestone/           # Milestone report in Typst
│   ├── lib.typ          # Typst template (Chinese academic paper style)
│   ├── milestone.typ    # Report source
│   └── ref.bib          # Bibliography
└── submodules/          # Reference implementations (git submodules, read-only)
    ├── diffusion-point-cloud/  # Luo & Hu, CVPR 2021 — simplest DDPM for point clouds
    ├── PVD/                    # Zhou et al., ICCV 2021 — point-voxel diffusion
    ├── Diffusion-SDF/          # Li et al., CVPR 2023 — SDF autoencoder + diffusion
    └── meshflow/               # Li et al., CVPR 2026 — flow matching + DiT for meshes
```

## Key decisions (from conversation history)

- **Training framework:** Flow matching, not DDPM. Simpler training (MSE), fewer sampling steps (50–100 vs 1000), and natural v-prediction support.
- **Base codebase:** `submodules/diffusion-point-cloud` — lightest (3–6M params), cleanest code, full training pipeline, single-GPU friendly. Will modify to support flow matching and switchable x/ε/v prediction.
- **MeshFlow** has no training code (inference only) and is too heavy for this project.
- **Not doing occupancy voxel grids initially** — point clouds are easier to train/eval and still test the manifold hypothesis (2048×3 ambient dim ≫ intrinsic dim).

## Reference papers

| Paper | Key role |
|-------|----------|
| JiT (Li & He, CVPR 2026) | Core theory: manifold hypothesis + x-prediction advantage |
| k-Diff (Jin & Wang, arXiv 2026) | Theoretical framework: ambient vs intrinsic dimension |
| Flow Matching (Lipman et al., 2022) | Training framework choice |
| diffusion-point-cloud (Luo & Hu, CVPR 2021) | Base code to modify |
| PVD / Diffusion-SDF / MeshFlow | Architecture/design reference only |
