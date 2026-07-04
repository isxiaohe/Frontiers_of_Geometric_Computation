# AGENTS.md

Repository guidance for coding agents working in this project.

## Project Context

This is the final project for "Frontiers of Geometric Computation" at PKU.
The project tests whether JiT's manifold-hypothesis argument about prediction
targets transfers to 3D shape generation.

Core question: under a controlled 3D generation setup, does direct
`x_0` prediction converge faster or perform better than `epsilon` or `v`
prediction when ambient dimension is much larger than intrinsic dimension?

The repository now contains a runnable baseline in `src/fgc3d/` plus scripts
for synthetic primitives, structural toy objects, and single-category ShapeNet
experiments. Treat `submodules/` as reference code unless the user explicitly
asks to edit it.

## Read First

Before making project decisions, read these local sources in order:

1. `INIT.md` for the original proposal and experiment intent.
2. `.claude/CLAUDE.md` for the latest repository-specific guidance and prior
   decisions.
3. `milestone/milestone.typ` when editing or summarizing current progress.
4. `submodules/diffusion-point-cloud/README.md` when implementation work begins.
5. `README.md` and `env/README.md` before changing training commands,
   environment setup, or server-run instructions.

Use the local files above as the source of truth before broad web research.

## Repository Layout

This directory is tracked by the git repository one level up:
`/Users/matthew-xh/Study/CS/Frontiers_of_Geometric_Computation`.
When checking history from inside `Final_Project/`, remember that paths in git
are rooted at the parent repository, for example `Final_Project/INIT.md`.

```text
Final_Project/
├── INIT.md
├── AGENTS.md
├── .claude/
│   └── CLAUDE.md
├── milestone/
│   ├── lib.typ
│   ├── milestone.typ
│   └── ref.bib
├── env/
│   ├── README.md
│   ├── requirements-base.txt
│   ├── requirements-cpu.txt
│   └── requirements-cuda-cu128.txt
├── scripts/
│   ├── prepare_shapenetcore_category.py
│   ├── train_generate_primitives.py
│   └── train_generate_shapenet_category.py
├── src/
│   └── fgc3d/
├── tests/
└── submodules/
    ├── diffusion-point-cloud/
    ├── PVD/
    ├── Diffusion-SDF/
    └── meshflow/
```

The `submodules/` directory contains reference implementations. Prefer reading
and adapting ideas from them; do not treat them as owned source unless the user
explicitly asks to edit submodule code.

The initial project state was committed as `b96b067`:
`docs: add project proposal, milestone report, and CLAUDE.md`.

## Current Technical Direction

- Use flow matching as the main training framework. The DDPM/VP code paths are
  reference/sanity baselines, not the main claim.
- Compare `x_0`, `epsilon`, and `v` prediction under controlled conditions:
  fixed dataset, model capacity, schedule, optimizer, training seed, and
  sampling seed.
- Prefer `pvdlite` for server-side ShapeNet smoke/main runs. It is a
  pure-PyTorch approximation of PVD/PVCNN local voxel-point features plus
  shape-level context.
- Point clouds are the current representation. Occupancy or SDF variants are
  extensions after the point-cloud baseline is stable.
- MeshFlow is useful as a flow-matching and mesh-generation reference, but it is
  not the base codebase for this project.

## Build and Verification

Project tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Environment snapshot:

```bash
PYTHONPATH=src .venv/bin/python scripts/record_env.py --output env/current-env.json
```

CPU setup:

```bash
uv venv --python 3.12 .venv
uv pip install -r env/requirements-cpu.txt
```

CUDA 12.8 wheel setup:

```bash
uv venv --python 3.12 .venv
uv pip install -r env/requirements-base.txt
uv pip install -r env/requirements-cuda-cu128.txt
```

Training/generation scripts accept `--device auto`, `--device cpu`, or
`--device cuda:0`. `auto` uses CUDA when PyTorch reports CUDA availability and
otherwise falls back to CPU.

For the milestone report:

```bash
cd milestone
typst compile milestone.typ milestone.pdf
```

The Typst template is configured for macOS fonts:

- Serif: `Songti SC`
- Sans serif: `Heiti SC`
- Monospace: `Menlo`

When editing Typst files, compile before claiming the report is fixed.

## Implementation Guidelines

- Keep changes scoped. Avoid large rewrites of reference code before there is a
  minimal controlled baseline.
- Preserve the central experimental control: the prediction target should change
  while dataset, model capacity, schedule, optimizer, and evaluation remain as
  comparable as possible.
- Prefer simple, inspectable training scripts and configs over abstraction-heavy
  experiment frameworks.
- Record assumptions about dataset category, sample count, point count, target
  parameterization, and evaluation normalization near the relevant code.
- If using metrics from `diffusion-point-cloud`, distinguish validation-time
  metrics from final test-script metrics; its README notes they are not directly
  comparable.
- Do not add network-dependent setup steps without documenting offline or local
  alternatives.

## Writing Guidelines

- Reports may use Chinese academic prose, matching the existing proposal and
  milestone style.
- Keep claims tied to code, experiments, or cited papers. If results are not yet
  run, phrase them as plans, hypotheses, or expected observations.
- For mathematical notation in Typst, verify syntax against the existing
  `milestone/` files before introducing new notation patterns.

## Git and Workspace Safety

- The worktree may contain user changes. Do not revert unrelated edits.
- Do not delete or overwrite generated reports, PDFs, or submodule contents
  unless explicitly requested.
- If a task requires editing submodules, state that clearly before making the
  change and keep edits minimal.
