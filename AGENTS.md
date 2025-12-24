# Repository Guidelines

## Project Structure & Module Organization
Core training lives in `train.py`; configuration defaults and scene scaffolding are in `arguments/` and `scene/`. Supporting math, camera, and vector-quantization helpers are under `utils/`. Rendering and UI integration code resides in `gaussian_renderer/`. Language feature preprocessing starts in `preprocess.py`, with evaluation logic in `eval_lerf.py` and the `eval/` package. Store reusable plots and figures in `assets/`; generated checkpoints and metrics belong in `output/` and `eval_result/`.

## Build, Test, and Development Commands
Set up the environment with `conda env create --file environment.yml` followed by `conda activate langsplat_v2`. Pre-compute language features via `python preprocess.py --dataset_path /abs/path/to/scene [--resolution <pixels>]`. Launch training with `bash train.sh` after updating dataset paths, or run `python train.py -s <scene_root> -m output/<run_id> --start_checkpoint <ply_checkpoint>` for custom sweeps. Validate LERF checkpoints using `bash eval_lerf.sh scene_name index checkpoint_id` or `python eval_lerf.py` for ad-hoc metrics.

## Coding Style & Naming Conventions
Use 4-space indents in Python, snake_case for functions and variables, and CamelCase for classes. Group imports as stdlib, third-party, then local modules. Move tensors explicitly to devices (e.g., `tensor.to(torch.device("cuda"))`) and guard CUDA-only logic. Add concise inline comments when manipulating geometry or high-dimensional tensors.

## Testing Guidelines
No formal unit suite exists; rely on deterministic script runs. Capture the exact command, dataset snapshot, and headline metrics (PSNR, language IoU) when validating changes. Re-run `python preprocess.py` and a short `python train.py ... --iterations 1000` smoke test after touching preprocessing or renderer code. Run `python eval_lerf.py ...` on a representative checkpoint before merging and stash resulting artifacts in `eval_result/`.

## Commit & Pull Request Guidelines
Write imperative, descriptive commit messages (e.g., `Add RVQ warmup for language features`). Keep PRs scoped, summarizing intent, affected scenes or datasets, and metric deltas. Link tracking issues, include reproducible commands, and share updated plots from `eval_result/` when metrics move. Host large assets externally and document download links instead of committing binaries.

## Data & Security Notes
Avoid committing scene datasets; reference absolute paths in configs. Review new scripts for filesystem writes and guard anything that assumes GPU availability so CPU runs fail gracefully.
