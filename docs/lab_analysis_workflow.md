# Usage Guide: `lab_quant_analysis.py`

This guide explains how to run and tune:
- `two_stage_framework/two_stage_framework_case_studies/lab_quant_analysis.py`

## Quick Start

From repository root:

```bash
cd two_stage_framework/two_stage_framework_case_studies
python3 lab_quant_analysis.py --mode all
```

Run only runtime scaling:

```bash
python3 lab_quant_analysis.py --mode runtime
```

Run only robustness analysis:

```bash
python3 lab_quant_analysis.py --mode robustness
```

## Parsed Arguments

Defined in `parse_args()`:
- `--output-dir` (line 613)
  - default: `case_studies/lab_analysis_outputs`
- `--mode` (line 614)
  - choices: `all`, `runtime`, `robustness`
- `--algorithm` (line 615)
  - choices: `forward`, `reverse`
- `--runtime-repeats` (line 616)
  - number of repeated trials for Task-1 sweeps
- `--robustness-repeats` (line 617)
  - number of repeated trials for Task-2 sweeps
- `--seed` (line 618)
  - RNG seed for reproducibility
- `--E` (line 619)
  - number of Monte-Carlo hazard samples
- `--N` (line 620)
  - DP horizon length

## Where to Adjust Sweep Values

### Task 1: Runtime scaling
Function: `run_runtime_scaling(...)`

- Tasks sweep list: line 476
  - `task_grid = [1, 2, 3, 4]`
- Agents sweep list: line 483
  - `agent_grid = [1, 2, 3, 4]`
- Map-size sweep list: line 490
  - `size_grid = [(12, 24), (15, 30), (18, 36)]`

### Task 2: Safety/completion robustness
Function: `run_robustness_analysis(...)`

- Hazard-count sweep list: line 539
  - `hazard_grid = [1, 2, 3, 4]`
- Spread-rate sweep list: line 553
  - `spread_grid = [0.01, 0.02, 0.04, 0.08, 0.10]`

## Base Configuration (defaults for all sweeps)

Defined in `main()` as `base_cfg` (lines 635-645):
- map size: `map_width`, `map_height` (lines 637-638)
- fixed counts: `n_tasks`, `n_agents`, `n_hazards` (lines 639-641)
- default spread rate: `spread_rate` (line 642)
- simulation budget: `E`, `N` from CLI args (lines 643-644)

If you want a different default lab setup, edit `base_cfg` first.

## Randomization Behavior

Implemented inside `create_parameters(...)` and helper samplers:
- Robot starts are sampled from exits:
  - call at line 310 (`sample_positions_from_exits`)
- Hazards are sampled from free cells, avoiding robot/task cells:
  - lines 315-317 (`sample_hazard_positions`)

## Outputs

Output root is `output_dir` (line 629), with temporary per-run folders in `tmp_runs` (line 631).

### Runtime mode outputs
- `runtime_scaling_raw_<algorithm>.csv` (line 499)
- `runtime_vs_tasks_boxplot_<algorithm>.png` (line 514)
- `runtime_vs_agents_boxplot_<algorithm>.png` (line 523)
- `runtime_vs_mapcells_boxplot_<algorithm>.png` (line 532)

### Robustness mode outputs
- `robustness_vs_hazard_count_raw_<algorithm>.csv` (line 551)
- `robustness_vs_spread_raw_<algorithm>.csv` (line 565)
- `safety_vs_hazard_count_boxplot_<algorithm>.png` (line 576)
- `task_completion_vs_hazard_count_boxplot_<algorithm>.png` (line 585)
- `safety_vs_spread_boxplot_<algorithm>.png` (line 598)
- `task_completion_vs_spread_boxplot_<algorithm>.png` (line 607)

## Useful Commands

Higher-fidelity run:

```bash
python3 lab_quant_analysis.py --mode all --runtime-repeats 5 --robustness-repeats 20 --E 2000 --N 70 --seed 42
```

Use reverse greedy allocator:

```bash
python3 lab_quant_analysis.py --mode all --algorithm reverse
```

Custom output folder:

```bash
python3 lab_quant_analysis.py --mode robustness --output-dir case_studies/my_lab_runs
```

## Notes

- The script intentionally skips expensive `alpha_G`/`gamma_G` postprocessing in `run_allocator(...)` (line 370) to keep sweeps practical.
- `task_completion_rate` is path-based (targets visited by any robot), computed in `compute_task_completion_rate(...)` (lines 351-357).
- If runs are slow, reduce `--E`, `--N`, and/or repeat counts first.
