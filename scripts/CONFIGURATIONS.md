# Paper training configurations

This directory retains 13 numbered recipes used by the current paper's main
experiments or appendix temperature study. Both base launchers and `common.sh` live in `launchers/`; the shared runtime,
upstream wrapper, and project OPD hook remain in this directory as dependencies.
Scripts preserve their previous training defaults; appendix pilot overrides are
explicit below. This export does not bundle evaluation/analysis code or artifacts.

## Numbering and provenance

Export filenames and displayed experiment IDs (`GRID_EXP_INDEX`) are numbered
consecutively. Explicit `RUN_NAME` values and training settings are unchanged;
automatically generated names use the new numeric suffix. Original IDs below
refer to the research workspace, whose scripts have not been changed.

| Export ID | Original workspace ID |
|---|---|
| `run-001` | `003` |
| `run-002` | `005` |
| `run-003` | `006` |
| `run-004` | `007` |
| `run-005` | `008` |
| `run-006` | `016` |
| `run-007` | `019` |
| `run-008` | `020` |
| `run-009` | `025` |
| `run-010` | `027` |
| `run-011` | `028` |
| `run-012` | `029` |
| `run-013` | `030` |

## Paper-to-recipe mapping

| Recipes | Paper use |
|---|---|
| `run-004` | Main DeepScaleR setting: teacher temperature 0.75, 100 rollouts. |
| `run-006` | Main JustRL setting: teacher temperature 1.0, 100 rollouts. |
| `run-007`, `run-008`, `run-009` | Section 5: Qwen3-4B Base, released-SFT warmup, and masking. |
| `run-010`, `run-011` | Section 5: Qwen3-8B Base and masking. |
| `run-012`, `run-013` | Section 5: Qwen3-30B-A3B Base and masking. |
| `run-001`, `run-002`, `run-003`, `run-004`, `run-005` | Appendix A.2: five teacher-scoring temperatures, **with the overrides below**. |

The masked configurations combine completed-only masking and EOS mapping;
they do not isolate masking from token mapping.

## Appendix A.2 training pilot

These are protocol reconstructions using retained recipes, not a claim that the
current filenames exactly match historical launchers. In particular, run-001 defaults
to temperature 0, and run-004 defaults to the longer main experiment. Override those
settings to reproduce the pilot protocol. Historical experiment identifiers drifted.

For each row, set the listed environment variables when invoking the indicated script.
All pilot runs use `N_SAMPLES_PER_EVAL_PROMPT=4`,
`EVAL_DATASETS="AMC23 AIME24 AIME25"`, `EVAL_MAX_RESPONSE_LEN=16384`,
and `EVAL_INTERVAL=20`. Use a distinct `RUN_NAME` for each pilot run.

| Script | `TEACHER_TEMPERATURE` | `GRID_NUM_ROLLOUT` | `SAVE_INTERVAL` |
|---|---:|---:|---:|
| `run-003` | 0.1 | 75 | 20 |
| `run-002` | 0.5 | 100 | 50 |
| `run-004` | 0.75 | 75 | 20 |
| `run-001` | 1.0 | 100 | 20 |
| `run-005` | 1.5 | 75 | 20 |

The 1.5 historical run has only the initial and rollout-20 evaluations in the paper;
the configured 75-rollout budget is not evidence that it finished.
The standalone frozen-teacher temperature table is a separate evaluation study;
its evaluation scripts remain outside this training-only export.

## Removed configurations (original workspace IDs)

The following IDs refer to the original research workspace, not the new export
numbering. Removed from this export: `001`, `002`, `004`, `009`–`015`, `017`, `018`,
`021`–`024`, `026`, and `031`. They cover superseded initial recipes, DAPO,
other temperature/evaluation protocols, older 16k Base/SFT comparisons, DeepMath
8B/cold-start experiments, and later refill/filter follow-ups not used by the current
manuscript. Their original source scripts are retained in the research workspace.
Run-032 was never included in this export and is not added here.

Selection was checked against the active sections in the current NeurIPS source
archive and the corresponding ICLR snapshot, then against the Figure 8 source mapping
and frozen temperature-pilot configurations. Evidence-index entries alone were not
treated as proof that an experiment appears in the current manuscript.
The manuscript's generic Qwen training table still states learning rate 1e-5 while
the retained 8k experiments use 1e-6; the actual experiment configurations are preserved.

## Supporting files

- `launchers/common.sh`: resolves shared experiment defaults, validates batch settings,
  and dispatches the selected base launcher.
- `launchers/run-deepmath-*.sh`: converts the initial model if necessary, starts the
  teacher service and Ray, and submits training.
- `on_policy_distillation.py`: project OPD reward hook; requests teacher token
  logprobs, processes scoring results, grades evaluation answers, and implements
  optional completed-only filtering and terminal-token mapping. The loss itself
  remains in `slime/`.
- `runtime.sh`: shared shell helpers for free ports, service health checks, Ray job
  status, worker environment forwarding, and proxy exclusions.
- `upstream_entry.py`: runs upstream training/conversion entry points while ensuring
  this project's `slime/` takes import precedence over the upstream implementation.

## Runtime and inputs

Activate the dependency environment. `SLIME_UPSTREAM_DIR` defaults to `../slime`,
relative to this project; it must point to the pinned upstream checkout. Its `train.py`, model configuration scripts, plugins, and conversion tool
are used without copied upstream entry points. `MEGATRON_PATH` defaults to its
`Megatron-LM`; `SGLANG_PATH` defaults to its **patched** `sglang/python`.
Source versions and patch application are maintained in [install_dependencies.sh](install_dependencies.sh).
`upstream_entry.py` places this export first on `sys.path`, checks the imported slime
location, and executes upstream entry points via `runpy`. Ray workers receive the same
project-first `PYTHONPATH` and the patched SGLang path.

All relative filesystem settings, including overrides, are resolved against the
project root, independent of the caller's working directory. Defaults are `models/`,
`data/`, and `outputs/`; logs, Ray state, and compile caches also live under outputs.
Environment installation and default paths are documented in [the root README](../README.md).
Required model/data files and output overrides are listed below.
W&B account and credentials come from the environment, with no private account
default. Shell tracing is disabled to avoid echoing credentials.

These are single-host launchers for a dedicated Ray environment. `STOP_EXISTING_RAY=0`
is the default; setting it to 1 explicitly stops existing local Ray processes.
Cleanup stops Ray only after this launcher successfully starts its Ray head, and
terminates its own teacher process. Do not share that local Ray session with other jobs.


## Inspection and verification

Renumbering changed filenames and `GRID_EXP_INDEX` only. All 13 full dry-runs
matched after normalizing displayed IDs, generated experiment suffixes, and run
timestamps; training settings and explicit run names were unchanged. All 18 shell
files passed syntax checks; no training was launched.

Use `--dry-run` to inspect resolved settings and the full training argument vector
without starting training, Ray, or SGLang. Set `SLIME_UPSTREAM_DIR` to include the
upstream model architecture arguments. Without it, dry-run explicitly reports the
unresolved model configuration. Dry-run does not validate model/data availability.

After pruning: all 17 shell files passed `bash -n`; all 13 numbered recipes plus
the two base launchers passed full dry-run. All five appendix override combinations
were checked for temperature, rollout count, save/eval intervals, sample count,
datasets, and response cap. Kept code and training defaults were not changed.
Earlier export checks covered project-first imports and 23 CPU cases for metrics,
EOS mapping and completed-only filtering; they were not rerun for this file removal.
Moving the three launcher files was checked separately: all 17 shell syntax checks
and 15 full dry-runs passed, with identical training argument vectors before/after.
Path portability was subsequently validated with 60 full dry-runs (15 entry points,
two caller directories, default and relative overrides), plus spaced-path/runtime-env
and upstream-bootstrap checks. No GPU training was run, and no tests are shipped.

## Retained script files

- `run-001-deepmath-deepscaler-deepseek-distill-1.5b.sh`
- `run-002-deepmath-deepscaler-deepseek-distill-1.5b-teacher-temp0.5.sh`
- `run-003-deepmath-deepscaler-deepseek-distill-1.5b-teacher-temp0.1.sh`
- `run-004-deepmath-deepscaler-deepseek-distill-1.5b-teacher-temp0.75.sh`
- `run-005-deepmath-deepscaler-deepseek-distill-1.5b-teacher-temp1.5.sh`
- `run-006-deepmath-justrl-deepseek-distill-1.5b-teacher-temp1.0.sh`
- `run-007-openthoughts-qwen3-4b-qwen3-1.7b-base-teacher-temp1.0-lr1e-6-len8k.sh`
- `run-008-openthoughts-qwen3-4b-qwen3-1.7b-sft-teacher-temp1.0-lr1e-6-len8k.sh`
- `run-009-openthoughts-qwen3-4b-qwen3-1.7b-base-completed-only.sh`
- `run-010-openthoughts-qwen3-8b-qwen3-1.7b-base-teacher-temp1.0-lr1e-6-len8k.sh`
- `run-011-openthoughts-qwen3-8b-qwen3-1.7b-base-completed-only.sh`
- `run-012-openthoughts-qwen3-30b-a3b-qwen3-1.7b-base-teacher-temp1.0-lr1e-6-len8k.sh`
- `run-013-openthoughts-qwen3-30b-a3b-qwen3-1.7b-base-completed-only.sh`
- `launchers/run-deepmath-1.5b-opd.sh`
- `launchers/run-deepmath-1.7b_Base-4b-opd.sh`

## Required model directories

Only the models used by the selected recipe are required. Each directory must contain
the model weights, configuration, and tokenizer files expected by the pinned environment.

| Experiment family | Student directory under `models/` | Teacher directory under `models/` |
|---|---|---|
| DeepScaleR and temperature pilot | `DeepSeek-R1-Distill-Qwen-1.5B` | `DeepScaleR-1.5B-Preview` |
| JustRL | `DeepSeek-R1-Distill-Qwen-1.5B` | `JustRL-DeepSeek-1.5B` |
| Qwen3 Base / masked | `Qwen3-1.7B-Base` | `Qwen3-4B`, `Qwen3-8B`, or `Qwen3-30B-A3B`, as selected |
| Qwen3 warmup (run-008) | `Qwen3-1.7B-SFT` | `Qwen3-4B` |

The SFT directory must contain the released SFT initialization used by the paper,
not a generic Qwen3-1.7B checkpoint. These scripts do not perform that SFT stage.

`REF_LOAD` defaults to the student directory name plus `_torch_dist` under `MODEL_ROOT`.
Supply a matching Megatron checkpoint, or the launcher invokes the upstream conversion
tool when `latest_checkpointed_iteration.txt` is absent. The conversion destination
must be writable. Override `STUDENT_MODEL`, `TEACHER_MODEL`, and `REF_LOAD` individually
if your directory names differ; model variables denote local directories.

## Required data files

```text
data/
├── DeepMath-103K/math_train_level6.jsonl
├── OpenThoughts3_opd.parquet
└── test_data/
    ├── AMC23/test.slime.jsonl
    ├── AIME24/test.slime.jsonl
    ├── AIME25/test.slime.jsonl
    └── AIME26/test.slime.jsonl
```

- DeepScaleR/JustRL recipes use the DeepMath level-at-least-6 training pool; JSONL rows
  need `prompt` and `label` fields.
- Qwen3 recipes use the prepared OpenThoughts3 parquet with `prompt`; their rollout
  label key is disabled. Keep the original experiment's filtering and prompt format.
- Evaluation JSONL rows need `prompt` and `label`. DeepScaleR/JustRL use AMC23, AIME24,
  and AIME25 by default; Qwen3 launchers additionally require AIME26.
- `prompt` must be compatible with the selected tokenizer's chat template. Renaming
  a raw dataset file does not replace the required preprocessing.

Override `PROMPT_DATA` for the training file, `EVAL_DATA_ROOT` for the evaluation root,
or `EVAL_AMC23_JSONL`, `EVAL_AIME24_JSONL`, `EVAL_AIME25_JSONL`, and
`EVAL_AIME26_JSONL` for individual files. `DATASET_DIR` optionally overrides the
DeepMath directory used by the base launchers. Preparation scripts and data downloads
are outside this training-only export; users must supply these prepared inputs.

## Generated paths

Checkpoints default to `outputs/checkpoints/<EXP_NAME>`, rollout dumps to
`outputs/runs/<EXP_NAME>`, and teacher logs to `outputs/logs/<EXP_NAME>`.
`SAVE_DIR`, `DUMP_DIR`, `RUN_LOG_DIR`, `TEACHER_LOG_FILE`, and `WANDB_DIR` can override
individual destinations using the same project-relative convention.
`WANDB_DIR` defaults to `outputs/runs/<EXP_NAME>/wandb`. The 1.5B launcher also accepts
`SAVE_EVAL_OUTPUT=1` for `outputs/runs/<EXP_NAME>/eval_output`, or an explicit relative
path. Leave it unset to use the recipe's existing behavior.

| Runtime variable | Default under `OUTPUT_ROOT` |
|---|---|
| `RAY_TEMP_DIR` | `ray` |
| `FLASHINFER_WORKSPACE_BASE` | `cache/flashinfer` |
| `TRITON_CACHE_DIR` | `cache/triton` |
| `TORCHINDUCTOR_CACHE_DIR` | `cache/torchinductor` |
| `VLLM_CACHE_ROOT` | `cache/vllm` |

Teacher logs and Ray's driver-log lookup use the configured locations rather than a
fixed system temporary directory. Existing `TMPDIR` does not control these script
paths. The scripts create output directories during actual execution, not during dry-run.
Keep the resolved `RAY_TEMP_DIR` short enough for Ray's Unix socket path limit when
placing the repository under a deeply nested directory.

