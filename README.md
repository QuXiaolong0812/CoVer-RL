# CoVer-RL for CMMEvent

This repository implements CoVer-RL for document-level multimodal event extraction on CMMEvent. The pipeline builds structured perturbations (SPP), warms up a Qwen3-VL verifier, trains an event-entailment scorer, and alternates generator and verifier LoRA updates with GRPO. It uses hierarchical structured feedback (HSFP) and provides a Triangular Consistency Reward (TCR) evidence-scoring path.

The [cloud configuration](configs/cmmevent_cloud.json) uses Qwen3-VL-8B-Instruct for generation and Qwen3-VL-4B-Instruct for verification. The [local configuration](configs/inference.json) uses 2B models for smoke tests. The commands below assume Linux Bash and the repository root as the working directory.

## Setup

Use an NVIDIA GPU with BF16 support for Qwen3-VL training. Install the PyTorch wheel that matches your driver, followed by the pinned dependencies in [requirements.txt](requirements.txt). The example uses Python 3.11 and the CUDA 12.6 wheels. See the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v280) for other wheel variants.

```bash
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

The pinned runtime includes Transformers, PEFT, Accelerate, Pillow, SentencePiece, and the Hugging Face Hub CLI. Confirm that the GPUs you intend to use appear in the Python environment.

```bash
python - <<'PY'
import torch
import torchvision

print("torch", torch.__version__, "torchvision", torchvision.__version__, "CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        free, total = torch.cuda.mem_get_info(index)
        print(f"cuda:{index}", torch.cuda.get_device_name(index),
              f"free={free / 2**30:.1f} GiB", f"total={total / 2**30:.1f} GiB",
              "BF16:", torch.cuda.is_bf16_supported(including_emulation=False))
PY
```

## Download the models

The cloud profile expects model folders under `/models`. Set `MODEL_ROOT="$PWD/models"` first if you prefer a writable directory inside the project. The five downloads use about 30 GB before caches and checkpoints. The commands pin specific revisions from the official model repositories. See the [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-download).

| Role | Model repository |
| --- | --- |
| Generator | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Verifier | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Text entailment backbone | [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base) |
| Image and text embeddings | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) |
| Region proposals | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |

```bash
MODEL_ROOT=${MODEL_ROOT:-/models}
mkdir -p "$MODEL_ROOT"
df -h "$MODEL_ROOT"

hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir "$MODEL_ROOT/Qwen3-VL-8B-Instruct"

hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir "$MODEL_ROOT/Qwen3-VL-4B-Instruct"

hf download microsoft/mdeberta-v3-base config.json tokenizer_config.json spm.model pytorch_model.bin \
  --revision a0484667b22365f84929a935b5e50a51f71f159d \
  --local-dir "$MODEL_ROOT/mdeberta-v3-base"

hf download google/siglip2-base-patch16-224 \
  --revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 \
  --local-dir "$MODEL_ROOT/siglip2-base-patch16-224"

hf download IDEA-Research/grounding-dino-tiny \
  --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 \
  --exclude 'pytorch_model.bin' --local-dir "$MODEL_ROOT/grounding-dino-tiny"
```

Each Qwen directory needs every safetensors shard, the index, and the processor and tokenizer files. All stages load from local folders after download. The Microsoft checkpoint initializes `train-nli`. TCR uses the resulting `nli_checkpoint/`, including its dev calibration file.

## Prepare CMMEvent

Obtain the annotations and images from the dataset authors. The original paper lists the data as available on request. [Liu et al., *Information Processing & Management*](https://doi.org/10.1016/j.ipm.2024.104027)

Place the files under the following paths, preserving image names and case.

```text
datasets/
└── CMMEvent/
    ├── annotation_text.json
    ├── annotation_image.json
    └── image/
        └── img/
```

Select the run configuration and devices. The example uses two GPUs. On a single GPU with sufficient memory for both backbones and training overhead, set `VER_DEVICE=cuda:0`. GRPO loads each policy onto its assigned device. Run these assignments again after opening a new shell.

```bash
CONFIG=configs/cmmevent_cloud.json
MODEL_ROOT=${MODEL_ROOT:-/models}
MODEL_ARGS=(--config "$CONFIG" --models-dir "$MODEL_ROOT")
GEN_DEVICE=cuda:0
VER_DEVICE=cuda:1
EVIDENCE_DEVICE=cpu
```

Preprocessing creates `datasets/processed/cmmevent/` with split JSONL files, split IDs, and an audit report. It requires a new output directory. If you already have a compatible processed copy, proceed directly to SPP. Use `--split-file` with a trusted split manifest when you need to reproduce a specific partition. The local dataset used for this project produced 4,121 train, 515 dev, and 515 test documents after preprocessing.

```bash
# Run only when the processed directory does not already exist.
python main.py preprocess --config "$CONFIG"

python build_spp.py --profile full \
  --data-dir datasets/processed/cmmevent --dataset-dir datasets/CMMEvent
```

The `full` profile enables all eight operators. It caps negative selections at 50 per operator for train and 15 for dev, pairing each with a clean candidate. Check `spp_audit.json` for counts and rejection reasons. To reproduce a reviewed A2 selection, add `--a2-review /path/to/a2-review.json` using a separately obtained review file that matches the train and dev SHA-256 hashes. The default command uses the built-in A2 selection rules.

## Run a smoke test

Replace each `REPLACE_WITH_RUN_ID` with the directory name printed by the preceding successful stage. The short training runs below check integration and memory use. Use the full checkpoints from the next section for experiments.

```bash
SPP_DIR='runs/spp/REPLACE_WITH_RUN_ID'

python main.py generate "${MODEL_ARGS[@]}" --device "$GEN_DEVICE" --split dev --limit 3
DEV_PREDICTIONS='runs/generator_infer/REPLACE_WITH_RUN_ID/predictions.jsonl'
python main.py evaluate-extraction --config "$CONFIG" --predictions "$DEV_PREDICTIONS"

python main.py train-verifier-sft "${MODEL_ARGS[@]}" --device "$VER_DEVICE" \
  --spp-dir "$SPP_DIR" --max-steps 20 --eval-samples 9
SMOKE_SFT='runs/verifier_sft/REPLACE_WITH_RUN_ID/checkpoint'

python main.py train-nli "${MODEL_ARGS[@]}" --device "$GEN_DEVICE" \
  --epochs 1 --train-limit 4 --dev-limit 2
SMOKE_NLI='runs/event_nli/REPLACE_WITH_RUN_ID/nli_checkpoint'

python main.py verify "${MODEL_ARGS[@]}" --device "$VER_DEVICE" --all \
  --predictions "$DEV_PREDICTIONS" --adapter-dir "$SMOKE_SFT"
DEV_FEEDBACK='runs/verify_generated/REPLACE_WITH_RUN_ID/feedback.jsonl'

python main.py reward-dry-run "${MODEL_ARGS[@]}" --predictions "$DEV_PREDICTIONS" \
  --feedback "$DEV_FEEDBACK" --nli-dir "$SMOKE_NLI" --evidence-device "$EVIDENCE_DEVICE"

python main.py train-rl "${MODEL_ARGS[@]}" --adapter-dir "$SMOKE_SFT" --nli-dir "$SMOKE_NLI" \
  --generator-device "$GEN_DEVICE" --verifier-device "$VER_DEVICE" --evidence-device "$EVIDENCE_DEVICE" \
  --cycles 1 --groups-per-update 1 --generator-group-size 2 --verifier-group-size 2 --dev-limit 3
```

Check each `summary.json` for `status: succeeded`. Inspect generator and verifier format validity, token limits, and `evidence_scored`. Confirm that the RL cycle contains both LoRA adapters and `training_state.pt`. When rewards tie within a group, its group-relative advantage is zero. Use `advantage_groups` and `grad_norm` alongside the format metrics.

## Train and resume

Train the verifier adapter on the same Qwen3-VL backbone that will load it. Train mDeBERTa on synthetic event-entailment pairs from train and calibrate it on dev.

```bash
python main.py train-verifier-sft "${MODEL_ARGS[@]}" --device "$VER_DEVICE" --spp-dir "$SPP_DIR"
SFT_DIR='runs/verifier_sft/REPLACE_WITH_RUN_ID/checkpoint'

python main.py train-nli "${MODEL_ARGS[@]}" --device "$GEN_DEVICE"
NLI_DIR='runs/event_nli/REPLACE_WITH_RUN_ID/nli_checkpoint'

RL_ARGS=("${MODEL_ARGS[@]}" --adapter-dir "$SFT_DIR" --nli-dir "$NLI_DIR" \
  --generator-device "$GEN_DEVICE" --verifier-device "$VER_DEVICE" --evidence-device "$EVIDENCE_DEVICE")
python -u main.py train-rl "${RL_ARGS[@]}"
```

The cloud profile defaults to three generator updates and one verifier update per cycle, five samples per group, and eight groups per update. The supplied CMMEvent preprocessing treats references as complete, so structural F1 supplies the outcome reward. With `--nli-dir`, the trainer also computes TCR evidence scores. Enable `rl.trace_rollouts` to retain those scores per rollout. You may omit `--nli-dir` when training only on complete references and run TCR separately as a diagnostic. Rows marked `reference_complete: false` require TCR.

Each completed cycle saves `generator/`, `verifier/`, and `training_state.pt`. To pause cleanly, add `--stop-after-cycle 1` to a run planned for more than one cycle. For resume, keep the original effective configuration, cycle budget, devices, input paths, and initial adapters. Preserve the earlier run directory because the best dev checkpoint can still point to it.

```bash
RESUME_DIR='runs/coevolution/REPLACE_WITH_RUN_ID/cycle-0001'
test -s "$RESUME_DIR/training_state.pt"
python -u main.py train-rl "${RL_ARGS[@]}" --resume "$RESUME_DIR"
```

## Evaluate the test split

Once a run has completed a dev evaluation, `best_checkpoint.json` points to the model pair selected by dev F1. Use that pair for test inference.

```bash
RL_RUN='runs/coevolution/REPLACE_WITH_RUN_ID'
PAIR=$(python -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8"))["pair"]; assert p, "No dev-selected checkpoint"; print(p)' \
  "$RL_RUN/best_checkpoint.json")

python main.py generate "${MODEL_ARGS[@]}" --device "$GEN_DEVICE" --split test --all \
  --generator-adapter-dir "$PAIR/generator"
TEST_PREDICTIONS='runs/generator_infer/REPLACE_WITH_RUN_ID/predictions.jsonl'
python main.py evaluate-extraction --config "$CONFIG" --split test --predictions "$TEST_PREDICTIONS"

python main.py verify "${MODEL_ARGS[@]}" --device "$VER_DEVICE" --all \
  --predictions "$TEST_PREDICTIONS" --adapter-dir "$PAIR/verifier"
TEST_FEEDBACK='runs/verify_generated/REPLACE_WITH_RUN_ID/feedback.jsonl'
python main.py reward-dry-run "${MODEL_ARGS[@]}" --predictions "$TEST_PREDICTIONS" --feedback "$TEST_FEEDBACK"

python main.py verifier-eval "${MODEL_ARGS[@]}" --device "$VER_DEVICE" --all \
  --spp-dir "$SPP_DIR" --adapter-dir "$PAIR/verifier"
```

Check `coverage.complete` in the extraction report. It reports document-level argument precision, recall, and F1, with separate results for single- and multi-event documents. The evaluator treats malformed generations as empty predictions. The verifier report tracks how many candidates were eligible and how many produced valid feedback. `verifier-eval` separately evaluates the base model and adapter on SPP dev.

For a zero-shot baseline, run the same test generation command without a generator adapter. For an outcome-reward-only baseline, use a separate configuration with `rl.alpha_start`, `rl.alpha_end`, and `rl.verifier_updates` set to zero, and omit `--nli-dir`.

## Logs and repository contents

Each run writes its resolved configuration, input fingerprints, log, metrics, and summary to `runs/<stage>/<run-id>/`. It archives logs and reports under `archives/<stage>/`. Back up model adapters, optimizer state, and generated training data separately if you plan to resume or reuse them. Use a persistent terminal session such as `tmux` for long jobs.

The source tree includes the training and evaluation scripts, configuration profiles, and lightweight tests. Keep raw and processed datasets, pretrained and trained weights, local manuscript drafts, and experiment outputs outside the source release.

## Tests

Run the suite with the processor assets from a local Qwen3-VL checkpoint. The tests use CPU and do not need the CMMEvent dataset or full model weights. If you already downloaded the 4B model, set `COVER_RL_TEST_MODEL_DIR` to that directory. For a code-only checkout, download just the processor assets.

```bash
MODEL_ROOT=${MODEL_ROOT:-"$PWD/models"}
PROCESSOR_DIR="$MODEL_ROOT/Qwen3-VL-4B-processor"
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --include '*.json' 'merges.txt' --local-dir "$PROCESSOR_DIR"
COVER_RL_TEST_MODEL_DIR="$PROCESSOR_DIR" python -B -m unittest discover -s tests -v
```
