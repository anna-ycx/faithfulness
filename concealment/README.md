# Concealment: does NSG catch hidden reasoning?

Extension of Mayne et al. 2026 ("A Positive Case for Faithfulness", arXiv 2602.02639).
Question: if a reference model follows a rule but hides it from its explanation,
does Normalized Simulatability Gain (NSG) drop, or is that a blind spot?

Everything for this project lives in this folder. Small fixes to the paper's own
code (running without a GPU, OpenAI routing, pinning one OpenRouter provider per
model) are in `src/utils.py`, `src/vllm_configs.py` and the three pipeline files.

Run everything from the repo root.

| File | What it does | Plan step |
|---|---|---|
| `prompts.py` | Rule and hiding wordings; `add_instruction()` inserts one into a stored reference prompt | used by all |
| `pairs.py` | Loads a dataset's pairs with each case's real feature values; counts pairs per changing feature | Test 1 |
| `feature_influence.py` | Does changing one feature, or adding the overtime rule, change the reference model's answer? | Test 2 |
| `pilot.py` | No rule / rule allowed / rule hidden, through the full NSG pipeline | Test 3 (to build) |
| `records.py` | Saves `run_info.json` (date, model, provider, settings, cost, git commit) next to results | used by all |

```bash
python -m concealment.pairs --dataset attrition income
python -m concealment.feature_influence            # dry run: plan, example prompt, cost
python -m concealment.feature_influence --run      # calls the API
python -m concealment.feature_influence --analyse-only
```

## Results

| Folder | What |
|---|---|
| `results/pair_counts/` | Test 1: pairs per changing feature, one CSV + summary per dataset |
| `results/pipeline_check_20pairs/` | First end-to-end run of the paper's pipeline on the repo's 20-pair test file (Qwen3-32B reference, gpt-oss-20b predictor) |
| `results/feature_influence/` | Test 2: 506 Qwen3-32B answers on Attrition |

Each results folder has `summary.txt` (the numbers) and, for API runs, `run_info.json`
(what produced them). Findings and decisions, one entry per step: `FINDINGS.md`.
