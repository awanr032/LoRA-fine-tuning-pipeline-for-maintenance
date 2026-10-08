# Distilling fine-grained maintenance information extraction into a small model

Turning short, messy industrial **maintenance work orders** into structured knowledge graphs
(entities with fine-grained types + relations), using **LLM distillation**: a large teacher model labels
data, and a small open model (**Mistral 7B + LoRA**) is fine-tuned to do the same job cheaply and locally.

Dataset: [MaintIE](https://github.com/nlp-tlp/maintie) (Bikaun et al., LREC-COLING 2024): 1,076 human-annotated
("gold") work orders with 224 fine-grained entity types and 6 relation types, plus 7,000 "silver" work orders
annotated only with the 5 top-level classes.

## The problem

The silver set has only coarse labels (`PhysicalObject`, `State`, `Process`, `Activity`, `Property`), so there is far
too little fine-grained training data. Example of what the task requires:

```
"<id> air conditioner thermostat not working"

air conditioner  -> PhysicalObject/EmittingObject/ElectricCoolingObject
thermostat       -> PhysicalObject/SensingObject/TemperatureSensingObject
not working      -> State/UndesirableState/FailedState
air conditioner --hasPart--> thermostat
not working     --hasParticipant/hasPatient--> thermostat
```

## Approach

```
silver texts (coarse labels)
      │
      ▼
Teacher: Qwen3.6-27B, self-hosted with vLLM on one A100 80GB
  - receives the silver entity spans + coarse types, assigns detailed types and relations
  - prompt: rules + allowed types with example terms + 20 worked examples
  - every answer validated (JSON, allowed types, detailed type must match coarse type, valid relation indices)
      │
      ▼
6,709 validated fine-grained training examples (95.9% pass rate)
      │
      ▼
Student: Mistral-7B-Instruct-v0.3 + LoRA (bf16), full extraction from the raw work order
      │
      ▼
Evaluation on 826 human-annotated gold test texts
```

### Leakage controls
- Silver texts duplicating gold texts were removed before labelling.
- A 150-text **prompt pool** of gold texts supplied the teacher's few-shot examples and example terms; it is **excluded** from validation and test.
- The 50-text pilot set was kept out of the prompt pool and reused for every teacher comparison.
- A safety check asserts that no validation or test text appears in training.
- Split indices are saved (`splits.json`) for reproducibility.

## Results

### Teacher pilot (50 gold texts, spans given, detailed type + relations)

| Teacher setup | Exact type acc. | First two levels | Relation F1 | Failed |
|---|---|---|---|---|
| DeepSeek API, first prompt (6 examples), 4k token limit | 51.8% | 61.4% | 0.69 | 12/50 (truncated) |
| Qwen3.6-27B, improved prompt, thinking **on** | 88.0% | 92.8% | 0.85 | 2/50 |
| Qwen3.6-27B, improved prompt, thinking **off** | **86.1%** | 91.6% | 0.85 | 3/50 |

Teacher and prompt changed together between the first and second rows, so the gain cannot be attributed to either alone.
Thinking off cost ~2 points but was many times faster, so it was used for the full labelling.

### Student on the gold test set (826 texts, 2,598 entities)

| Metric | Precision | Recall | F1 (95% CI) |
|---|---|---|---|
| **Entity exact (span + full detailed type)** | 0.765 | 0.784 | **0.774** (0.757–0.791) |
| Entity, first two type levels | 0.828 | 0.848 | 0.838 (0.823–0.851) |
| Entity, top-level class | 0.933 | 0.956 | 0.944 (0.935–0.952) |
| Entity span only | 0.938 | 0.960 | 0.949 (0.941–0.957) |
| Relation (spans + relation type) | 0.806 | 0.837 | 0.821 (0.801–0.842) |
| Relation strict (+ both entity types exact) | 0.532 | 0.553 | 0.542 (0.515–0.573) |

Invalid JSON: **0 / 826**. Confidence intervals by bootstrap resampling over texts.

Exact-type F1 by top-level class:

| Class | F1 | Gold entities |
|---|---|---|
| Activity | 0.896 | 603 |
| PhysicalObject | **0.688** | 1,513 |
| Process | 0.973 | 111 |
| Property | 0.964 | 29 |
| State | 0.874 | 342 |

### Official MaintIE test set (108 texts; "clean 91" excludes 17 texts that appeared in the teacher's prompt pool)

The official split was recreated with MaintIE's own `create_datasets.py` (seed 1337, 80/10/10: 860 / 108 / 108).

| System | Training labels | Entity micro F1 | Entity macro F1 | Relation (spans only) | Relation (with types) |
|---|---|---|---|---|---|
| **Mistral 7B + LoRA, distilled** | 6,709 teacher labels | **63.8** (CI 58.9–68.7) | **42.9** | **80.2** | **29.8** |
| Mistral 7B + LoRA, gold | 860 expert labels | 58.0 | 30.2 | 74.9 | 25.5 |
| Mistral 7B + LoRA, distilled (quantity-matched) | 860 teacher labels | 55.2 | 25.9 | 74.1 | 17.0 |
| SpERT (published) | 860 expert labels | 64.43 | 38.86 | 55.67 | 27.32 |
| REBEL, silver pre-training (published) | 860 expert + silver | n/a | n/a | 76.07 | 0.00 |

On the clean 91 texts the distilled model scores 61.5 entity micro F1, 42.9 macro, 80.3 / 27.6 for relations.

### Controlled comparisons (same model and settings; paired bootstrap, all 108 texts)

| Comparison | Entity micro | Entity macro | Relation (with types) |
|---|---|---|---|
| Distilled 6,709 vs gold 860 | **+5.8, p=0.003** | **+12.7, p<0.001** | +4.3, n.s. |
| Distilled 860 vs gold 860 (label quality) | −2.8, n.s. | −4.3, n.s. | **−8.5, p=0.007** |
| Distilled 6,709 vs distilled 860 (label quantity) | **+8.6, p<0.001** | **+17.0, p<0.001** | **+12.8, p<0.001** |

**Takeaway:** per example, expert labels are somewhat better (especially for typed relations), but cheap teacher labels
at ~8x the volume outperform them, most clearly on rare types. On the clean 91 texts the main micro-F1 gain is
borderline (+4.0, p=0.051) while macro remains highly significant. All results are single training runs; seed repeats are in progress.

### Where the remaining errors come from (internal test, 459 type errors)

50% inherited from systematic teacher conventions (e.g. "cracked" → FailedState where annotators use DegradedState),
32% words never seen in training, 11% inconsistent teacher labels, and only **7% genuine student errors**:
label quality and coverage, not student capacity, are the bottleneck.

Full numbers with confidence intervals and p-values: [`results/RESULTS_REGISTRY.md`](results/RESULTS_REGISTRY.md) and `results/all_results.csv`.

### Findings
- The student finds the right words almost always (span F1 0.949) and rarely confuses top-level classes (0.944).
- Most errors are **fine-grained subtypes within the right class**, concentrated in **PhysicalObject** (58% of test entities,
  by far the most subtypes, classified by function rather than name).
- Strict relation F1 is low because a relation needs two exact entity types (roughly 0.77 × 0.77).
- Rough estimate: type accuracy on found entities ≈ 0.774 / 0.949 ≈ 82%, close to the teacher's ~86% pilot accuracy.

### Limitations
- 17 test types (52 entities, ~2%) never appear in the training labels; all are rare physical-object types.
- Teacher labels inherit silver entity spans, so silver span errors carry over.
- MaintIE is public, so the teacher may have seen it during pretraining.
- Single training run per model so far; seed repeats in progress.
- 17 official test texts appeared in the teacher's prompt pool; results are also reported on the clean 91.
- Silver spans come from a model trained on fine-grained data, so the pipeline is not fully independent of expert annotation.

## Engineering notes
- **Self-hosting vs API:** the API teacher hit concurrency limits tied to account balance and the model behind
  `deepseek-flash` changed on 10 Sept 2026; self-hosting gave no per-token cost, no rate limits and a pinned model.
- **vLLM:** continuous batching, PagedAttention and prefix caching (94% prefix-cache hit rate, since every request shares
  the same long instructions). Parallel requests turned an hours-long sequential pilot into minutes.
- **Thinking mode:** with thinking on, labelling was estimated at ~21 hours; off, ~128 output tokens per text and a large speed-up.
- See [`docs/lessons_learned.md`](docs/lessons_learned.md) for the debugging stories.

## Repository layout

```
src/maintie_lora/
  evaluate.py                                 scoring (parse_json, locate, score_text, prf, macro_from, ...)
                                               shared by notebooks 05, 07 and 08, which used to each carry
                                               their own copy
tests/
  test_evaluate.py                            unit tests for src/maintie_lora/evaluate.py
notebooks/
  01_teacher_labelling_deepseek_api.ipynb     first teacher (API), kept for the record
  02_teacher_labelling_selfhosted_vllm.ipynb  final teacher: Qwen3.6-27B on vLLM
  03_build_training_data.ipynb                train / val / test files + split indices
  04_finetune_mistral_lora.ipynb              LoRA (or QLoRA on smaller GPUs) fine-tuning
  05_evaluate_student.ipynb                   generation + scoring against gold (internal test)
  06_official_split_and_baseline_data.ipynb   official MaintIE split, gold and quantity-matched training data
  07_official_eval_and_comparisons.ipynb      official-test evaluation, paired significance tests
  08_error_analysis.ipynb                     error kinds and error-source attribution
scripts/
  finetune_mistral_qlora.py                   standalone QLoRA training script
  organize_results_drive.py                   freezes raw results on Drive with a checksum manifest
results/
  RESULTS_REGISTRY.md                         every number in the project, versioned
  all_results.csv                             the same, machine-readable
docs/lessons_learned.md
```

Each Colab notebook clones this repository and adds `src/` to `sys.path` so it can import
`maintie_lora.evaluate` instead of redefining the same scoring functions locally.

## Development

```
pip install -e ".[dev]"
pytest
```

## Reproducing
1. Run notebook 02 on a GPU runtime with ≥80 GB (or use an AWQ 4-bit teacher on 40 GB). Add `HF_TOKEN` to Colab Secrets.
2. Run notebook 03 to build the data files.
3. Run notebook 04 (A100 recommended; falls back to QLoRA on smaller GPUs). Accept Mistral's terms on Hugging Face first.
4. Run notebook 05 to evaluate on the internal test set.
5. Run notebook 06, train the two baselines with notebook 04, then run notebooks 07 and 08.

Paths assume Google Drive at `MyDrive/mistral_ft/`.

## Stack
Python · Hugging Face `transformers`, `peft`, `trl`, `datasets` · vLLM · PyTorch · Google Colab (A100 80GB) · Qwen3.6-27B · Mistral-7B-Instruct-v0.3
