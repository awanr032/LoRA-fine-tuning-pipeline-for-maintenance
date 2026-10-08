# Results registry: LLM distillation for fine-grained maintenance IE (MaintIE)

Single source of truth for every number in the paper. All values are from Colab runs on an A100 80GB.
Machine-readable copy: `all_results.csv`. Raw files: Google Drive `MyDrive/mistral_ft/paper_results/` (see manifest).

**Status:** single training run per model. Seed repeats (v7) and gold+distilled (v8) still to do.
**Frozen versions should not be edited**: add a new version instead.

## Version index

| Version | Content | Status |
|---|---|---|
| v1_teacher_pilots | Teacher configurations on 50 gold pilot texts | frozen |
| v2_teacher_labels | Silver labelling with the final teacher | frozen |
| v3_training_runs / v3_internal_eval | Student training and evaluation on internal 826-text test | frozen |
| v4_official_eval | Official MaintIE test (108 / clean 91) + published baselines | frozen |
| v5_controlled_comparisons | Paired tests: distilled vs gold, quantity vs quality | frozen (single seeds) |
| v6_error_analysis | Type-error kinds and attribution | frozen |
| v7_seeds | 2 extra seeds per model, mean ± sd | **to do** |
| v8_combined | Gold 860 + distilled 6,709 | optional |

## Data and splits

| Item | Count | Notes |
|---|---|---|
| Gold corpus | 1,076 texts | 224 entity types, 6 relation types, double-annotated by two experts |
| Silver corpus | 7,000 texts | coarse (5 classes); annotated by a model trained on fine-grained data, reviewed by one expert |
| Silver texts duplicating gold | 7 | removed before labelling |
| Prompt pool (gold) | 150 | teacher few-shot (20) + example terms; excluded from all evaluation |
| Internal validation | 100 | 50 pilot + 50 |
| Internal test | 826 | gold minus prompt pool and validation |
| Official split (seed 1337, 80/10/10) | 860 / 108 / 108 | from MaintIE `create_datasets.py` |
| Official test texts in prompt pool | 17 | hence "clean 91" |
| Test types unseen in distilled training | 17 types / 52 entities | internal test |

## v1: Teacher pilots (50 gold texts, entity spans given)

| Teacher setup | Exact type acc. | First two levels | Relation F1 | Failures |
|---|---|---|---|---|
| DeepSeek-flash, prompt v1 (6 ex), 1k tokens | 11.4% | 14.5% | 0.20 | 40/50 empty (reasoning used token budget) |
| DeepSeek-flash, prompt v1, 4k tokens | 51.8% | 61.4% | 0.69 | 12/50 truncated |
| Qwen3.6-27B, prompt v2 (20 ex + terms), thinking on | 88.0% | 92.8% | 0.85 | 2/50 |
| **Qwen3.6-27B, prompt v2, thinking off (used)** | **86.1%** | **91.6%** | **0.85** | 3/50 |

Caveat: rows 2→3 change teacher and prompt together.

## v2: Teacher labelling (Qwen3.6-27B, thinking off)

6,993 labelled; **6,709 passed validation (95.9%)**. Failures: 239 unknown type, ~50 coarse conflicts, 14 id mismatch,
12 invalid relation indices, 4 no entity list. ~128 output tokens/text; vLLM prefix-cache hit rate 94%.
Abandoned DeepSeek partial run: 1,633 labelled (1,443 passed), $2.42 ($0.0015/text), not used.

## v3: Student training

| Run | Train data | Epochs | Selection | Notes |
|---|---|---|---|---|
| distilled_qwen_nothink_v1 | 6,709 teacher labels | 2 (840 steps) | best val loss (internal val 100) | train 0.187 / val 0.271 |
| gold_official_v1 | 860 expert labels (official train) | max 5 | best val loss at step 100 (~2 epochs) | val 0.364 → 0.421 at end (overfitting, avoided) |
| distilled_860_v1 | 860 random teacher labels (seed 2026) | max 5 | best val loss (official dev) | log to add |

Shared settings: Mistral-7B-Instruct-v0.3, LoRA bf16, r=16, α=32, dropout 0.05, all attention+MLP projections,
lr 2e-4 cosine, warmup 3%, effective batch 16, max length 1024, seed 42.

## v3: Internal test (826 texts), distilled 6,709

| Metric | P | R | F1 (95% CI) |
|---|---|---|---|
| Entity exact (span + type) | 0.765 | 0.784 | **0.774** (0.757–0.791) |
| Entity, two levels | 0.828 | 0.848 | 0.838 (0.823–0.851) |
| Entity, top-level class | 0.933 | 0.956 | 0.944 (0.935–0.952) |
| Entity span only | 0.938 | 0.960 | 0.949 (0.941–0.957) |
| Relation, spans only | 0.806 | 0.837 | 0.821 (0.801–0.842) |
| Relation, with entity types | 0.532 | 0.553 | 0.542 (0.515–0.573) |

Invalid JSON 0/826. Exact F1 by class: Activity 0.896 (603), PhysicalObject 0.688 (1,513), Process 0.973 (111),
Property 0.964 (29), State 0.874 (342).
Difficulty diagnostic: 82 official texts F1 62.5 (5.4% unseen-type entities) vs 744 other texts 79.2 (1.6%).

## v4: Official MaintIE test

| System | Test | Entity micro F1 | Entity macro F1 | Rel. spans only | Rel. with types |
|---|---|---|---|---|---|
| **Distilled 6,709** | all 108 | **63.8** (58.9–68.7) | **42.9** | **80.2** (74.3–85.9) | **29.8** (23.1–37.1) |
| **Distilled 6,709** | clean 91 | 61.5 (56.2–66.9) | 42.9 | 80.3 (74.1–86.8) | 27.6 (19.5–36.3) |
| Gold 860 (same model) | all 108 | 58.0 | 30.2 | 74.9 | 25.5 |
| Gold 860 (same model) | clean 91 | 57.5 | 31.0 | 76.7 | 25.9 |
| Distilled 860 | all 108 | 55.2 | 25.9 | 74.1 | 17.0 |
| Distilled 860 | clean 91 | 54.2 | 27.9 | 72.9 | 15.8 |
| SpERT FG (published) | all 108 | 64.43 | 38.86 | 55.67 | 27.32 |
| SpERT CG+FG (published) | all 108 | 55.41 | 27.03 | 71.40 | 15.52 |
| REBEL CG+FG (published) | all 108 | n/a | n/a | 76.07 | 0.00 |

Note: MaintIE's written "strict/loose" definitions appear swapped relative to their numbers; we name metrics explicitly.

## v5: Paired comparisons (bootstrap, 2,000 resamples; diff = A − B)

**All 108**

| A vs B | Entity micro | Entity macro | Rel. spans | Rel. types |
|---|---|---|---|---|
| Distilled 860 vs Gold 860 (quality) | −2.8 (−7.0, +1.7) p=.218 | −4.3 (−8.1, +0.4) p=.080 | −0.8 p=.820 | **−8.5 (−15.5, −1.8) p=.007** |
| Distilled 6,709 vs Distilled 860 (quantity) | **+8.6 (+5.7, +11.5) p<.001** | **+17.0 (+11.2, +19.0) p<.001** | +6.2 p=.062 | **+12.8 p<.001** |
| Distilled 6,709 vs Gold 860 (main) | **+5.8 (+2.0, +9.9) p=.003** | **+12.7 (+6.7, +16.2) p<.001** | +5.3 p=.131 | +4.3 p=.219 |

**Clean 91**

| A vs B | Entity micro | Entity macro | Rel. spans | Rel. types |
|---|---|---|---|---|
| Distilled 860 vs Gold 860 | −3.3 (−7.6, +1.0) p=.124 | −3.1 (−7.9, +1.5) p=.190 | −3.8 p=.293 | **−10.1 (−17.6, −3.3) p=.002** |
| Distilled 6,709 vs Distilled 860 | **+7.3 p<.001** | **+15.0 p<.001** | **+7.4 p=.042** | **+11.8 p<.001** |
| Distilled 6,709 vs Gold 860 | +4.0 (−0.0, +8.3) p=.051 | **+11.9 (+4.8, +15.5) p<.001** | +3.6 p=.286 | +1.7 p=.629 |

Uncorrected p-values; consider Holm-Bonferroni. Earlier 5,000-resample run of the main comparison: all 108 p=.002, clean 91 p=.046.

## v6: Error analysis (internal test, distilled 6,709)

459 type errors on correctly found spans.
Kinds: different branch 281 (61%), sibling 141 (31%), too general/specific 25 (5%), different top-level 12 (3%).
Sources: **inherited from teacher 230 (50%)**, unseen word 148 (32%), mixed teacher labels 50 (11%), **genuine student error 31 (7%)**.
Predictions with non-existent types: 5. Method: exact lowercase word match, 50% majority rule (heuristic; audit a sample).
Top inherited conventions: "cracked" → FailedState (72/72; gold DegradedState), "shaft" → Guiding (55/55; gold Transforming),
"fit" → Assemble (81/83; gold Replace), "battery" → ElectrochemicalStoring (gold ChemicalToElectricalEnergyGenerating).

## Key findings (draft)

1. Without any human fine-grained labels, the distilled student matches published SpERT on entity micro F1 and exceeds it on macro F1 and span-level relations.
2. Same model and settings: 6,709 teacher labels beat 860 expert labels (entity micro +5.8, p=.003 on all 108; macro +12.7, p<.001).
3. Per example, expert labels are somewhat better (quantity-matched: typed relations −8.5, p=.007), so the gain comes from cheap scale.
4. Half of the student's type errors are inherited teacher conventions; only 7% are genuine learning failures, so label quality and coverage are the bottleneck.

## Limitations to state

Single seeds; small official test; silver spans from a model trained on gold; possible pretraining contamination (public data);
17 official test texts in prompt pool (clean-91 reported); unseen types; teacher/prompt confounded in pilot; heuristic error attribution.
