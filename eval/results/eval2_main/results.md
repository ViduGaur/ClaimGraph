# Held-out topics — gold mode: final

Gold labels: blinded model-assisted consensus labels with researcher adjudication.
181 pairs from 6 topics (19 sampled pairs without a usable gold label are excluded).
Gold relations: {'SUPPORTS': 28, 'EXTENDS': 25, 'QUALIFIES': 49, 'CONTRADICTS': 25, 'UNRELATED': 54}. Gold conflict verdicts: {'context_dependent': 25, 'genuine': 25}. Label sources: {'majority (pending adjudication)': 106, 'unanimous': 75}.

Brackets are 95% bootstrap intervals (1000 draws, items resampled within sampling strata). ‘–’ = undefined (for example a precision when nothing is asserted). An *abstention* is a face-value conflict the system declines to assert (relation CONTRADICTS flagged uncertain, verdict insufficient); it counts as wrong for accuracy and F1 and is never an asserted contradiction.

## Main comparison (evaluation sample)

| System | Asserted con. | Abstains | False con. /100 ↓ | Con-P | Con-R | Con-F1 | Macro-F1 | Acc. | Verdict acc. |
|---|---|---|---|---|---|---|---|---|---|
| Similarity only | 0 | 0 | 0.0 [0, 0] | – | 0.0 [0, 0] | 0.0 [0, 0] | 5.4 [4, 7] | 15.5 [11, 20] | 0.0 [0, 0] |
| NLI only | 49 | 0 | 22.1 [19, 25] | 18.4 [8, 29] | 36.0 [17, 55] | 24.3 [11, 36] | 14.9 [11, 19] | 27.6 [22, 33] | 18.0 [8, 29] |
| LLM direct (claims only) | 44 | 0 | 16.6 [12, 22] | 31.8 [19, 44] | 56.0 [38, 74] | 40.6 [25, 53] | 41.0 [35, 48] | 42.5 [36, 49] | 36.0 [24, 50] |
| LLM direct (+ passages) | 57 | 0 | 19.3 [14, 24] | 38.6 [27, 50] | 88.0 [73, 100] | 53.7 [40, 64] | 41.3 [35, 47] | 44.2 [38, 51] | 46.0 [32, 59] |
| ClaimGraph-R (rules, lexicon context) | 4 | 18 | 1.7 [0, 3] | 25.0 [0, 100] | 4.0 [0, 14] | 6.9 [0, 22] | 11.8 [9, 16] | 27.1 [22, 33] | 2.0 [0, 7] |
|   w/o context gate | 68 | 0 | 27.1 [23, 30] | 27.9 [19, 38] | 76.0 [61, 92] | 40.9 [29, 52] | 35.4 [30, 41] | 42.0 [35, 48] | 38.0 [26, 51] |
| ClaimGraph (full) | 7 | 16 | 2.8 [2, 4] | 28.6 [0, 57] | 8.0 [0, 18] | 12.5 [0, 28] | 39.0 [32, 45] | 47.0 [40, 53] | 40.0 [29, 52] |

## Per-class precision / recall / F1

| System | Suppo P/R/F1 | Exten P/R/F1 | Quali P/R/F1 | Contr P/R/F1 | Unrel P/R/F1 |
|---|---|---|---|---|---|
| Similarity only | 15.5 / 100.0 / 26.8 | – / 0.0 / 0.0 | – / 0.0 / 0.0 | – / 0.0 / 0.0 | – / 0.0 / 0.0 |
| NLI only | 14.3 / 3.6 / 5.7 | – / 0.0 / 0.0 | – / 0.0 / 0.0 | 18.4 / 36.0 / 24.3 | 32.0 / 74.1 / 44.7 |
| LLM direct (claims only) | 35.8 / 67.9 / 46.9 | 36.8 / 28.0 / 31.8 | 46.2 / 24.5 / 32.0 | 31.8 / 56.0 / 40.6 | 64.1 / 46.3 / 53.8 |
| LLM direct (+ passages) | 35.8 / 85.7 / 50.5 | 30.8 / 16.0 / 21.1 | 43.5 / 20.4 / 27.8 | 38.6 / 88.0 / 53.7 | 95.2 / 37.0 / 53.3 |
| ClaimGraph-R (rules, lexicon context) | 14.3 / 3.6 / 5.7 | – / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 25.0 / 4.0 / 6.9 | 31.5 / 87.0 / 46.3 |
| w/o context gate | 56.5 / 46.4 / 51.0 | 12.5 / 8.0 / 9.8 | 16.7 / 4.1 / 6.6 | 27.9 / 76.0 / 40.9 | 64.5 / 74.1 / 69.0 |
| ClaimGraph (full) | 56.5 / 46.4 / 51.0 | 12.5 / 8.0 / 9.8 | 49.1 / 57.1 / 52.8 | 28.6 / 8.0 / 12.5 | 64.5 / 74.1 / 69.0 |

## Ablations (absolute scores)

| System | Asserted con. | Abstains | False con. /100 ↓ | Con-P | Con-R | Con-F1 | Macro-F1 | Acc. | Verdict acc. |
|---|---|---|---|---|---|---|---|---|---|
| ClaimGraph (full) | 7 | 16 | 2.8 [2, 4] | 28.6 [0, 57] | 8.0 [0, 18] | 12.5 [0, 28] | 39.0 [32, 45] | 47.0 [40, 53] | 40.0 [29, 52] |
|   w/o context gate | 68 | 0 | 27.1 [23, 30] | 27.9 [19, 38] | 76.0 [61, 92] | 40.9 [29, 52] | 35.4 [30, 41] | 42.0 [35, 48] | 38.0 [26, 51] |
|   w/o stance | 5 | 10 | 2.8 [1, 5] | 0.0 [0, 0] | 0.0 [0, 0] | 0.0 [0, 0] | 19.1 [14, 25] | 32.0 [26, 39] | 10.0 [4, 18] |
|   w/o NLI | 7 | 16 | 2.8 [2, 4] | 28.6 [0, 57] | 8.0 [0, 18] | 12.5 [0, 28] | 38.0 [32, 44] | 46.4 [40, 52] | 40.0 [29, 52] |
|   w/o quote grounding | 4 | 15 | 1.7 [1, 3] | 25.0 [0, 75] | 4.0 [0, 12] | 6.9 [0, 20] | 39.1 [33, 46] | 47.5 [41, 54] | 40.0 [29, 52] |
| ClaimGraph-R (rules, lexicon context) | 4 | 18 | 1.7 [0, 3] | 25.0 [0, 100] | 4.0 [0, 14] | 6.9 [0, 22] | 11.8 [9, 16] | 27.1 [22, 33] | 2.0 [0, 7] |
| ClaimGraph-R w/o context gate | 25 | 0 | 11.0 [7, 15] | 20.0 [5, 36] | 20.0 [6, 35] | 20.0 [5, 33] | 14.4 [11, 18] | 29.3 [24, 35] | 10.0 [2, 19] |
| ClaimGraph-R w/o stance | 4 | 23 | 1.7 [0, 3] | 25.0 [0, 100] | 4.0 [0, 14] | 6.9 [0, 22] | 12.6 [9, 17] | 27.1 [22, 33] | 2.0 [0, 7] |

## Ablations: paired difference to the full system (percentage points; * = bootstrap p < 0.05)

| Variant | Δ false_con_per_100 | Δ con_f1 | Δ macro_f1 | Δ accuracy | Δ verdict_acc |
|---|---|---|---|---|---|
| w/o context gate | +24.3 [+21, +28]* | +28.4 [+10, +43]* | -3.6 [-8, +1] | -5.0 [-10, +1] | -2.0 [-21, +18] |
| w/o stance | +0.0 [-2, +3] | -12.5 [-28, +0] | -20.0 [-27, -14]* | -14.9 [-23, -7]* | -30.0 [-42, -19]* |
| w/o NLI | +0.0 [+0, +0] | +0.0 [+0, +0] | -1.0 [-3, +0] | -0.6 [-2, +0] | +0.0 [+0, +0] |
| w/o quote grounding | -1.1 [-2, +0] | -5.6 [-19, +1] | +0.1 [-4, +3] | +0.6 [-1, +3] | +0.0 [-6, +6] |
| ClaimGraph-R (rules, lexicon context) | -1.1 [-3, +1] | -5.6 [-25, +14] | -27.2 [-34, -20]* | -19.9 [-28, -13]* | -38.0 [-51, -26]* |

## Baselines: paired difference to the full system

| Variant | Δ false_con_per_100 | Δ con_f1 | Δ macro_f1 | Δ accuracy | Δ verdict_acc |
|---|---|---|---|---|---|
| NLI only | +19.3 [+17, +23]* | +11.8 [-12, +31] | -24.1 [-32, -17]* | -19.3 [-28, -11]* | -22.0 [-41, -2]* |
| LLM direct (claims only) | +13.8 [+9, +19]* | +28.1 [+4, +46]* | +2.0 [-7, +11] | -4.4 [-14, +4] | -4.0 [-23, +15] |
| LLM direct (+ passages) | +16.6 [+12, +22]* | +41.2 [+22, +57]* | +2.3 [-7, +11] | -2.8 [-12, +7] | +6.0 [-17, +27] |

## Is the gate just being conservative? Degenerate alternatives

| System | Asserted con. | Abstains | False con. /100 ↓ | Con-P | Con-R | Con-F1 | Macro-F1 | Acc. | Verdict acc. |
|---|---|---|---|---|---|---|---|---|---|
| ClaimGraph (full) | 7 | 16 | 2.8 [2, 4] | 28.6 [0, 57] | 8.0 [0, 18] | 12.5 [0, 28] | 39.0 [32, 45] | 47.0 [40, 53] | 40.0 [29, 52] |
|   gate: always context-dependent | 0 | 0 | 0.0 [0, 0] | – | 0.0 [0, 0] | 0.0 [0, 0] | 36.8 [31, 42] | 49.7 [43, 56] | 44.0 [32, 58] |
|   gate: always abstain | 0 | 68 | 0.0 [0, 0] | – | 0.0 [0, 0] | 0.0 [0, 0] | 27.3 [22, 32] | 31.5 [27, 36] | 0.0 [0, 0] |
| Constant: QUALIFIES | 0 | 0 | 0.0 [0, 0] | – | 0.0 [0, 0] | 0.0 [0, 0] | 8.5 [7, 10] | 27.1 [22, 33] | 50.0 [38, 63] |
| Constant: UNRELATED | 0 | 0 | 0.0 [0, 0] | – | 0.0 [0, 0] | 0.0 [0, 0] | 9.2 [8, 10] | 29.8 [24, 35] | 0.0 [0, 0] |

| Variant | Δ false_con_per_100 | Δ con_f1 | Δ macro_f1 | Δ accuracy | Δ verdict_acc |
|---|---|---|---|---|---|
| gate: always context-dependent | -2.8 [-4, -2]* | -12.5 [-28, +0] | -2.2 [-6, +1] | +2.8 [-1, +6] | +4.0 [-5, +14] |
| gate: always abstain | -2.8 [-4, -2]* | -12.5 [-28, +0] | -11.8 [-16, -8]* | -15.5 [-19, -12]* | -40.0 [-52, -29]* |
| Constant: QUALIFIES | -2.8 [-4, -2]* | -12.5 [-28, +0] | -30.5 [-37, -24]* | -19.9 [-28, -12]* | +10.0 [-2, +22] |
| Constant: UNRELATED | -2.8 [-4, -2]* | -12.5 [-28, +0] | -29.8 [-36, -23]* | -17.1 [-24, -10]* | -40.0 [-52, -29]* |

### Gate verdict against gold relation (sampled face-value conflicts of the full system)

| Gate verdict | n | SUPPORTS | EXTENDS | QUALIFIES | CONTRADICTS | UNRELATED |
|---|---|---|---|---|---|---|
| genuine | 7 | 1 | 0 | 3 | 2 | 1 |
| context_dependent | 45 | 3 | 2 | 26 | 7 | 7 |
| insufficient | 16 | 1 | 0 | 4 | 10 | 1 |

- conflicts the gate did not assert (context-dependent or insufficient) whose gold relation is indeed not CONTRADICTS: 44/61
- 'context-dependent' verdicts whose gold relation is QUALIFIES: 26/45
- 'genuine' verdicts whose gold relation is CONTRADICTS: 2/7

Share of gold contradictions by gate verdict: genuine 28.6%, context_dependent 15.6%, insufficient 62.5%. Fisher's exact test, context-dependent versus insufficient: p = 0.0007; context-dependent versus the other two verdicts: p = 0.0034.

## How abstentions are scored: sensitivity

| System | Abstentions counted as | Asserted con. | False con. /100 | Con-P | Con-R | Con-F1 | Macro-F1 | Acc. |
|---|---|---|---|---|---|---|---|---|
| ClaimGraph (full) | abstention (primary) | 7 | 2.8 | 28.6 | 8.0 | 12.5 | 39.0 | 47.0 |
| ClaimGraph (full) | CONTRADICTS | 23 | 6.1 | 52.2 | 48.0 | 50.0 | 46.5 | 52.5 |
| ClaimGraph (full) | QUALIFIES | 7 | 2.8 | 28.6 | 8.0 | 12.5 | 38.9 | 49.2 |
| ClaimGraph-R (rules, lexicon context) | abstention (primary) | 4 | 1.7 | 25.0 | 4.0 | 6.9 | 11.8 | 27.1 |
| ClaimGraph-R (rules, lexicon context) | CONTRADICTS | 22 | 9.9 | 18.2 | 16.0 | 17.0 | 13.8 | 28.7 |
| ClaimGraph-R (rules, lexicon context) | QUALIFIES | 4 | 1.7 | 25.0 | 4.0 | 6.9 | 16.4 | 31.5 |

### Relation-level reading: an abstention counted as the CONTRADICTS relation it is filed under

In the two-level representation an unasserted conflict has relation CONTRADICTS (flagged uncertain) and verdict insufficient. The tables above score it as an abstention, which was fixed as the primary reading before the gold labels existed. This table scores the relation label instead. The gold labels have no 'cannot tell' category, so the two readings differ a great deal; both are reported.

| System | Asserted con. | Abstains | False con. /100 ↓ | Con-P | Con-R | Con-F1 | Macro-F1 | Acc. | Verdict acc. |
|---|---|---|---|---|---|---|---|---|---|
| Similarity only | 0 | 0 | 0.0 [0, 0] | – | 0.0 [0, 0] | 0.0 [0, 0] | 5.4 [4, 7] | 15.5 [11, 20] | 0.0 [0, 0] |
| NLI only | 49 | 0 | 22.1 [19, 25] | 18.4 [8, 29] | 36.0 [17, 55] | 24.3 [11, 36] | 14.9 [11, 19] | 27.6 [22, 33] | 18.0 [8, 29] |
| LLM direct (claims only) | 44 | 0 | 16.6 [12, 22] | 31.8 [19, 44] | 56.0 [38, 74] | 40.6 [25, 53] | 41.0 [35, 48] | 42.5 [36, 49] | 36.0 [24, 50] |
| LLM direct (+ passages) | 57 | 0 | 19.3 [14, 24] | 38.6 [27, 50] | 88.0 [73, 100] | 53.7 [40, 64] | 41.3 [35, 47] | 44.2 [38, 51] | 46.0 [32, 59] |
| ClaimGraph-R (rules, lexicon context) | 22 | 0 | 9.9 [6, 14] | 18.2 [5, 35] | 16.0 [4, 31] | 17.0 [4, 30] | 13.8 [10, 18] | 28.7 [23, 35] | 2.0 [0, 7] |
|   w/o context gate | 68 | 0 | 27.1 [23, 30] | 27.9 [19, 38] | 76.0 [61, 92] | 40.9 [29, 52] | 35.4 [30, 41] | 42.0 [35, 48] | 38.0 [26, 51] |
| ClaimGraph (full) | 23 | 0 | 6.1 [3, 9] | 52.2 [30, 74] | 48.0 [32, 64] | 50.0 [32, 64] | 46.5 [40, 53] | 52.5 [46, 59] | 40.0 [29, 52] |
|   gate: always context-dependent | 0 | 0 | 0.0 [0, 0] | – | 0.0 [0, 0] | 0.0 [0, 0] | 36.8 [31, 42] | 49.7 [43, 56] | 44.0 [32, 58] |
|   gate: always abstain | 68 | 0 | 27.1 [23, 30] | 27.9 [19, 38] | 76.0 [61, 92] | 40.9 [29, 52] | 35.4 [30, 41] | 42.0 [35, 48] | 0.0 [0, 0] |

| Variant | Δ false_con_per_100 | Δ con_f1 | Δ macro_f1 | Δ accuracy | Δ verdict_acc |
|---|---|---|---|---|---|
| w/o context gate | +21.0 [+18, +23]* | -9.1 [-24, +6] | -11.1 [-15, -6]* | -10.5 [-16, -5]* | -2.0 [-21, +18] |
| gate: always context-dependent | -6.1 [-9, -3]* | -50.0 [-64, -32]* | -9.7 [-14, -5]* | -2.8 [-7, +2] | +4.0 [-5, +14] |
| gate: always abstain | +21.0 [+18, +23]* | -9.1 [-24, +6] | -11.1 [-15, -6]* | -10.5 [-16, -5]* | -40.0 [-52, -29]* |
| NLI only | +16.0 [+12, +19]* | -25.7 [-48, -2]* | -31.6 [-39, -24]* | -24.9 [-34, -17]* | -22.0 [-41, -2]* |
| LLM direct (claims only) | +10.5 [+6, +15]* | -9.4 [-31, +11] | -5.5 [-15, +4] | -9.9 [-19, -1]* | -4.0 [-23, +15] |
| LLM direct (+ passages) | +13.3 [+8, +18]* | +3.7 [-15, +24] | -5.2 [-15, +4] | -8.3 [-18, +1] | +6.0 [-17, +27] |

## Estimates for the whole candidate pool (design-weighted)

The sample over-represents conflicts. Weighting each item by stratum size / items scored estimates the same quantities over all candidate pairs the sample was drawn from. Strata with few sampled items make these noisy.

| System | False con. /100 pairs (pool) | Accuracy (pool) | Macro-F1 (pool) | Con-R (pool) |
|---|---|---|---|---|
| Similarity only | 0.0 [0, 0] | 13.3 [9, 19] | 4.7 | 0.0 |
| NLI only | 10.4 [9, 11] | 42.0 [33, 51] | 17.0 | 23.4 |
| LLM direct (claims only) | 10.2 [6, 14] | 45.9 [37, 55] | 40.3 | 55.6 |
| LLM direct (+ passages) | 12.9 [9, 17] | 43.1 [34, 53] | 39.7 | 78.7 |
| ClaimGraph-R (rules, lexicon context) | 0.9 [0, 2] | 43.5 [34, 53] | 14.6 | 1.5 |
|   w/o context gate | 8.5 [8, 9] | 52.4 [43, 62] | 32.5 | 34.0 |
| ClaimGraph (full) | 0.5 [0, 1] | 55.3 [46, 65] | 35.6 | 3.1 |

| Stratum | Pool | Scored |
|---|---|---|
| abstained | 17 | 16 |
| all_unrelated | 116 | 8 |
| asserted_contradiction | 8 | 7 |
| direct_contra_no_conflict | 192 | 20 |
| extends | 49 | 14 |
| gate_rejected | 80 | 24 |
| gate_rejected_nli_contra | 22 | 21 |
| nli_contra_no_conflict | 116 | 24 |
| others_relate | 448 | 19 |
| qualifies_partial | 12 | 11 |
| supports | 140 | 17 |

## Per topic (false con. /100 · macro-F1 · accuracy)

| Topic | N | gold con. | NLI only | LLM direct (+ passages) | w/o context gate | ClaimGraph (full) |
|---|---|---|---|---|---|---|
| adam_vs_sgd | 27 | 5 | 18.5 · 11.3 · 14.8 | 11.1 · 26.7 · 51.9 | 33.3 · 28.9 · 40.7 | 0.0 · 31.8 · 44.4 |
| icl_label_correctness | 38 | 5 | 18.4 · 13.0 · 18.4 | 26.3 · 26.6 · 28.9 | 36.8 · 25.9 · 21.1 | 2.6 · 29.9 · 39.5 |
| label_smoothing_distillation | 31 | 2 | 25.8 · 14.5 · 32.3 | 12.9 · 43.9 · 48.4 | 22.6 · 49.9 · 58.1 | 3.2 · 50.3 · 64.5 |
| large_batch_generalization | 37 | 3 | 32.4 · 13.6 · 29.7 | 24.3 · 34.2 · 40.5 | 29.7 · 25.7 · 32.4 | 0.0 · 30.3 · 40.5 |
| robustness_accuracy_tradeoff | 29 | 7 | 17.2 · 15.4 · 27.6 | 13.8 · 42.1 · 51.7 | 20.7 · 43.2 · 58.6 | 10.3 · 35.9 · 41.4 |
| transformers_time_series | 19 | 3 | 15.8 · 19.0 · 52.6 | 26.3 · 36.2 · 52.6 | 10.5 · 14.3 · 52.6 | 0.0 · 24.3 · 57.9 |

## NLI-only baseline across thresholds

| NLI threshold | Asserted con. | False con. /100 | Con-P | Con-R | Con-F1 | Macro-F1 |
|---|---|---|---|---|---|---|
| 0.3 | 62 | 28.2 | 17.7 | 44.0 | 25.3 | 14.5 |
| 0.5 | 49 | 22.1 | 18.4 | 36.0 | 24.3 | 14.9 |
| 0.7 | 31 | 12.7 | 25.8 | 32.0 | 28.6 | 15.0 |
| 0.9 | 22 | 9.4 | 22.7 | 20.0 | 21.3 | 13.7 |

## Grounding of the LLM context table (scored pairs)

```
{
 "values": 858,
 "grounded_local": 481,
 "grounded_paper": 336,
 "dropped": 41,
 "downgraded_immaterial": 43,
 "failures": 0,
 "pairs": 181,
 "dropped_rate": 0.0478
}
```

## Confusion — NLI only (rows = gold, columns = predicted)

| gold \ pred | SUPPORTS | EXTENDS | QUALIFIES | CONTRADICTS | UNRELATED | ABSTAIN |
|---|---|---|---|---|---|---|
| SUPPORTS | 1 | 0 | 0 | 7 | 20 | 0 |
| EXTENDS | 3 | 0 | 0 | 2 | 20 | 0 |
| QUALIFIES | 1 | 0 | 0 | 19 | 29 | 0 |
| CONTRADICTS | 0 | 0 | 0 | 9 | 16 | 0 |
| UNRELATED | 2 | 0 | 0 | 12 | 40 | 0 |

## Confusion — LLM direct (claims only) (rows = gold, columns = predicted)

| gold \ pred | SUPPORTS | EXTENDS | QUALIFIES | CONTRADICTS | UNRELATED | ABSTAIN |
|---|---|---|---|---|---|---|
| SUPPORTS | 19 | 1 | 2 | 4 | 2 | 0 |
| EXTENDS | 8 | 7 | 5 | 1 | 4 | 0 |
| QUALIFIES | 14 | 3 | 12 | 16 | 4 | 0 |
| CONTRADICTS | 3 | 2 | 2 | 14 | 4 | 0 |
| UNRELATED | 9 | 6 | 5 | 9 | 25 | 0 |

## Confusion — LLM direct (+ passages) (rows = gold, columns = predicted)

| gold \ pred | SUPPORTS | EXTENDS | QUALIFIES | CONTRADICTS | UNRELATED | ABSTAIN |
|---|---|---|---|---|---|---|
| SUPPORTS | 24 | 0 | 2 | 2 | 0 | 0 |
| EXTENDS | 19 | 4 | 2 | 0 | 0 | 0 |
| QUALIFIES | 14 | 1 | 10 | 23 | 1 | 0 |
| CONTRADICTS | 0 | 2 | 1 | 22 | 0 | 0 |
| UNRELATED | 10 | 6 | 8 | 10 | 20 | 0 |

## Confusion — ClaimGraph-R (rules, lexicon context) (rows = gold, columns = predicted)

| gold \ pred | SUPPORTS | EXTENDS | QUALIFIES | CONTRADICTS | UNRELATED | ABSTAIN |
|---|---|---|---|---|---|---|
| SUPPORTS | 1 | 0 | 0 | 0 | 23 | 4 |
| EXTENDS | 3 | 0 | 0 | 2 | 20 | 0 |
| QUALIFIES | 1 | 0 | 0 | 1 | 39 | 8 |
| CONTRADICTS | 0 | 0 | 1 | 1 | 20 | 3 |
| UNRELATED | 2 | 0 | 2 | 0 | 47 | 3 |

## Confusion — w/o context gate (rows = gold, columns = predicted)

| gold \ pred | SUPPORTS | EXTENDS | QUALIFIES | CONTRADICTS | UNRELATED | ABSTAIN |
|---|---|---|---|---|---|---|
| SUPPORTS | 13 | 9 | 0 | 5 | 1 | 0 |
| EXTENDS | 4 | 2 | 9 | 2 | 8 | 0 |
| QUALIFIES | 2 | 4 | 2 | 33 | 8 | 0 |
| CONTRADICTS | 0 | 0 | 1 | 19 | 5 | 0 |
| UNRELATED | 4 | 1 | 0 | 9 | 40 | 0 |

## Confusion — ClaimGraph (full) (rows = gold, columns = predicted)

| gold \ pred | SUPPORTS | EXTENDS | QUALIFIES | CONTRADICTS | UNRELATED | ABSTAIN |
|---|---|---|---|---|---|---|
| SUPPORTS | 13 | 9 | 3 | 1 | 1 | 1 |
| EXTENDS | 4 | 2 | 11 | 0 | 8 | 0 |
| QUALIFIES | 2 | 4 | 28 | 3 | 8 | 4 |
| CONTRADICTS | 0 | 0 | 8 | 2 | 5 | 10 |
| UNRELATED | 4 | 1 | 7 | 1 | 40 | 1 |

## Confusion — gate: always context-dependent (rows = gold, columns = predicted)

| gold \ pred | SUPPORTS | EXTENDS | QUALIFIES | CONTRADICTS | UNRELATED | ABSTAIN |
|---|---|---|---|---|---|---|
| SUPPORTS | 13 | 9 | 5 | 0 | 1 | 0 |
| EXTENDS | 4 | 2 | 11 | 0 | 8 | 0 |
| QUALIFIES | 2 | 4 | 35 | 0 | 8 | 0 |
| CONTRADICTS | 0 | 0 | 20 | 0 | 5 | 0 |
| UNRELATED | 4 | 1 | 9 | 0 | 40 | 0 |

## Annotation agreement

```
{
 "method": "blinded model-assisted consensus labels with researcher adjudication",
 "items": 200,
 "complete": 200,
 "agreement_levels": {
  "majority": 106,
  "unanimous": 75,
  "split": 19
 },
 "final_sources": {
  "majority (pending adjudication)": 106,
  "unanimous": 75,
  "unresolved": 19
 },
 "label_distribution": {
  "gemini": {
   "QUALIFIES": 60,
   "UNRELATED": 42,
   "CONTRADICTS": 12,
   "SUPPORTS": 21,
   "EXTENDS": 65
  },
  "qwen": {
   "CONTRADICTS": 32,
   "UNRELATED": 76,
   "QUALIFIES": 53,
   "SUPPORTS": 33,
   "EXTENDS": 6
  },
  "haiku": {
   "QUALIFIES": 42,
   "UNRELATED": 48,
   "CONTRADICTS": 46,
   "SUPPORTS": 34,
   "EXTENDS": 30
  }
 },
 "final_distribution": {
  "QUALIFIES": 49,
  "UNRELATED": 54,
  "CONTRADICTS": 25,
  "SUPPORTS": 28,
  "null": 19,
  "EXTENDS": 25
 },
 "fleiss_kappa": 0.428,
 "pairwise": {
  "gemini-qwen": {
   "agreement": 0.535,
   "kappa": 0.422
  },
  "gemini-haiku": {
   "agreement": 0.515,
   "kappa": 0.398
  },
  "qwen-haiku": {
   "agreement": 0.605,
   "kappa": 0.496
  }
 },
 "spot_check": {
  "checked": 0,
  "researcher_agrees": 0
 },
 "adjudication": {
  "flagged": 125,
  "adjudicated": 0,
  "researcher_kept_majority": 0
 }
}
```
