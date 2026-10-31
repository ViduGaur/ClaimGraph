# Layer study on the C4 four-state labels (E1-E3)

196 pairs from 90 paper pairs and 8 topics; 183 have a consensus status (none 103, genuine 4, context-dep. 61, insufficient 15). Reference: blinded model-assisted consensus labels, unadjudicated. Development data: every number below was computed on topics that have been looked at, and nothing here is a held-out result. Intervals are 95% bootstrap intervals over paper pairs. With 4 GENUINE pairs no system can be validated on that class.

## E1. Opposition detection

Target: the annotators' majority reading of face-value opposition (dimension A), all 196 pairs, 93 opposed. Precision, recall and F1 weight the subsampled stratum by 97/30. "Flagged / right" are raw counts.

| Detector | Flagged / right | Precision | Recall | F1 | Topics where precision ≥ current | GENUINE pairs kept (of 4) |
|---|---|---|---|---|---|---|
| Current trigger (stance oppose, or partial + NLI) | 140 / 81 | 57.9 [45, 68] | 87.1 [78, 93] | 69.5 [58, 77] | 8 of 8 | 4 |
| Pair-analysis stance = oppose | 137 / 81 | 59.1 [46, 69] | 87.1 [78, 93] | 70.4 [59, 78] | 8 of 8 | 4 |
| Relation call with passages = CONTRADICTS | 85 / 66 | 68.6 [52, 82] | 71.0 [55, 83] | 69.8 [56, 79] | 6 of 8 | 4 |
| Relation call, claims only = CONTRADICTS | 63 / 44 | 63.1 [47, 78] | 47.3 [36, 61] | 54.1 [43, 64] | 6 of 8 | 3 |
| NLI only | 44 / 25 | 41.9 [24, 61] | 26.9 [17, 40] | 32.8 [20, 45] | 4 of 8 | 1 |
| Both agree: stance oppose AND relation call | 69 / 61 | 88.4 [79, 95] | 65.6 [49, 78] | 75.3 [63, 84] | 7 of 8 | 4 |
| Either: stance oppose OR relation call | 153 / 86 | 52.4 [40, 63] | 92.5 [85, 98] | 66.9 [55, 76] | 5 of 8 | 4 |
| At least 2 of 3 readings (stance, two relation calls) | 89 / 70 | 76.7 [65, 85] | 75.3 [63, 86] | 76.0 [65, 83] | 8 of 8 | 4 |
| All 3 readings | 36 / 32 | 88.9 [77, 98] | 34.4 [24, 46] | 49.6 [38, 61] | 6 of 8 | 3 |
| Calibrated: pair-analysis stance only, p ≥ 0.5 | 137 / 81 | 59.1 [46, 69] | 87.1 [78, 93] | 70.4 [59, 78] | 8 of 8 | 4 |
| Calibrated: NLI scores only (continuous), p ≥ 0.5 | 43 / 20 | 31.7 [16, 50] | 21.5 [11, 38] | 25.6 [14, 39] | 3 of 8 | 1 |
| Calibrated: stance + NLI scores, p ≥ 0.5 | 115 / 64 | 55.7 [43, 67] | 68.8 [57, 82] | 61.5 [52, 70] | 7 of 8 | 2 |
| Calibrated: stance + relation call, p ≥ 0.5 | 81 / 64 | 71.2 [54, 84] | 68.8 [54, 82] | 70.0 [56, 79] | 6 of 8 | 4 |
| Calibrated: stance + both relation calls, p ≥ 0.5 | 82 / 65 | 77.2 [63, 86] | 69.9 [55, 83] | 73.3 [61, 81] | 8 of 8 | 4 |
| Calibrated: stance + both relation calls + NLI scores, p ≥ 0.5 | 86 / 65 | 70.1 [54, 82] | 69.9 [55, 83] | 70.0 [56, 79] | 8 of 8 | 4 |

| Calibrated score (leave-one-topic-out) | AUC pooled | AUC within topic | Log-loss | Coefficients on all pairs (intercept first) |
|---|---|---|---|---|
| Calibrated: pair-analysis stance only | 0.56 [0.47, 0.67] | 0.68 | 0.653 | -1.18, +1.50, -0.30 |
| Calibrated: NLI scores only (continuous) | 0.41 [0.32, 0.55] | 0.54 | 0.712 | -0.22, +0.50, -0.22 |
| Calibrated: stance + NLI scores | 0.62 [0.54, 0.72] | 0.72 | 0.654 | -1.40, +1.56, -0.36, +0.74, -0.01 |
| Calibrated: stance + relation call | 0.78 [0.68, 0.86] | 0.83 | 0.527 | -1.99, +1.38, +2.05 |
| Calibrated: stance + both relation calls | 0.80 [0.71, 0.88] | 0.84 | 0.520 | -2.13, +1.32, +1.91, +0.78 |
| Calibrated: stance + both relation calls + NLI scores | 0.81 [0.72, 0.88] | 0.85 | 0.524 | -2.21, +1.35, +1.91, +0.72, +0.33, -0.05 |

Constant log-loss 0.692. A calibrated score is asked to do one thing here: beat the rules above it out of fold.

Paired differences (percentage points, same bootstrap samples):

| Compared | Δ precision | Δ recall | Δ F1 |
|---|---|---|---|
| Both agree − current | +30.5 [+21, +42] | -21.5 [-32, -12] | +5.8 [-3, +15] |
| Relation call − current | +10.8 [-2, +25] | -16.1 [-28, -5] | +0.3 [-10, +10] |
| Both agree − relation call | +19.8 [+9, +33] | -5.4 [-11, -1] | +5.5 [-0, +12] |
| 2 of 3 − relation call | +8.1 [-3, +20] | +4.3 [-2, +12] | +6.2 [-0, +14] |
| 2 of 3 − current | +18.9 [+10, +28] | -11.8 [-21, -2] | +6.5 [-1, +15] |
| Calibrated (stance + both calls) − current | +19.3 [+9, +30] | -17.2 [-29, -6] | +3.8 [-5, +13] |
| Calibrated (stance + both calls) − relation call | +8.5 [-2, +20] | -1.1 [-5, +3] | +3.6 [-2, +10] |
| Calibrated (stance + both calls) − both agree | -11.2 [-23, -4] | +4.3 [+1, +10] | -2.0 [-7, +3] |
| Calibrated with NLI − without | -7.0 [-16, -1] | +0.0 [+0, +0] | -3.3 [-8, -0] |

Where the 140 → 81 gap comes from: stance oppose: 137 pairs, 56 not confirmed; partial + NLI: 3 pairs, 3 not confirmed. Of the 59 unconfirmed pairs, 32 are unanimous "not opposed". The system misses 12 opposed pairs; its stance for them: {'unrelated': 10, 'agree': 1, 'partial': 1}.

| Signal | AUC, all 196 pairs | AUC, 140 system-opposed pairs |
|---|---|---|
| rel_ctx | 0.76 [0.69, 0.83] | 0.80 [0.73, 0.85] |
| rel_claims | 0.64 [0.58, 0.72] | 0.62 [0.55, 0.71] |
| nli_con | 0.53 [0.43, 0.64] | 0.53 [0.44, 0.65] |
| nli_con_max | 0.53 [0.43, 0.65] | 0.53 [0.44, 0.66] |
| nli_ent | 0.53 [0.45, 0.61] | 0.50 [0.40, 0.59] |
| sim | 0.49 [0.39, 0.59] | 0.50 [0.40, 0.61] |

End to end, four states, 183 pairs with a consensus status. The sample holds every system-opposed pair but only 30 of 97 pairs without any opposition signal, so plain accuracy understates every system and the constant; the weighted column corrects for that within the labelled pool.

| System | Accuracy | Accuracy, stratum-weighted | Macro-F1 | GENUINE said / right | CONTEXT_DEP. said / right | INSUFFICIENT said / right | NONE said / right |
|---|---|---|---|---|---|---|---|
| Constant NONE | 56.3 [47, 68] | 68.0 | 18.0 [16, 25] | 0 / 0 | 0 / 0 | 0 / 0 | 183 / 103 |
| Current trigger (stance oppose, or partial + NLI) → always CONTEXT_DEPENDENT | 55.2 [46, 64] | 67.2 | 29.0 [25, 37] | 0 / 0 | 130 / 57 | 0 / 0 | 53 / 44 |
| Current trigger (stance oppose, or partial + NLI) → V1 hard gate | 48.1 [39, 57] | 62.0 | 31.3 [24, 38] | 9 / 1 | 100 / 43 | 21 / 0 | 53 / 44 |
| Relation call with passages = CONTRADICTS → always CONTEXT_DEPENDENT | 70.5 [62, 78] | 73.9 | 36.4 [32, 47] | 0 / 0 | 79 / 45 | 0 / 0 | 104 / 84 |
| Relation call with passages = CONTRADICTS → V1 hard gate | 67.2 [60, 75] | 71.5 | 45.6 [36, 54] | 7 / 1 | 49 / 34 | 23 / 4 | 104 / 84 |
| Both agree: stance oppose AND relation call → always CONTEXT_DEPENDENT | 76.5 [70, 83] | 82.8 | 39.4 [36, 51] | 0 / 0 | 64 / 45 | 0 / 0 | 119 / 95 |
| Both agree: stance oppose AND relation call → V1 hard gate | 71.0 [63, 79] | 78.8 | 42.0 [34, 48] | 6 / 1 | 48 / 34 | 10 / 0 | 119 / 95 |
| At least 2 of 3 readings (stance, two relation calls) → always CONTEXT_DEPENDENT | 73.8 [66, 81] | 79.9 | 38.4 [35, 50] | 0 / 0 | 84 / 51 | 0 / 0 | 99 / 84 |
| At least 2 of 3 readings (stance, two relation calls) → V1 hard gate | 69.4 [61, 78] | 76.7 | 45.9 [37, 57] | 8 / 1 | 60 / 39 | 16 / 3 | 99 / 84 |
| Calibrated: stance + both relation calls → always CONTEXT_DEPENDENT | 72.1 [65, 79] | 78.7 | 37.2 [33, 48] | 0 / 0 | 77 / 46 | 0 / 0 | 106 / 86 |
| Calibrated: stance + both relation calls → V1 hard gate | 67.8 [60, 76] | 75.5 | 44.5 [35, 54] | 8 / 1 | 53 / 34 | 16 / 3 | 106 / 86 |

Error decomposition: one stage at a time is replaced by the reference answer (where the reference defines one; the V1 gate supplies the system's sufficiency and comparability answers).

| Opposition | Sufficiency | Comparability | Accuracy | Macro-F1 |
|---|---|---|---|---|
| system | system | system | 48.1 [39, 57] | 31.3 [24, 38] |
| **reference** | system | system | 84.2 [77, 91] | 56.5 [48, 69] |
| system | **reference** | system | 58.5 [50, 66] | 48.5 [36, 59] |
| system | system | **reference** | 51.4 [41, 61] | 39.8 [25, 51] |
| system | **reference** | **reference** | 62.8 [54, 71] | 65.7 [50, 76] |
| **reference** | **reference** | system | 95.6 [92, 99] | 78.3 [73, 99] |
| **reference** | system | **reference** | 87.4 [81, 93] | 67.7 [49, 80] |
| **reference** | system | always context-dep. | 86.3 [80, 93] | 52.9 [46, 70] |
| system | system | always context-dep. | 50.3 [41, 59] | 28.2 [23, 36] |

Relation and conflict status kept apart: each cached reading against the C4 status (raw counts; C4 has no five-way relation labels, so the relation itself cannot be scored).

| Reading | none | genuine | context-dep. | insufficient | no consensus |
|---|---|---|---|---|---|
| stance: agree | 12 | 0 | 1 | 0 | 0 |
| stance: oppose | 56 | 4 | 57 | 10 | 10 |
| stance: partial | 6 | 0 | 1 | 0 | 0 |
| stance: unrelated | 29 | 0 | 2 | 5 | 3 |
| relation call: CONTRADICTS | 19 | 4 | 45 | 11 | 6 |
| relation call: EXTENDS | 16 | 0 | 3 | 0 | 0 |
| relation call: QUALIFIES | 24 | 0 | 6 | 1 | 3 |
| relation call: SUPPORTS | 36 | 0 | 7 | 3 | 3 |
| relation call: UNRELATED | 8 | 0 | 0 | 0 | 1 |

## E2. Factor layer and evidence sufficiency

### (a) Explanatory match on the 61 CONTEXT_DEPENDENT pairs

System set: the factors a representation marks as differing. Reference set: factors named by at least one annotator (and, in the last column, by at least two). Hit = the two sets share a factor.

| Representation (factor level) | Mean set size | Hit | Precision | Recall | Jaccard | Shuffle p (hit) | Hit, ≥2 annotators |
|---|---|---|---|---|---|---|---|
| LLM status, LLM materiality (current) | 1.4 | 49.2 [33, 69] | 42.2 [29, 56] | 25.7 [15, 41] | 0.22 | 0.056 | 34.4 [20, 52] |
| LLM status, no materiality flag | 1.5 | 50.8 [34, 71] | 39.4 [28, 51] | 27.2 [17, 43] | 0.20 | 0.099 | 36.1 [22, 54] |
| LLM status, any stated difference (incl. partial) | 2.1 | 54.1 [40, 73] | 32.0 [24, 42] | 30.1 [20, 46] | 0.19 | 0.224 | 36.1 [22, 54] |
| Names and numbers compared in code | 2.0 | 50.8 [34, 71] | 31.9 [22, 43] | 27.9 [17, 44] | 0.19 | 0.214 | 36.1 [22, 54] |
| Every factor both papers state | 2.5 | 55.7 [41, 74] | 29.8 [22, 40] | 33.1 [22, 48] | 0.20 | 0.313 | 37.7 [24, 55] |
| Baseline: always "experimental conditions" | 1.0 | 65.6 [49, 84] | 65.6 [49, 84] | 29.4 [22, 40] | 0.34 | – | 36.1 [18, 57] |

| Representation (role level) | Mean set size | Hit | Precision | Recall | Jaccard | Shuffle p (hit) | Hit, ≥2 annotators |
|---|---|---|---|---|---|---|---|
| LLM status, LLM materiality (current) | 1.1 | 63.9 [49, 82] | 63.2 [53, 78] | 42.2 [31, 59] | 0.43 | 0.604 | 52.5 [38, 69] |
| LLM status, no materiality flag | 1.2 | 65.6 [53, 82] | 61.3 [51, 75] | 45.1 [34, 63] | 0.42 | 0.503 | 54.1 [40, 71] |
| LLM status, any stated difference (incl. partial) | 1.6 | 80.3 [69, 91] | 60.4 [51, 73] | 56.9 [48, 70] | 0.48 | 0.199 | 60.7 [48, 75] |
| Names and numbers compared in code | 1.6 | 75.4 [61, 88] | 57.3 [48, 69] | 53.9 [44, 68] | 0.43 | 0.243 | 57.4 [43, 73] |
| Every factor both papers state | 1.7 | 82.0 [72, 92] | 58.5 [50, 68] | 60.8 [51, 74] | 0.47 | 0.213 | 63.9 [52, 76] |
| Baseline: always "scope" | 1.0 | 98.4 [94, 100] | 98.4 [94, 100] | 58.8 [51, 70] | 0.67 | – | 77.0 [60, 90] |

Annotators against each other on the same pairs (both called it CONTEXT_DEPENDENT): mean pairwise Jaccard 0.51 at factor level, 0.67 at role level. Pairs in which a role is named by at least one annotator: scope 60, measurement 20, intervention 18, strength 4. Factors: experimental conditions 40, training & design 24, population 17, methodology 17, model / method 15, task 6, evaluation metric 4, dataset 4, baseline 3, sample size 3, data scale 2, time period 1.

### (b) Signal beyond the semantic layer

Opposition (all pairs), leave-one-topic-out. Factor features: context mismatches, context agreements, an intervention mismatch, number of factors stated by both.

| Score | AUC | Log-loss |
|---|---|---|
| Semantic readings only (stance + two relation calls) | 0.80 [0.71, 0.88] | 0.520 |
| Semantic + factor features | 0.80 [0.71, 0.87] | 0.532 |
| Factor features only | 0.53 [0.44, 0.62] | 0.701 |

Adding the factor features changes the AUC by -0.002 [-0.015, +0.012].

GENUINE against CONTEXT_DEPENDENT. With this few genuine pairs no row can be told from another; the table is descriptive.

| Uniform score (agreements − mismatches over context factors) | AUC, consensus GENUINE (4 of 65) | AUC, any annotator GENUINE (14 of 93 opposed pairs) |
|---|---|---|
| LLM materiality flag used (current) | 0.69 [0.56, 0.86] | 0.58 [0.42, 0.74] |
| No materiality flag | 0.69 [0.56, 0.86] | 0.59 [0.43, 0.74] |
| Names and numbers compared in code | 0.72 [0.59, 0.88] | 0.58 [0.43, 0.73] |
| Mismatches in role "scope" only | 0.73 [0.59, 0.90] | 0.65 [0.46, 0.79] |
| Mismatches in role "measurement" only | 0.58 [0.46, 0.72] | 0.45 [0.32, 0.61] |
| Mismatches in role "strength" only | 0.50 [0.50, 0.50] | 0.50 [0.50, 0.50] |
| Mismatches in role "intervention" only | 0.68 [0.52, 0.85] | 0.54 [0.40, 0.66] |

### (c) Evidence sufficiency

Each definition is fixed in advance and nothing is fitted. An AUC above 0.5 means that pairs with less of that evidence are the ones annotators call insufficient.

| Definition (higher = more evidence) | AUC, status INSUFFICIENT (15 of 80 opposed pairs) | AUC, any annotator "insufficient" (45 of 93) | AUC, dimension B, all pairs (86 of 196) |
|---|---|---|---|
| Current count: scope + measurement factors stated by both (paper-level half) | 0.56 [0.34, 0.84] | 0.42 [0.32, 0.57] | 0.57 [0.46, 0.68] |
| All twelve factors stated by both | 0.50 [0.27, 0.83] | 0.37 [0.27, 0.51] | 0.57 [0.46, 0.69] |
| Stated by both at claim level only (scope + measurement) | 0.56 [0.42, 0.75] | 0.52 [0.43, 0.63] | 0.54 [0.47, 0.62] |
| Number of roles with a factor stated by both (0-4) | 0.50 [0.28, 0.81] | 0.41 [0.30, 0.55] | 0.57 [0.46, 0.69] |
| Coverage of task + intervention + outcome measure (0-3) | 0.51 [0.29, 0.81] | 0.37 [0.24, 0.52] | 0.55 [0.45, 0.67] |
| Fewer factors stated by one paper only | 0.33 [0.24, 0.45] | 0.43 [0.34, 0.53] | 0.39 [0.32, 0.46] |
| Share of stated factors that both papers state | 0.51 [0.34, 0.79] | 0.45 [0.36, 0.57] | 0.56 [0.47, 0.66] |
| A difference was found (evidence of non-comparability is evidence) | 0.59 [0.45, 0.82] | 0.51 [0.42, 0.63] | 0.61 [0.53, 0.69] |
| Factors on which the papers agree | 0.65 [0.58, 0.73] | 0.47 [0.37, 0.59] | 0.55 [0.48, 0.61] |
| Stated values in total (either paper) | 0.55 [0.31, 0.87] | 0.40 [0.29, 0.53] | 0.60 [0.49, 0.72] |
| Baseline: length of the shorter claim | 0.43 [0.25, 0.74] | 0.48 [0.34, 0.67] | 0.54 [0.45, 0.64] |
| Baseline: claim similarity | 0.75 [0.58, 0.90] | 0.71 [0.59, 0.81] | 0.61 [0.51, 0.70] |

The target itself is unstable: on the 93 pairs opposed by majority, the three annotators call 4 (gemini), 14 (qwen), 41 (haiku) pairs INSUFFICIENT.

### (d) Factor and role ablations

| Count of factors stated by both, restricted to | AUC, status INSUFFICIENT | AUC, any annotator "insufficient" |
|---|---|---|
| all twelve factors | 0.50 [0.27, 0.83] | 0.37 [0.27, 0.51] |
| role scope only | 0.49 [0.30, 0.77] | 0.41 [0.32, 0.57] |
| role measurement only | 0.56 [0.40, 0.74] | 0.40 [0.27, 0.53] |
| role strength only | 0.50 [0.50, 0.50] | 0.50 [0.50, 0.50] |
| role intervention only | 0.51 [0.37, 0.70] | 0.46 [0.34, 0.58] |
| without role scope | 0.54 [0.37, 0.78] | 0.41 [0.29, 0.54] |
| without role measurement | 0.48 [0.28, 0.79] | 0.39 [0.29, 0.53] |
| without role strength | 0.50 [0.27, 0.83] | 0.37 [0.27, 0.51] |
| without role intervention | 0.53 [0.30, 0.83] | 0.39 [0.28, 0.54] |
| without "dataset" | 0.53 [0.30, 0.85] | 0.38 [0.28, 0.53] |
| without "data scale" | 0.49 [0.27, 0.83] | 0.37 [0.27, 0.50] |
| without "population" | 0.49 [0.26, 0.83] | 0.38 [0.28, 0.52] |
| without "task" | 0.51 [0.31, 0.81] | 0.41 [0.30, 0.54] |
| without "experimental conditions" | 0.52 [0.32, 0.81] | 0.38 [0.27, 0.51] |
| without "training & design" | 0.47 [0.25, 0.82] | 0.35 [0.26, 0.49] |
| without "evaluation metric" | 0.48 [0.28, 0.79] | 0.39 [0.29, 0.53] |
| without "methodology" | 0.50 [0.27, 0.83] | 0.37 [0.27, 0.51] |
| without "sample size" | 0.50 [0.27, 0.83] | 0.37 [0.27, 0.51] |
| without "time period" | 0.50 [0.27, 0.83] | 0.37 [0.27, 0.51] |
| without "model / method" | 0.54 [0.32, 0.85] | 0.38 [0.28, 0.53] |
| without "baseline" | 0.50 [0.26, 0.83] | 0.38 [0.27, 0.53] |

Pairs in which both papers state a factor: dataset 37, data scale 3, population 14, task 54, experimental conditions 77, training & design 16, evaluation metric 74, methodology 3, sample size 0, time period 1, model / method 110, baseline 48. Never stated by both: sample size.

## E3. Conditional-prediction frames

Extraction: 165 claims, one call each; 165 valid frames. Effects: positive 96, negative 33, mixed 15, not_stated 13, none 8. The effect quote is found in the claim sentence for 148 of 152 frames with an effect. 58 frames are about a method the paper itself proposes; 55 claims restrict themselves to conditions. Conditions: 230 of 241 values survive the quote check (claim 160, passage 36, paper 34). Both frames exist for 196 of 196 pairs.

| Frame component | AUC, all 196 pairs | AUC, 140 system-opposed pairs |
|---|---|---|
| both_effect | 0.40 [0.32, 0.49] | 0.39 [0.30, 0.49] |
| incompatible | 0.48 [0.41, 0.56] | 0.50 [0.42, 0.59] |
| new_mismatch (reversed) | 0.59 [0.52, 0.66] | 0.59 [0.50, 0.68] |
| s_intervention | 0.46 [0.37, 0.56] | 0.43 [0.30, 0.56] |
| s_outcome | 0.50 [0.41, 0.59] | 0.45 [0.34, 0.58] |

Effect relation of the two frames against the reference (opposed / not opposed): agree: 10 / 28; effect_vs_none: 12 / 6; mixed: 20 / 22; no effect stated: 28 / 11; opposite: 23 / 36.

Opposition detection from frames (target and weighting as in E1):

| Detector | Flagged / right | Precision | Recall | F1 |
|---|---|---|---|---|
| Current trigger | 140 / 81 | 57.9 [45, 68] | 87.1 [78, 93] | 69.5 [58, 77] |
| E1: both readings agree | 69 / 61 | 88.4 [79, 95] | 65.6 [49, 78] | 75.3 [63, 84] |
| Frames, no similarity threshold: both state an effect, effects incompatible, no unnamed new method | 52 / 29 | 41.5 [25, 59] | 31.2 [20, 43] | 35.6 [23, 47] |
| Frames, same rule with a similarity threshold chosen leave-one-topic-out | 51 / 28 | 40.7 [24, 58] | 30.1 [19, 42] | 34.6 [22, 46] |
| Frames, calibrated score (leave-one-topic-out), p ≥ 0.5 | 62 / 34 | 39.3 [22, 59] | 36.6 [23, 52] | 37.9 [23, 52] |
| Current trigger unless the frames veto (no effect stated, unnamed new method, or same effect) | 61 / 39 | 63.9 [49, 77] | 41.9 [28, 55] | 50.6 [36, 62] |
| E1 both-agree unless the frames veto | 32 / 29 | 90.6 [77, 100] | 31.2 [20, 43] | 46.4 [32, 59] |
| Calibrated: stance + relation calls + frame features, p ≥ 0.5 | 84 / 66 | 74.6 [60, 86] | 71.0 [56, 84] | 72.7 [60, 82] |
| POST HOC — frames, no threshold, interventions always aligned directly (no role swap) | 46 / 26 | 47.3 [29, 67] | 28.0 [17, 39] | 35.2 [22, 47] |
| POST HOC — as above, and "it depends" against a definite effect also counts as opposed | 79 / 44 | 45.4 [30, 61] | 47.3 [33, 60] | 46.3 [32, 58] |

Calibrated scores, out of fold: frames only AUC 0.61 [0.50, 0.72]; semantic readings only 0.80 [0.71, 0.88]; both 0.83 [0.75, 0.90]; adding the frame features changes the AUC by +0.027 [-0.012, +0.067]. Similarity thresholds chosen per held-out topic: [0.5, 0.55]. Alignment used: {'direct': 105, 'b_comparator': 46, 'a_comparator': 45}. The two rows marked POST HOC were added after the pre-specified rows had been read, to see whether the role-swap rule or the "mixed" category explains the failure; they are not evidence for a design.

What the frames say about the system's errors (no similarity threshold):

| Frame verdict | False opposed (system opposed, reference not) | True opposed (both) | Missed by the system (reference opposed) |
|---|---|---|---|
| a claim says "it depends" | 12 | 16 | 2 |
| a claim states no effect | 7 | 27 | 1 |
| new method not named by the other claim | 16 | 6 | 2 |
| same effect | 14 | 9 | 1 |
| kept: effect vs none | 1 | 9 | 3 |
| kept: opposite effects | 9 | 14 | 3 |
| total | 59 | 81 | 12 |

The same local entailment model (OnnxNLI) on three representations of the claims:

| Entailment model input (mean contradiction score) | AUC, all 196 pairs | AUC, 140 system-opposed pairs |
|---|---|---|
| Raw claim sentences (current) | 0.53 [0.43, 0.64] | 0.53 [0.44, 0.65] |
| Cached one-clause conclusions of the pair analysis | 0.69 [0.60, 0.77] | 0.66 [0.55, 0.77] |
| Frames rendered as one sentence each | 0.55 [0.47, 0.63] | 0.54 [0.45, 0.64] |

Conclusions minus raw sentences: +0.16 [+0.04, +0.28]. The conclusions were written by the pair analysis, which had read both claims, so this is not an independent reading: added to the calibrated semantic score (stance + two relation calls) it changes the out-of-fold AUC by +0.033 [+0.005, +0.067].

End to end, four states, 183 pairs with a consensus status:

| System | Accuracy | Accuracy, stratum-weighted | Macro-F1 | GENUINE said / right | CONTEXT_DEP. said / right | INSUFFICIENT said / right | NONE said / right |
|---|---|---|---|---|---|---|---|
| Constant NONE | 56.3 [47, 68] | 68.0 | 18.0 [16, 25] | 0 / 0 | 0 / 0 | 0 / 0 | 183 / 103 |
| Current ClaimGraph (trigger → V1 gate) | 48.1 [39, 57] | 62.0 | 31.3 [24, 38] | 9 / 1 | 100 / 43 | 21 / 0 | 53 / 44 |
| Frames end to end (threshold leave-one-topic-out) | 49.7 [41, 60] | 56.1 | 27.0 [21, 38] | 1 / 0 | 30 / 7 | 16 / 4 | 136 / 80 |
| Frames end to end (no similarity threshold) | 50.3 [42, 61] | 56.5 | 28.5 [22, 39] | 1 / 0 | 30 / 7 | 17 / 5 | 135 / 80 |
| E1 both-agree → frame conditions | 68.9 [62, 77] | 77.2 | 39.3 [34, 51] | 2 / 0 | 39 / 28 | 23 / 3 | 119 / 95 |
| E1 both-agree unless frames veto → frame conditions | 60.7 [50, 73] | 71.2 | 28.9 [23, 41] | 1 / 0 | 16 / 9 | 13 / 2 | 153 / 100 |
| E1 both-agree unless frames veto → always CONTEXT_DEPENDENT | 64.5 [55, 76] | 74.0 | 29.4 [24, 41] | 0 / 0 | 30 / 18 | 0 / 0 | 153 / 100 |
| E1 both-agree unless frames veto → V1 gate | 61.7 [52, 73] | 72.0 | 33.2 [26, 38] | 4 / 1 | 20 / 12 | 6 / 0 | 153 / 100 |

| Decision layer on the reference-opposed pairs | Accuracy | GENUINE said / right | INSUFFICIENT said / right |
|---|---|---|---|
| Always CONTEXT_DEPENDENT | 76.2 [63, 88] | 0 / 0 | 0 / 0 |
| V1 gate on the pair-analysis table | 63.7 [51, 76] | 7 / 1 | 18 / 5 |
| Frame conditions | 61.3 [49, 75] | 4 / 0 | 27 / 7 |

INSUFFICIENT against the frame conditions on the same 80 pairs (AUC): fewer shared conditions 0.52 [0.36, 0.71]; more one-sided conditions 0.61 [0.48, 0.73]; neither claim conditional 0.63 [0.51, 0.81].

| Frame-based factor set on CONTEXT_DEPENDENT pairs | Mean size | Hit | Precision | Recall |
|---|---|---|---|---|
| Conditions both claims state and that differ | 0.4 | 21.3 [12, 33] | 59.1 [39, 81] | 9.6 [5, 16] |
| Conditions quoted from a claim sentence itself | 1.7 | 78.7 [66, 90] | 53.8 [44, 67] | 41.2 [32, 51] |
| Pair-analysis table, current (for comparison) | 1.4 | 49.2 [33, 68] | 42.2 [30, 56] | 25.7 [15, 39] |
| Baseline: always "experimental conditions" | 1.0 | 65.6 [48, 82] | 65.6 [48, 82] | 29.4 [21, 39] |

Claim-sentence conditions minus the pair table: precision +11.7 [-2, +25], recall +15.4 [+3, +27] percentage points.

