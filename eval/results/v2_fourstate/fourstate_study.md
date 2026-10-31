# Four-state development study: opposition, evidence sufficiency, comparability

**Development data, not a paper result.** Batch `c4`: 196 pairs labelled blind by three annotator models (Gemini 3 Flash, Qwen 3.7 Plus, Claude Haiku 4.5) under the four-state protocol; consensus = majority per dimension; unadjudicated. The pairs come from the eight topics already used for development.

## Summary

1. **States.** Of 196 pairs: none 103, genuine 4, context-dep. 61, insufficient 15, no consensus 13. Consensus GENUINE contradictions are rare (4); the annotators differ widely on that state (gemini 14, qwen 8, haiku 1). Of the 25 forced-choice 'contradictions' of the first protocol, 17 are now context-dependent, 4 insufficient, 3 genuine.
2. **Agreement is moderate on stance and weak below it**: Fleiss κ 0.53 for stance, 0.27 for sufficiency and 0.14 for the final status on opposed pairs.
3. **Opposition is the weakest step of the system.** The annotator majority sees opposition in only 81 of the 140 pairs the system reads as opposed.
4. **Table A (sufficiency).** The E = 0 rule finds 0 of 10 INSUFFICIENT pairs with 11 predictions (precision 0.0, recall 0.0; AUC 0.40 [0.22, 0.58]).
5. **Table B (comparability, the P3 hypothesis) is underpowered**: 4 GENUINE against 57 CONTEXT_DEPENDENT pairs. Yes/no rules do badly (the V1 gate calls 6 genuine, 1 correctly). Graded scores have point estimates above chance but intervals that reach it: uniform AUC 0.74 [0.47, 0.97], V1 hand weights 0.79 [0.50, 1.00], fitted role weights 0.69 [0.32, 0.96]; shuffled-factor p = 0.102 (uniform). Against 'at least one annotator says GENUINE' (12 positives) hand weights give 0.69 [0.47, 0.87]. Fitted role weights never beat uniform or hand weights.
6. **'Any shared context'** no longer predicts contradictions: GENUINE 2/21 with E = 0 against 2/109 with E > 0 (p = 0.12). Nor does it mark the pairs annotators call INSUFFICIENT: 0/21 against 10/109 (p = 0.36). The earlier finding was an artefact of the forced-choice labels.
7. **Explanatory value.** On 51 context-dependent pairs the system's material mismatches share at least one factor with the annotators' in 39.2% of pairs (chance 31.5%, p = 0.034); precision 27.8%, recall 34.5%.

## 1. Sample and reference states

| Stratum | Pool | Annotated | none | genuine | context-dep. | insufficient | no consensus |
|---|---|---|---|---|---|---|---|
| the system reads the pair as a face-value conflict (all such pairs, every topic) | 140 | 140 | 59 | 4 | 57 | 10 | 10 |
| not system-opposed, but an annotator model saw opposition under the first protocol (all such pairs of batch eval2) | 26 | 26 | 14 | 0 | 4 | 5 | 3 |
| not system-opposed and no annotator model saw opposition under the first protocol (random sample of batch eval2) | 97 | 30 | 30 | 0 | 0 | 0 | 0 |
| **all** |  | 196 | 103 | 4 | 61 | 15 | 13 |

Final status as each annotator model labelled it (all pairs):

| Annotator | none | genuine | context-dep. | insufficient |
|---|---|---|---|---|
| gemini | 96 | 14 | 80 | 6 |
| qwen | 113 | 8 | 60 | 15 |
| haiku | 83 | 1 | 49 | 63 |

Forced-choice relation under the first protocol against the four-state status (pairs labelled under both):

| First protocol | none | genuine | context-dep. | insufficient | no consensus |
|---|---|---|---|---|---|
| CONTRADICTS | 1 | 3 | 17 | 4 | 0 |
| QUALIFIES | 18 | 0 | 17 | 6 | 4 |
| SUPPORTS | 8 | 0 | 1 | 0 | 0 |
| EXTENDS | 11 | 0 | 0 | 0 | 0 |
| UNRELATED | 27 | 0 | 0 | 0 | 1 |

## 2. Agreement between the annotator models, per dimension

| Dimension | Pairs | Fleiss κ | Unanimous | 2–1 | 3-way | Pairwise Cohen κ (gemini–qwen, gemini–haiku, qwen–haiku) |
|---|---|---|---|---|---|---|
| A. face-value stance | 196 | 0.53 | 126 | 70 | 0 | 0.64, 0.45, 0.49 |
| B. evidence sufficiency (all pairs) | 196 | 0.33 | 99 | 97 | 0 | 0.37, 0.24, 0.44 |
| B. evidence sufficiency (pairs with majority stance oppose) | 93 | 0.27 | 54 | 39 | 0 | 0.34, 0.17, 0.44 |
| C. comparability (pairs where all three answered it) | 41 | 0.35 | 33 | 8 | 0 | 0.55, 0.19, 0.30 |
| D. final status (all pairs) | 196 | 0.41 | 94 | 89 | 13 | 0.55, 0.32, 0.42 |
| D. final status (pairs with majority stance oppose) | 93 | 0.14 | 34 | 46 | 13 | 0.20, 0.10, 0.24 |

E. explanatory factors, on the 61 pairs at least two annotators call context-dependent: mean pairwise Jaccard overlap of the named factor sets 0.35. Named by at least two annotators: experimental conditions 22, training & design 16, methodology 10, population 8, model / method 2, dataset 1, task 1, evaluation metric 1.

Every annotator's final status equals the status its own answers to A–C imply (196/196, 196/196, 195/196).

## 3. Opposition: the system's stance against the reference stance

| System | Pairs | Reference: oppose | Reference: not oppose |
|---|---|---|---|
| reads the pair as opposed | 140 | 81 | 59 |
| does not | 56 | 12 | 44 |

Of the 140 pairs the system reads as opposed, the annotator majority agrees for 81 (58%). The non-opposed pairs were sampled, not exhaustive, so the second row is not a recall estimate.

## 4. TABLE A — evidence sufficiency

System rule (fixed in advance, nothing fitted): *insufficient* when no scope or measurement factor is stated by both papers (E = 0). Positive class: INSUFFICIENT.

| Population and reference | Pairs | Reference insufficient | System says insufficient | Both | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Opposed by reference and by system; final status INSUFFICIENT vs GENUINE/CONTEXT_DEPENDENT | 71 | 10 | 11 | 0 | 0.0 [0, 0] | 0.0 [0, 0] | 0.0 [0, 0] | 70.4 [59, 80] |
| Opposed by reference (any system stance); same target | 80 | 15 | 18 | 5 | 27.8 [7, 50] | 33.3 [10, 58] | 30.3 [8, 50] | 71.2 [61, 81] |
| All annotated pairs; dimension B alone (sufficiency majority) | 196 | 86 | 59 | 34 | 57.6 [44, 70] | 39.5 [30, 50] | 46.9 [36, 57] | 60.7 [54, 67] |
| Annotator gemini alone: pairs it calls opposed, its own sufficiency answer | 85 | 4 | 13 | 0 | 0.0 [0, 0] | 0.0 [0, 0] | 0.0 [0, 0] | 80.0 [72, 88] |
| Annotator qwen alone: pairs it calls opposed, its own sufficiency answer | 72 | 9 | 11 | 0 | 0.0 [0, 0] | 0.0 [0, 0] | 0.0 [0, 0] | 72.2 [61, 82] |
| Annotator haiku alone: pairs it calls opposed, its own sufficiency answer | 94 | 47 | 11 | 3 | 27.3 [0, 57] | 6.4 [0, 14] | 10.3 [0, 21] | 44.7 [35, 54] |

Ranking by the amount of shared context instead of the E = 0 cut (AUC; 0.50 = no signal):

| Population | AUC, fewer shared context factors → insufficient [95% CI] | AUC, fewer shared factors of any role | Fisher p for the E = 0 rule |
|---|---|---|---|
| Opposed by reference and by system; final status INSUFFICIENT vs GENUINE/CONTEXT_DEPENDENT | 0.40 [0.22, 0.58] | 0.28 | 0.343 |
| Opposed by reference (any system stance); same target | 0.56 [0.38, 0.74] | 0.50 | 0.308 |
| All annotated pairs; dimension B alone (sufficiency majority) | 0.57 [0.49, 0.64] | 0.57 | 0.012 |
| Annotator gemini alone: pairs it calls opposed, its own sufficiency answer | 0.48 [0.33, 0.60] | 0.41 | 1.000 |
| Annotator qwen alone: pairs it calls opposed, its own sufficiency answer | 0.43 [0.22, 0.62] | 0.38 | 0.337 |
| Annotator haiku alone: pairs it calls opposed, its own sufficiency answer | 0.34 [0.23, 0.45] | 0.26 | 0.198 |

| Shared context E (system) on pairs opposed by reference and system | Pairs | genuine | context-dep. | insufficient |
|---|---|---|---|---|
| 0 | 11 | 2 | 9 | 0 |
| 0.5–1 | 34 | 2 | 26 | 6 |
| 1.5–2 | 21 | 0 | 19 | 2 |
| > 2 | 5 | 0 | 3 | 2 |

## 5. TABLE B — context comparability (the P3 hypothesis)

Population: pairs opposed by reference and by system whose reference status is GENUINE or CONTEXT_DEPENDENT (sufficient evidence): **61 pairs, 4 GENUINE, 57 CONTEXT_DEPENDENT.** Insufficient pairs are excluded. Positive class: GENUINE.

| Rule (nothing fitted) | Pairs | Reference GENUINE | System says GENUINE | Both | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Always CONTEXT_DEPENDENT | 61 | 4 | 0 | 0 | – | 0.0 [0, 0] | 0.0 [0, 0] | 93.4 [87, 98] |
| V1 hard gate: no material mismatch in any factor | 61 | 4 | 6 | 1 | 16.7 [0, 50] | 25.0 [0, 100] | 20.0 [0, 55] | 86.9 [79, 95] |
| Typed rule: no material scope/measurement mismatch | 61 | 4 | 11 | 1 | 9.1 [0, 30] | 25.0 [0, 100] | 13.3 [0, 38] | 78.7 [69, 89] |
| Typed rule, any stated difference decides | 61 | 4 | 1 | 0 | 0.0 [0, 0] | 0.0 [0, 0] | 0.0 [0, 0] | 91.8 [84, 98] |
| No shared context at all (E = 0) | 61 | 4 | 11 | 2 | 18.2 [0, 45] | 50.0 [0, 100] | 26.7 [0, 56] | 82.0 [72, 92] |

Graded models, leave-one-topic-out. AUC compares pairs within a topic; log-loss skill is relative to the constant model.

| Model | Fitted numbers | AUC [95% CI] | Log-loss | Skill vs constant |
|---|---|---|---|---|
| Constant (stance alone: every opposed pair alike) | 1 | 0.50 | 0.255 | +0.0% |
| Sufficiency only: is any context factor stated by both papers? | 2 | 0.60 [0.36, 0.96] | 0.256 | -0.4% |
| Uniform weights, scope + measurement mismatches only | 2 | 0.74 [0.47, 0.97] | 0.226 | +11.5% |
| Uniform weights, agreement and mismatch | 3 | 0.87 [0.68, 1.00] | 0.210 | +17.6% |
| V1 hand weights, mismatches only | 2 | 0.79 [0.50, 1.00] | 0.215 | +15.9% |
| P3: role weights (scope, measurement, strength; intervention outside) | 5 | 0.69 [0.32, 0.96] | 0.233 | +8.7% |
| Role weights incl. intervention | 6 | 0.79 [0.45, 1.00] | 0.226 | +11.4% |
| Twelve free mismatch weights + agreement (exploratory) | 14 | 0.81 [0.51, 1.00] | 0.242 | +5.3% |

Shuffled-factor control (1000 shuffles within topic): Uniform weights, scope + measurement mismatches only: real AUC 0.74, shuffled 0.51 [0.26, 0.83], p = 0.102; P3: role weights (scope, measurement, strength; intervention outside): real AUC 0.69, shuffled 0.51 [0.22, 0.84], p = 0.157.

Because consensus GENUINE is rare, the same question against softer targets on pairs opposed by both the annotator majority and the system (exploratory). AUC within topic of each score for GENUINE; the first, second and last three columns need no fitting, the role-weight column is fitted leave-one-topic-out.

| Target | Pairs | GENUINE | Uniform: fewer context mismatches | V1 hand weights | P3 role weights (fitted) | V1 gate (yes/no) | Typed rule (yes/no) | No shared context (E = 0) | Uniform, named values compared in code |
|---|---|---|---|---|---|---|---|---|---|
| consensus status (Table B population) | 61 | 4 | 0.74 [0.49, 1.00] | 0.79 [0.51, 1.00] | 0.69 [0.33, 0.97] | 0.60 [0.39, 0.87] | 0.52 [0.30, 0.81] | 0.60 [0.35, 0.98] | 0.79 [0.58, 0.97] |
| at least one annotator says GENUINE (vs none) | 81 | 12 | 0.61 [0.41, 0.82] | 0.69 [0.47, 0.87] | 0.49 [0.38, 0.62] | 0.53 [0.43, 0.67] | 0.62 [0.47, 0.81] | 0.49 [0.37, 0.64] | 0.60 [0.41, 0.80] |
| annotator gemini alone: GENUINE vs CONTEXT_DEPENDENT | 74 | 10 | 0.72 [0.53, 0.87] | 0.79 [0.64, 0.95] | 0.53 [0.35, 0.72] | 0.56 [0.42, 0.75] | 0.69 [0.52, 0.88] | 0.50 [0.37, 0.67] | 0.69 [0.50, 0.86] |
| annotator qwen alone: GENUINE vs CONTEXT_DEPENDENT | 62 | 6 | 0.56 [0.27, 0.87] | 0.61 [0.30, 0.92] | 0.49 [0.18, 0.80] | 0.54 [0.40, 0.74] | 0.47 [0.30, 0.68] | 0.55 [0.37, 0.80] | 0.60 [0.30, 0.89] |
| annotator haiku alone: GENUINE vs CONTEXT_DEPENDENT | 44 | 1 | 0.90 [0.70, 1.00] | 1.00 [1.00, 1.00] | 0.90 [0.67, 1.00] | 0.50 [0.50, 0.50] | 0.30 [0.00, 0.50] | 0.90 [0.67, 1.00] | 0.90 [0.68, 1.00] |

Graded score against the yes/no gate (AUC, V1 hand weights minus V1 gate; paired bootstrap):

| Target | Δ AUC [95% CI] | share of resamples with Δ ≤ 0 |
|---|---|---|
| consensus status (Table B population) | +0.18 [-0.14, +0.53] | 0.159 |
| at least one annotator says GENUINE (vs none) | +0.16 [-0.06, +0.36] | 0.078 |
| annotator gemini alone: GENUINE vs CONTEXT_DEPENDENT | +0.23 [+0.03, +0.42] | 0.015 |
| annotator qwen alone: GENUINE vs CONTEXT_DEPENDENT | +0.07 [-0.21, +0.39] | 0.313 |
| annotator haiku alone: GENUINE vs CONTEXT_DEPENDENT | +0.50 [+0.50, +0.50] | 0.000 |

Deterministic comparison of stated values (names as sets, quantities as numbers) against the status the LLM gave, over the factor entries of system-opposed pairs where code can compare:

| Factor | LLM status | code: same | code: overlap | code: different |
|---|---|---|---|---|
| dataset | overlap | 0 | 2 | 4 |
| dataset | different (immaterial) | 0 | 0 | 1 |
| dataset | different | 0 | 2 | 21 |
| data scale | different | 0 | 0 | 1 |
| evaluation metric | same | 4 | 0 | 9 |
| evaluation metric | overlap | 1 | 1 | 19 |
| evaluation metric | different (immaterial) | 0 | 0 | 4 |
| evaluation metric | different | 0 | 0 | 27 |
| model / method | same | 8 | 0 | 4 |
| model / method | overlap | 2 | 0 | 18 |
| model / method | different (immaterial) | 0 | 0 | 11 |
| model / method | different | 0 | 0 | 52 |
| baseline | same | 5 | 0 | 5 |
| baseline | overlap | 0 | 0 | 10 |
| baseline | different (immaterial) | 0 | 0 | 11 |
| baseline | different | 0 | 0 | 12 |

Free-text factors (experimental conditions, training & design, population, task, methodology) cannot be compared in code and keep the LLM's status.

## 6. End-to-end four-state decision

All 183 annotated pairs with a consensus status (none 103, genuine 4, context-dep. 61, insufficient 15). The sample over-represents opposed pairs, so these are not prevalence-weighted figures. Macro-F1 averages the four states; with so few GENUINE pairs it is unstable. † a direct-LLM QUALIFIES is read as CONTEXT_DEPENDENT although that prompt did not require face-value opposition.

| System | Accuracy [95% CI] | Macro-F1 [95% CI] | GENUINE: asserted / correct | CONTEXT-DEP. F1 | INSUFFICIENT F1 | NONE F1 |
|---|---|---|---|---|---|---|
| Never opposed (constant NONE) | 56.3 [49, 63] | 18.0 [16, 20] | 0 / 0 | 0.0 | 0.0 | 72.0 |
| NLI only (contradiction → GENUINE) | 46.4 [39, 54] | 18.3 [16, 22] | 43 / 1 | 0.0 | 0.0 | 69.1 |
| LLM direct, claims only † | 40.4 [33, 48] | 22.1 [18, 27] | 61 / 3 | 13.8 | 0.0 | 65.3 |
| LLM direct, + passages † | 38.3 [32, 45] | 22.7 [19, 27] | 79 / 4 | 13.0 | 0.0 | 68.2 |
| Stance only: every opposed pair GENUINE | 26.2 [20, 33] | 15.6 [13, 19] | 130 / 4 | 0.0 | 0.0 | 56.4 |
| Stance + always CONTEXT_DEPENDENT | 55.2 [48, 63] | 29.0 [25, 33] | 0 / 0 | 59.7 | 0.0 | 56.4 |
| Stance + always INSUFFICIENT | 29.5 [23, 37] | 17.6 [14, 21] | 0 / 0 | 0.0 | 13.8 | 56.4 |
| Stance + sufficiency rule, then always CONTEXT_DEPENDENT | 50.3 [43, 57] | 28.2 [24, 33] | 0 / 0 | 56.5 | 0.0 | 56.4 |
| V1 hard gate (simple four-state rule) | 48.1 [40, 55] | 31.3 [25, 40] | 9 / 1 | 53.4 | 0.0 | 56.4 |
| Typed four-state rule (scope/measurement mismatches only) | 45.4 [38, 53] | 29.0 [25, 34] | 26 / 1 | 52.8 | 0.0 | 56.4 |
| Typed rule, any stated difference decides | 49.7 [43, 57] | 28.2 [24, 33] | 3 / 0 | 56.3 | 0.0 | 56.4 |
| P3 [uniform_ctx], parameters frozen from the forced-label study | 50.3 [43, 57] | 28.2 [24, 33] | 0 / 0 | 56.5 | 0.0 | 56.4 |
| P3 [hand], parameters frozen from the forced-label study | 49.7 [42, 57] | 28.0 [24, 33] | 1 / 0 | 55.6 | 0.0 | 56.4 |
| P3 [role], parameters frozen from the forced-label study | 50.3 [43, 58] | 28.2 [25, 33] | 0 / 0 | 56.5 | 0.0 | 56.4 |

Decision layer alone: the 71 pairs opposed by both reference and system (genuine 4, context-dep. 57, insufficient 10), three states.

| Decision rule | Accuracy [95% CI] | GENUINE P / R | CONTEXT-DEP. P / R | INSUFFICIENT P / R |
|---|---|---|---|---|
| every opposed pair GENUINE | 5.6 [1, 11] | 5.6 / 100.0 | – / 0.0 | – / 0.0 |
| always CONTEXT_DEPENDENT | 80.3 [72, 89] | – / 0.0 | 80.3 / 100.0 | – / 0.0 |
| always INSUFFICIENT | 14.1 [7, 23] | – / 0.0 | – / 0.0 | 14.1 / 100.0 |
| sufficiency rule, then always CONTEXT_DEPENDENT | 67.6 [58, 77] | – / 0.0 | 80.0 / 84.2 | 0.0 / 0.0 |
| V1 hard gate (simple four-state rule) | 62.0 [51, 73] | 14.3 / 25.0 | 81.1 / 75.4 | 0.0 / 0.0 |
| Typed four-state rule (scope/measurement mismatches only) | 54.9 [42, 66] | 7.1 / 25.0 | 82.6 / 66.7 | 0.0 / 0.0 |
| Typed rule, any stated difference decides | 66.2 [56, 77] | 0.0 / 0.0 | 79.7 / 82.5 | 0.0 / 0.0 |
| P3 [uniform_ctx], parameters frozen from the forced-label study | 67.6 [56, 79] | – / 0.0 | 80.0 / 84.2 | 0.0 / 0.0 |
| P3 [hand], parameters frozen from the forced-label study | 66.2 [55, 77] | 0.0 / 0.0 | 79.7 / 82.5 | 0.0 / 0.0 |
| P3 [role], parameters frozen from the forced-label study | 67.6 [55, 77] | – / 0.0 | 80.0 / 84.2 | 0.0 / 0.0 |

Confusion of the V1 hard gate on all scored pairs (rows: reference, columns: system):

|  | none | genuine | context-dep. | insufficient |
|---|---|---|---|---|
| none | 44 | 2 | 47 | 10 |
| genuine | 0 | 1 | 1 | 2 |
| context-dep. | 4 | 5 | 43 | 9 |
| insufficient | 5 | 1 | 9 | 0 |

## 7. The 'any shared context' finding, retested

Under the forced-choice labels, opposed pairs with no shared context were 'contradictions' 10/16 times against 9/52. The same split against the four-state labels (pairs the system reads as opposed, with a consensus status):

| Shared context (system) | Pairs | none | genuine | context-dep. | insufficient | old protocol: CONTRADICTS / labelled |
|---|---|---|---|---|---|---|
| E = 0 | 21 | 10 | 2 | 9 | 0 | 10 / 16 |
| E > 0 | 109 | 49 | 2 | 48 | 10 | 9 / 49 |

Fisher's exact test, E = 0 versus E > 0: GENUINE p = 0.122; INSUFFICIENT p = 0.364.

## 8. Is the factor table explanatory? System mismatches against the factors annotators name

Pairs opposed by the system whose reference status is CONTEXT_DEPENDENT. Reference set: factors named by at least two annotators (pairs where no factor reaches two votes are scored against the union and reported separately). System set: factors with a material mismatch. Chance: the same system sets reassigned among these pairs within a topic.

| Reference set | System set | Pairs | Precision | Recall | Mean Jaccard | Pairs with ≥ 1 shared factor | Chance: ≥ 1 shared [95%] | p |
|---|---|---|---|---|---|---|---|---|
| named by ≥ 2 annotators | material mismatch (status different) | 51 | 27.8 | 34.5 | 0.24 | 39.2 | 31.5 [24, 39] | 0.034 |
| named by ≥ 2 annotators | any stated difference | 51 | 18.9 | 36.2 | 0.18 | 41.2 | 33.8 [27, 41] | 0.029 |
| named by ≥ 2 annotators | always 'experimental conditions' (most frequent) | 51 | 41.2 | 36.2 | 0.38 | 41.2 | – | – |
| named by any annotator | material mismatch (status different) | 57 | 43.0 | 26.2 | 0.23 | 50.9 | 45.6 [39, 53] | 0.105 |
| named by any annotator | any stated difference | 57 | 32.5 | 30.8 | 0.20 | 56.1 | 53.3 [47, 60] | 0.291 |
| named by any annotator | always 'experimental conditions' (most frequent) | 57 | 68.4 | 30.0 | 0.34 | 68.4 | – | – |

For scale: two annotator models name factor sets with mean Jaccard overlap 0.35.

Per factor, over the 57 context-dependent pairs:

| Factor | Role | System: material mismatch | Annotators: named by ≥ 2 | Both | Named by any annotator |
|---|---|---|---|---|---|
| dataset | scope | 11 | 1 | 1 | 4 |
| data scale | scope | 1 | 0 | 0 | 2 |
| population | scope | 0 | 6 | 0 | 15 |
| task | scope | 0 | 1 | 0 | 6 |
| experimental conditions | scope | 22 | 21 | 12 | 39 |
| training & design | scope | 9 | 16 | 6 | 23 |
| evaluation metric | measurement | 12 | 1 | 0 | 4 |
| methodology | measurement | 0 | 10 | 0 | 17 |
| sample size | strength | 0 | 0 | 0 | 3 |
| time period | strength | 0 | 0 | 0 | 1 |
| model / method | intervention | 21 | 2 | 1 | 13 |
| baseline | intervention | 3 | 0 | 0 | 3 |

## 9. Factor and role ablations

Table A is the only place the factor evidence shows a measurable signal, so the ablation is run there: AUC of 'less shared context → insufficient' on pairs opposed by reference and system, with one factor or role left out of the count.

| Count of shared factors | AUC | Δ vs scope + measurement |
|---|---|---|
| scope + measurement (the rule) | 0.40 | – |
| all twelve factors | 0.32 | -0.08 |
| scope only | 0.35 | -0.05 |
| measurement only | 0.52 | +0.12 |
| intervention only (method, baseline) | 0.35 | -0.05 |
| scope + measurement without dataset | 0.46 | +0.06 |
| scope + measurement without data scale | 0.39 | -0.00 |
| scope + measurement without population | 0.38 | -0.01 |
| scope + measurement without task | 0.42 | +0.02 |
| scope + measurement without experimental conditions | 0.41 | +0.02 |
| scope + measurement without training & design | 0.36 | -0.04 |
| scope + measurement without evaluation metric | 0.35 | -0.05 |
| scope + measurement, paper-level statements count fully | 0.34 | -0.05 |

Factors never stated by both papers in any system-opposed pair: sample size, time period.
