# Fresh-topic validation of the frozen E1 opposition detectors (batch f1)

Design, estimators, criteria and verdict rule were fixed in `eval/annotation/f1/PREREGISTRATION.md` before any pair was labelled. Reference: blinded model-assisted consensus labels under the C4 protocol, unadjudicated. Intervals: 95% bootstrap over paper pairs.

## A. Dataset

600 candidate pairs from 3 fresh topics; 84 sampled, 84 with three labels (252 labels), from 41 paper pairs. Reference: 36 opposed at face value, 48 not. Status: CONTEXT_DEPENDENT 28, INSUFFICIENT_EVIDENCE 6, NONE 48, None 2.

| Stratum | Definition | Pool | Labelled | Weight | Opposed by reference |
|---|---|---|---|---|---|
| A | pair-analysis stance = oppose and relation call with passages = CONTRADICTS (the E1 both-agree set) | 28 | 28 | 1.00 | 19 |
| B | the current trigger fires, not in A | 24 | 24 | 1.00 | 8 |
| C | the current trigger does not fire; both relation calls say CONTRADICTS | 43 | 13 | 3.31 | 7 |
| D | the current trigger does not fire; exactly one relation call says CONTRADICTS | 128 | 12 | 10.67 | 1 |
| E | no LLM reading signals opposition | 377 | 7 | 53.86 | 1 |

Candidate pool by topic: attention_explanation: {'E': 135, 'D': 25, 'C': 10, 'B': 16, 'A': 14}; batchnorm_covariate_shift: {'C': 18, 'E': 103, 'D': 65, 'A': 9, 'B': 5}; lottery_ticket_pruning: {'E': 139, 'D': 38, 'C': 15, 'A': 5, 'B': 3}.

## B. Frozen E1 validation

Precision is design-weighted (an estimate for the whole candidate pool). Recall (C4-comparable) is taken as C4 took it: over the labelled sample, which holds the trigger-fired pairs and a sample of the others. Full-pool recall weights every stratum; C4 never measured it, and it is imprecise because stratum E is sampled thinly.

| Detector | Flagged | True positives | Precision | Recall (C4-comparable) | F1 (C4-comparable) | Recall, full pool | Kept of the trigger's true positives | Δ precision vs current |
|---|---|---|---|---|---|---|---|---|
| Current trigger (stance oppose, or partial + NLI) | 52 | 27 | 51.9 [33, 70] | 75.0 [57, 91] | 61.4 [44, 75] | 23.5 [10, 70] | 100.0 [100, 100] | +0.0 [+0, +0] |
| Relation call with passages = CONTRADICTS | 49 | 27 | 33.8 [16, 59] | 75.0 [57, 90] | 63.5 [45, 77] | 46.1 [18, 95] | 70.4 [52, 89] | -18.1 [-35, +4] |
| E1 both agree: stance oppose AND relation call | 28 | 19 | 67.9 [44, 88] | 52.8 [38, 69] | 59.4 [43, 73] | 16.6 [7, 47] | 70.4 [52, 89] | +15.9 [+3, +32] |
| E1 2 of 3: stance, relation with passages, relation claims only | 46 | 31 | 62.0 [41, 77] | 86.1 [75, 96] | 75.6 [60, 85] | 41.1 [17, 97] | 88.9 [80, 100] | +10.1 [-7, +26] |
| NLI only (frozen comparison) | 12 | 7 | 68.5 [14, 91] | 19.4 [3, 39] | 29.2 [5, 51] | 22.6 [1, 61] | 7.4 [0, 19] | +16.6 [-38, +45] |

Pre-registered outcome per E1 detector: E1 both agree: stance oppose AND relation call: **intermediate** (collapse on: batchnorm_covariate_shift); E1 2 of 3: stance, relation with passages, relation claims only: **failure** (precision below 65%).

## C. Topic-level robustness

| Topic | Labelled / opposed | Current trigger: flagged / right, precision, recall | Relation call with passages = CONTRADICTS: flagged / right, precision, recall | E1 both agree: flagged / right, precision, recall | E1 2 of 3: flagged / right, precision, recall |
|---|---|---|---|---|---|
| attention_explanation | 37 / 18 | 30 / 16, 53.3, 88.9 | 17 / 12, 69.5, 66.7 | 14 / 10, 71.4, 55.6 | 21 / 16, 73.8, 88.9 |
| batchnorm_covariate_shift | 28 / 10 | 14 / 5, 35.7, 50.0 | 22 / 9, 23.1, 90.0 | 9 / 4, 44.4, 40.0 | 18 / 10, 59.1, 100.0 |
| lottery_ticket_pruning | 19 / 8 | 8 / 6, 75.0, 75.0 | 10 / 6, 35.9, 75.0 | 5 / 5, 100.0, 62.5 | 7 / 5, 43.0, 62.5 |

E1 both agree: stance oppose AND relation call: collapsed topics: ['batchnorm_covariate_shift']; precision gain over the current trigger with one topic removed: without attention_explanation +14.3, without batchnorm_covariate_shift +21.1, without lottery_ticket_pruning +13.1.
E1 2 of 3: stance, relation with passages, relation claims only: collapsed topics: ['lottery_ticket_pruning']; precision gain over the current trigger with one topic removed: without attention_explanation +5.2, without batchnorm_covariate_shift +6.9, without lottery_ticket_pruning +17.7.

## D. End to end

Four states on the 82 labelled pairs with a consensus status (2 without one are left out).

| System | Accuracy | Accuracy, design-weighted | Macro-F1 | GENUINE said / right | CONTEXT_DEP. said / right | INSUFFICIENT said / right | NONE said / right |
|---|---|---|---|---|---|---|---|
| Constant NONE | 58.5 [43, 74] | 81.5 [61, 95] | 24.6 [20, 29] | 0 / 0 | 0 / 0 | 0 / 0 | 82 / 48 |
| Current trigger (stance oppose, or partial + NLI) → V1 gate | 51.2 [40, 63] | 80.5 [60, 93] | 42.1 [29, 58] | 2 / 0 | 44 / 18 | 5 / 1 | 31 / 23 |
| Relation call with passages = CONTRADICTS → V1 gate | 54.9 [42, 68] | 68.4 [43, 87] | 45.0 [33, 57] | 1 / 0 | 29 / 18 | 18 / 1 | 34 / 26 |
| E1 both agree: stance oppose AND relation call → V1 gate | 67.1 [59, 76] | 82.6 [63, 95] | 53.5 [38, 76] | 1 / 0 | 26 / 15 | 1 / 1 | 54 / 39 |
| E1 2 of 3: stance, relation with passages, relation claims only → V1 gate | 64.6 [54, 75] | 81.1 [61, 94] | 50.9 [40, 63] | 2 / 0 | 32 / 19 | 11 / 1 | 37 / 33 |

| System | none: precision / recall | genuine: precision / recall | context-dep.: precision / recall | insufficient: precision / recall |
|---|---|---|---|---|
| Constant NONE | 58.5 / 100.0 | – / – | – / 0.0 | – / 0.0 |
| Current trigger (stance oppose, or partial + NLI) → V1 gate | 74.2 / 47.9 | 0.0 / – | 40.9 / 64.3 | 20.0 / 16.7 |
| Relation call with passages = CONTRADICTS → V1 gate | 76.5 / 54.2 | 0.0 / – | 62.1 / 64.3 | 5.6 / 16.7 |
| E1 both agree: stance oppose AND relation call → V1 gate | 72.2 / 81.2 | 0.0 / – | 57.7 / 53.6 | 100.0 / 16.7 |
| E1 2 of 3: stance, relation with passages, relation claims only → V1 gate | 89.2 / 68.8 | 0.0 / – | 59.4 / 67.9 | 9.1 / 16.7 |

## F. Verdict by the pre-registered rule

**PROMISING BUT INCONCLUSIVE** — no E1 detector meets the success criteria and at least one does not meet a failure condition.

