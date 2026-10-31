# V2 development study: does graded factor evidence separate genuine from explained conflicts?

**Development data, not a paper result.** Batch `eval2` (run `main`), consensus labels of three annotator models, unadjudicated. These pairs were the V1 test set; they are development data from V2 onwards.

## Summary

1. **The factor evidence carries signal, and all of it is the sufficiency axis.** Opposed pairs in which no context factor is stated by both papers are consensus contradictions 10/16 times; pairs with shared context 9/52 (Fisher p = 0.0010; the same direction under each annotator's own labels). That split alone gives AUC 0.69 and is the model the selection rule picks.
2. **Graded divergence adds nothing detectable.** Among the 52 pairs with shared context, no mismatch score separates the 9 contradictions (AUC 0.47 for context mismatches, 0.50 with the V1 weights). Shuffling factor tables among pairs on the same side of E = 0 does not lower the AUC (real 0.66, shuffled 0.67, p = 0.57). With 9 positives this cannot rule out a moderate effect; it shows none.
3. **Fitted role weights do not beat simpler models.** Out-of-fold log-loss: role weights 0.597, uniform 0.577, V1 hand weights 0.575, sufficiency only 0.544, constant 0.617. On all opposed pairs the fitted order is as hypothesised (scope 0.86 > measurement 0.46 > intervention 0.17), but the scope weight mostly encodes 'no mismatch because nothing is stated': fitted only on pairs with shared context it falls to 0.23 and the measurement weight to 0.00. The agreement weight is fitted at zero throughout.
4. **P3 as specified does not fix recall.** Out-of-fold it asserts 2 pairs (role weights) or 0 (uniform), none of them a consensus contradiction; V1 asserted 7, 2 correct. The only rule that reaches about half of the contradictions is adding the pairs with no shared context, which needs no fitted number and is the V1 'counting unasserted conflicts' reading.
5. **The rule 'an intervention mismatch means the claims are not opposed' is not supported**: pairs with a method or baseline mismatch are contradictions as often as the others.
6. **Beyond stance:** over all scored pairs, adding 'is any context stated' to the stance signals moves AUC from 0.70 to 0.79; adding mismatch counts on top changes nothing.

- annotated pairs: 200; with a consensus label: 181; consensus contradictions: 25
- pairs the system reads as opposed (V1 trigger, unchanged): 68 scored, of which 19 are consensus contradictions (28%). The other 6 contradictions are lost at the stance step (LLM stance: {'partial': 1, 'unrelated': 5}); no context model can recover them.
- topics (folds): adam_vs_sgd 13/4, icl_label_correctness 18/4, label_smoothing_distillation 9/2, large_batch_generalization 13/2, robustness_accuracy_tradeoff 13/7, transformers_time_series 2/0 (opposed pairs / contradictions)

## 1. What the factor tables contain on opposed pairs

Contradictions / pairs for each status of each factor. `different*` = a stated difference the LLM marked immaterial.

| Factor | Role | not stated by both | same | partly | different* | different |
|---|---|---|---|---|---|---|
| dataset | scope | 17/58 | – | 1/3 | – | 1/7 |
| data scale | scope | 19/68 | – | – | – | – |
| population | scope | 18/63 | 1/1 | 0/3 | 0/1 | – |
| task | scope | 16/45 | 1/7 | 2/15 | 0/1 | – |
| experimental conditions | scope | 14/35 | 1/2 | 0/2 | – | 4/29 |
| training & design | scope | 18/63 | – | – | – | 1/5 |
| evaluation metric | measurement | 15/44 | 0/5 | 2/7 | 0/2 | 2/10 |
| methodology | measurement | 19/68 | – | – | – | – |
| sample size | strength | 19/68 | – | – | – | – |
| time period | strength | 19/68 | – | – | – | – |
| model / method | intervention | 8/23 | 2/7 | 2/9 | 1/6 | 6/23 |
| baseline | intervention | 12/44 | 1/6 | 3/6 | 1/6 | 2/6 |

Factors never stated by both papers in any opposed pair: **data scale, methodology, sample size, time period**. Their weights cannot be estimated from this data, whatever the model.

## 2. Where the signal is

E = number of scope and measurement factors stated by both papers (a paper-level statement counts half). E = 0 is what V1 called *insufficient evidence*.

| Shared context | Opposed pairs | Consensus contradictions | Rate | …of which unanimous | Gemini | Qwen | Haiku |
|---|---|---|---|---|---|---|---|
| E = 0 (none stated by both) | 16 | 10 | 62% | 6 | 7/17 | 10/17 | 10/17 |
| E > 0 | 52 | 9 | 17% | 3 | 3/60 | 15/60 | 22/60 |

Per-annotator columns: that model's own CONTRADICTS labels over all opposed pairs, including three-way splits.

Fisher's exact test, E = 0 versus E > 0 (consensus labels): p = 0.0010. By topic (E = 0 vs E > 0): adam_vs_sgd 2/2 vs 2/11, icl_label_correctness 1/1 vs 3/17, label_smoothing_distillation 2/3 vs 0/6, large_batch_generalization 2/5 vs 0/8, robustness_accuracy_tradeoff 3/5 vs 4/8, transformers_time_series 0/0 vs 0/2.

| Contradiction rate by amount of context mismatch (pairs with E > 0) | Pairs | Contradictions | Rate |
|---|---|---|---|
| no mismatch | 3 | 0 | 0% |
| 0.5 (one partial) | 15 | 4 | 27% |
| 1 | 20 | 3 | 15% |
| 1.5–2 | 11 | 2 | 18% |
| > 2 | 3 | 0 | 0% |

| Score (no fitting: fewer mismatches = more likely genuine) | AUC, all opposed pairs | AUC, only pairs with E > 0 (n = 52, 9 contradictions) |
|---|---|---|
| all mismatches (uniform) | 0.62 | 0.49 [0.16, 0.80] |
| scope + measurement mismatches | 0.66 | 0.47 [0.21, 0.75] |
| scope mismatches | 0.64 | 0.45 [0.19, 0.70] |
| hand-weighted mismatches (V1 weights) | 0.64 | 0.50 [0.21, 0.79] |
| intervention mismatches | 0.54 | 0.51 [0.24, 0.77] |
| agreeing factors (reversed: more agreement = more genuine) | 0.49 | 0.56 [0.30, 0.82] |
| E = 0 versus E > 0 alone | 0.69 | – |

AUCs compare pairs within the same topic. 0.50 = no signal.

## 3. Fitted models: leave-one-topic-out cross-validation

Target: consensus CONTRADICTS among the 68 opposed pairs. Log-loss skill = 1 − loss / loss of the constant model (negative = worse than ignoring the factors). Ridge penalty fixed in advance (λ = 1.0). Last column: the same model fitted and tested only on pairs with E > 0, where sufficiency cannot help.

| Model | Fitted numbers | AUC [95% CI] | Log-loss | Skill vs constant | 5-fold AUC (200 repeats) | AUC, E > 0 only |
|---|---|---|---|---|---|---|
| Constant (stance alone: every opposed pair alike) | 1 | 0.50 (by construction) | 0.617 | +0.0% | – | – |
| Semantic signals only (NLI contradiction, entailment, similarity) | 4 | 0.34 [0.15, 0.55] | 0.657 | -6.5% | 0.38 ± 0.07 | 0.33 |
| Sufficiency only: is any context factor stated by both papers? | 2 | 0.69 [0.55, 0.82] | 0.544 | +11.8% | 0.63 ± 0.02 | – |
| Uniform weights, scope + measurement mismatches only | 2 | 0.66 [0.48, 0.83] | 0.577 | +6.5% | 0.68 ± 0.02 | 0.48 |
| Role weights: scope, measurement (mismatch only) | 3 | 0.66 [0.48, 0.83] | 0.586 | +5.0% | 0.66 ± 0.02 | 0.47 |
| Uniform weights, mismatches only | 2 | 0.62 [0.42, 0.82] | 0.582 | +5.7% | 0.66 ± 0.01 | 0.46 |
| Uniform weights, agreement and mismatch | 3 | 0.61 [0.41, 0.81] | 0.588 | +4.7% | 0.66 ± 0.02 | 0.46 |
| V1 hand weights, mismatches only | 2 | 0.64 [0.44, 0.84] | 0.575 | +6.8% | 0.68 ± 0.02 | 0.47 |
| V1 hand weights, agreement and mismatch | 3 | 0.64 [0.43, 0.82] | 0.582 | +5.7% | 0.67 ± 0.02 | 0.46 |
| P3: role weights (scope, measurement, strength; intervention outside) | 5 | 0.65 [0.47, 0.82] | 0.597 | +3.3% | 0.66 ± 0.02 | 0.47 |
| Role weights incl. intervention | 6 | 0.63 [0.43, 0.82] | 0.599 | +2.9% | 0.63 ± 0.03 | 0.45 |
| Role weights incl. intervention, signs unconstrained | 6 | 0.58 [0.39, 0.76] | 0.613 | +0.7% | 0.59 ± 0.04 | 0.30 |
| Twelve free mismatch weights + agreement (exploratory) | 14 | 0.61 [0.40, 0.80] | 0.616 | +0.2% | 0.61 ± 0.03 | 0.37 |
| Role weights, LLM materiality flag honoured | 6 | 0.62 [0.42, 0.80] | 0.607 | +1.6% | 0.63 ± 0.03 | 0.43 |
| Semantic signals + role weights | 9 | 0.59 [0.40, 0.78] | 0.628 | -1.8% | 0.60 ± 0.03 | 0.25 |

## 4. Is a bigger model better? Selection by the rule fixed in advance

Rule: the model with the fewest fitted numbers whose out-of-fold log-loss is within one standard error of the best; ties go to the model without hand-set constants. Differences are paired over pairs (bootstrap).

| Model | Fitted numbers | Log-loss | Difference to best (context_stated) ± SE | Within 1 SE | Difference to constant ± SE |
|---|---|---|---|---|---|
| Sufficiency only: is any context factor stated by both papers? | 2 | 0.544 | +0.000 ± 0.000 | yes | -0.073 ± 0.039 |
| Uniform weights, scope + measurement mismatches only | 2 | 0.577 | +0.033 ± 0.034 | yes | -0.040 ± 0.036 |
| Uniform weights, mismatches only | 2 | 0.582 | +0.037 ± 0.039 | yes | -0.035 ± 0.035 |
| V1 hand weights, mismatches only | 2 | 0.575 | +0.031 ± 0.039 | yes | -0.042 ± 0.040 |
| Uniform weights, agreement and mismatch | 3 | 0.588 | +0.044 ± 0.040 | no | -0.029 ± 0.035 |
| V1 hand weights, agreement and mismatch | 3 | 0.582 | +0.038 ± 0.039 | yes | -0.035 ± 0.041 |
| Role weights: scope, measurement (mismatch only) | 3 | 0.586 | +0.042 ± 0.035 | no | -0.031 ± 0.035 |
| P3: role weights (scope, measurement, strength; intervention outside) | 5 | 0.597 | +0.052 ± 0.037 | no | -0.020 ± 0.038 |
| Role weights incl. intervention | 6 | 0.599 | +0.055 ± 0.040 | no | -0.018 ± 0.039 |
| Twelve free mismatch weights + agreement (exploratory) | 14 | 0.616 | +0.072 ± 0.035 | no | -0.001 ± 0.032 |

Selected by the rule: **Sufficiency only: is any context factor stated by both papers?**. Simplest eligible model that uses factor values: **Uniform weights, scope + measurement mismatches only**.

## 5. Shuffled-factor controls

500 shuffles each. (a) every opposed pair gets the factor table of another opposed pair of the same topic: tests whether the tables carry any signal. (b) the swap is only among pairs of the same topic *and* the same side of E = 0: tests whether the factor values carry signal beyond whether any context is stated.

| Model | Real AUC | (a) shuffled: mean [2.5%, 97.5%] | p | (b) shuffled within E cell: mean [2.5%, 97.5%] | p |
|---|---|---|---|---|---|
| Uniform weights, scope + measurement mismatches only | 0.66 | 0.50 [0.38, 0.66] | 0.028 | 0.67 [0.56, 0.78] | 0.575 |
| Uniform weights, agreement and mismatch | 0.61 | 0.49 [0.36, 0.66] | 0.076 | 0.65 [0.51, 0.79] | 0.715 |
| V1 hand weights, mismatches only | 0.64 | 0.50 [0.37, 0.66] | 0.042 | 0.68 [0.55, 0.79] | 0.743 |
| P3: role weights (scope, measurement, strength; intervention outside) | 0.65 | 0.51 [0.35, 0.67] | 0.056 | 0.67 [0.56, 0.78] | 0.649 |
| Role weights incl. intervention | 0.63 | 0.50 [0.33, 0.66] | 0.064 | 0.67 [0.55, 0.80] | 0.727 |

## 6. Fitted coefficients (all opposed development pairs)

Log-odds of *genuine contradiction*. A mismatch weight w enters as −w per mismatched factor, the agreement weight as +w per agreeing factor; constrained weights cannot go below 0. Intervals: percentile bootstrap over pairs. Odds ratio = exp(±w): the factor by which one mismatch (or agreement) multiplies the odds of a genuine contradiction.

**Uniform weights, scope + measurement mismatches only**

| Term | Weight | 95% interval | Odds ratio | Bootstrap fits at exactly 0 |
|---|---|---|---|---|
| intercept | -0.38 | [-1.21, 0.33] | – | – |
| diff:context | 0.88 | [0.09, 1.86] | 0.42 | 1% |

**V1 hand weights, mismatches only**

| Term | Weight | 95% interval | Odds ratio | Bootstrap fits at exactly 0 |
|---|---|---|---|---|
| intercept | -0.18 | [-1.01, 0.69] | – | – |
| diff:hand-weighted | 0.74 | [0.12, 1.70] | 0.48 | 1% |

**P3: role weights (scope, measurement, strength; intervention outside)**

| Term | Weight | 95% interval | Odds ratio | Bootstrap fits at exactly 0 |
|---|---|---|---|---|
| intercept | -0.44 | [-1.24, 0.24] | – | – |
| same | 0.00 | [0.00, 0.52] | 1.00 | 68% |
| diff:scope | 0.87 | [0.05, 1.91] | 0.42 | 2% |
| diff:measurement | 0.44 | [0.00, 1.36] | 0.64 | 18% |
| diff:strength | 0.00 | [0.00, 0.00] | 1.00 | 100% |

**Role weights incl. intervention**

| Term | Weight | 95% interval | Odds ratio | Bootstrap fits at exactly 0 |
|---|---|---|---|---|
| intercept | -0.35 | [-1.10, 0.47] | – | – |
| same | 0.00 | [0.00, 0.53] | 1.00 | 73% |
| diff:scope | 0.86 | [0.09, 1.74] | 0.42 | 1% |
| diff:measurement | 0.46 | [0.00, 1.36] | 0.63 | 22% |
| diff:strength | 0.00 | [0.00, 0.00] | 1.00 | 100% |
| diff:intervention | 0.17 | [0.00, 1.10] | 0.84 | 33% |

**Role weights incl. intervention, signs unconstrained**

| Term | Weight | 95% interval | Odds ratio | Bootstrap fits at exactly 0 |
|---|---|---|---|---|
| intercept | -0.29 | [-1.20, 0.50] | – | – |
| same | -0.21 | [-1.35, 0.56] | 0.81 | 0% |
| diff:scope | 0.86 | [0.12, 1.78] | 0.42 | 0% |
| diff:measurement | 0.47 | [-0.63, 1.42] | 0.62 | 0% |
| diff:strength | 0.00 | [0.00, 0.00] | 1.00 | 100% |
| diff:intervention | 0.18 | [-0.60, 1.08] | 0.84 | 0% |

**Twelve free mismatch weights + agreement (exploratory)**

| Term | Weight | 95% interval | Odds ratio | Bootstrap fits at exactly 0 |
|---|---|---|---|---|
| intercept | -0.38 | [-1.10, 0.30] | – | – |
| same | 0.00 | [0.00, 0.57] | 1.00 | 71% |
| diff:dataset | 0.47 | [0.00, 1.06] | 0.62 | 12% |
| diff:data scale | 0.00 | [0.00, 0.00] | 1.00 | 100% |
| diff:population | 0.34 | [0.04, 0.63] | 0.71 | 2% |
| diff:task | 0.42 | [0.00, 1.13] | 0.66 | 14% |
| diff:experimental conditions | 0.89 | [0.05, 1.94] | 0.41 | 2% |
| diff:training & design | 0.00 | [0.00, 0.68] | 1.00 | 53% |
| diff:evaluation metric | 0.49 | [0.00, 1.38] | 0.61 | 18% |
| diff:methodology | 0.00 | [0.00, 0.00] | 1.00 | 100% |
| diff:sample size | 0.00 | [0.00, 0.00] | 1.00 | 100% |
| diff:time period | 0.00 | [0.00, 0.00] | 1.00 | 100% |
| diff:model / method | 0.42 | [0.00, 1.22] | 0.66 | 20% |
| diff:baseline | 0.00 | [0.00, 0.80] | 1.00 | 62% |

## 7. Asserted contradictions: precision, recall and false contradictions

Over all 181 scored pairs (25 consensus contradictions). Every decision of a fitted model is out-of-fold: coefficients and threshold come from the other topics.

Rows 'P3 [model]' apply the proposal exactly: E = 0 → *insufficient*; otherwise assert when p ≥ τ, with the coefficients fitted on training pairs with E > 0 and τ the lowest value at which at least half of the asserted training pairs are contradictions (no such value → nothing is asserted).

| Decision rule | Asserted | Correct | Precision | Recall | F1 | False con. /100 |
|---|---|---|---|---|---|---|
| No context (assert every opposed pair) | 68 | 19 | 27.9 | 76.0 | 40.9 | 27.1 |
| V1 hard gate — asserted | 7 | 2 | 28.6 | 8.0 | 12.5 | 2.8 |
| V1 hard gate — asserted + unasserted (insufficient) | 23 | 12 | 52.2 | 48.0 | 50.0 | 6.1 |
| Typed hard gate — asserted | 7 | 2 | 28.6 | 8.0 | 12.5 | 2.8 |
| Typed hard gate — asserted + insufficient | 17 | 8 | 47.1 | 32.0 | 38.1 | 5.0 |
| Opposed and no intervention mismatch (rule proposed for question A) | 42 | 12 | 28.6 | 48.0 | 35.8 | 16.6 |
| Only the pairs with no shared context (E = 0) | 16 | 10 | 62.5 | 40.0 | 48.8 | 3.3 |
| Semantic signals only (NLI contradiction, entailment, similarity) — no sufficiency rule, max-F1 threshold | 24 | 3 | 12.5 | 12.0 | 12.2 | 11.6 |
| Sufficiency only: is any context factor stated by both papers? — no sufficiency rule, max-F1 threshold | 29 | 12 | 41.4 | 48.0 | 44.4 | 9.4 |
| Uniform weights, scope + measurement mismatches only — no sufficiency rule, max-F1 threshold | 25 | 10 | 40.0 | 40.0 | 40.0 | 8.3 |
| Uniform weights, agreement and mismatch — no sufficiency rule, max-F1 threshold | 17 | 7 | 41.2 | 28.0 | 33.3 | 5.5 |
| V1 hand weights, mismatches only — no sufficiency rule, max-F1 threshold | 25 | 13 | 52.0 | 52.0 | 52.0 | 6.6 |
| P3: role weights (scope, measurement, strength; intervention outside) — no sufficiency rule, max-F1 threshold | 22 | 12 | 54.5 | 48.0 | 51.1 | 5.5 |
| Role weights incl. intervention — no sufficiency rule, max-F1 threshold | 22 | 11 | 50.0 | 44.0 | 46.8 | 6.1 |
| Twelve free mismatch weights + agreement (exploratory) — no sufficiency rule, max-F1 threshold | 26 | 11 | 42.3 | 44.0 | 43.1 | 8.3 |
| Role weights, LLM materiality flag honoured — no sufficiency rule, max-F1 threshold | 22 | 10 | 45.5 | 40.0 | 42.6 | 6.6 |
| P3 [uniform_ctx] — asserted | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| P3 [uniform_ctx] — asserted + insufficient | 16 | 10 | 62.5 | 40.0 | 48.8 | 3.3 |
| P3 [hand] — asserted | 3 | 0 | 0.0 | 0.0 | 0.0 | 1.7 |
| P3 [hand] — asserted + insufficient | 19 | 10 | 52.6 | 40.0 | 45.5 | 5.0 |
| P3 [role2] — asserted | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| P3 [role2] — asserted + insufficient | 16 | 10 | 62.5 | 40.0 | 48.8 | 3.3 |
| P3 [role] — asserted | 2 | 0 | 0.0 | 0.0 | 0.0 | 1.1 |
| P3 [role] — asserted + insufficient | 18 | 10 | 55.6 | 40.0 | 46.5 | 4.4 |
| P3 [role4] — asserted | 3 | 0 | 0.0 | 0.0 | 0.0 | 1.7 |
| P3 [role4] — asserted + insufficient | 19 | 10 | 52.6 | 40.0 | 45.5 | 5.0 |

The annotator models had no 'cannot tell' option. The rows that add the *insufficient* pairs therefore show how those pairs were labelled when a label was forced, not whether abstaining on them was right.

## 8. Role and factor ablations

Leave-one-topic-out AUC of the uniform model (agreement + mismatch, all roles) with one role or one factor removed.

| Removed | AUC, all opposed | Δ | AUC, E > 0 only | Δ |
|---|---|---|---|---|
| role: scope | 0.57 | -0.05 | 0.51 | +0.05 |
| role: measurement | 0.59 | -0.03 | 0.42 | -0.04 |
| role: strength | 0.61 | +0.00 | 0.46 | +0.00 |
| role: intervention | 0.65 | +0.04 | 0.48 | +0.02 |
| factor: dataset | 0.61 | +0.00 | 0.45 | -0.01 |
| factor: population | 0.61 | -0.00 | 0.45 | -0.01 |
| factor: task | 0.59 | -0.02 | 0.45 | -0.01 |
| factor: experimental conditions | 0.62 | +0.00 | 0.53 | +0.07 |
| factor: training & design | 0.63 | +0.01 | 0.49 | +0.03 |
| factor: evaluation metric | 0.59 | -0.03 | 0.42 | -0.04 |
| factor: model / method | 0.61 | +0.00 | 0.45 | -0.01 |
| factor: baseline | 0.66 | +0.05 | 0.51 | +0.05 |
| all but scope (keep only scope) | 0.64 | +0.02 | 0.47 | +0.01 |
| all but measurement (keep only measurement) | 0.57 | -0.04 | 0.47 | +0.01 |
| all but intervention (keep only intervention) | 0.50 | -0.12 | 0.47 | +0.01 |

Nothing removed: 0.61 (all opposed), 0.46 (E > 0 only).

## 9. Sensitivity to the reference labels and to the penalty

Leave-one-topic-out AUC against other readings of the same annotations.

| Target | n | positives | context_stated | uniform_ctx | hand | role | role4 | semantic | uniform_ctx, E > 0 only |
|---|---|---|---|---|---|---|---|---|---|
| consensus (main) | 68 | 19 | 0.69 | 0.66 | 0.64 | 0.65 | 0.63 | 0.34 | 0.47 |
| unanimous pairs only | 27 | 9 | 0.81 | 0.80 | 0.81 | 0.83 | 0.85 | 0.52 | 0.44 |
| vote share incl. three-way splits (soft label; AUC against ≥ 2 votes) | 77 | 19 | 0.69 | 0.69 | 0.66 | 0.66 | 0.64 | 0.62 | 0.54 |
| annotator gemini alone | 77 | 10 | 0.75 | 0.72 | 0.75 | 0.70 | 0.73 | 0.34 | 0.44 |
| annotator qwen alone | 77 | 25 | 0.62 | 0.57 | 0.56 | 0.57 | 0.53 | 0.67 | 0.43 |
| annotator haiku alone | 77 | 32 | 0.57 | 0.60 | 0.60 | 0.61 | 0.61 | 0.43 | 0.55 |

| Penalty λ | uniform_ctx | hand | role | role4 | free12 |
|---|---|---|---|---|---|
| 0.1 | 0.66 | 0.64 | 0.65 | 0.63 | 0.59 |
| 1.0 | 0.66 | 0.64 | 0.65 | 0.63 | 0.61 |
| 10.0 | 0.66 | 0.64 | 0.64 | 0.64 | 0.60 |

## 10. Factor evidence beyond semantic stance, on all scored pairs

Target: consensus CONTRADICTS among all 181 scored pairs. 'Stance' = the LLM stance label and NLI contradiction. The other rows add, for opposed pairs, the named factor evidence.

| Model | AUC [95% CI] | Log-loss | Δ log-loss vs stance ± SE |
|---|---|---|---|
| Stance | 0.70 [0.59, 0.81] | 0.375 | – |
| Stance + is any context stated | 0.79 [0.67, 0.90] | 0.344 | -0.031 ± 0.016 |
| Stance + context mismatches | 0.76 [0.65, 0.86] | 0.360 | -0.015 ± 0.010 |
| Stance + both | 0.76 [0.64, 0.87] | 0.345 | -0.030 ± 0.018 |

## 11. What is learned, what is calibrated, what is deterministic

| Component | Kind |
|---|---|
| NLI model, embedding model | pretrained learned models, frozen, deterministic at inference |
| LLM stance and factor-table extraction | pretrained learned model, frozen (temperature 0, cached); the extraction is model-assisted |
| quote grounding, one-sided → unknown, paper-level → half weight | deterministic code |
| factor roles, evidence features d and e, sufficiency E | deterministic code |
| comparison of two stated values | the LLM's status label; deterministic comparison of numeric and named values is not implemented yet |
| comparability coefficients and assertion threshold | **fitted on development data** (penalised logistic regression, 2–6 numbers) |
| sufficiency rule | set a priori (at least one scope or measurement factor stated by both papers); the development labels have no 'cannot tell' option and cannot calibrate it |
| conflict status | deterministic given the table and the fitted numbers |
| explanation | LLM wording after the decision, checked against the evidence record |

**Is the evidence model itself a learning algorithm?** "Evidence model" as a general term is not: it names a way
of combining pieces of evidence into a judgement, and its numbers can come from anywhere (V1's weights were set
by hand, which is still an evidence model). *This implementation* estimates its few coefficients from labelled
development pairs by penalised logistic regression, which is statistical learning in the plain sense of fitting
parameters to data. It is a small calibrated statistical model inside an otherwise rule-based pipeline: it has
at most six fitted numbers, it sees only counts derived from the factor table, and it never sees text. ClaimGraph
as a whole is not a learned or end-to-end trained system; no pretrained model is trained or fine-tuned.
