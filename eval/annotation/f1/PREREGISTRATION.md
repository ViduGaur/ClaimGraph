# Fresh-topic validation of the E1 opposition detectors — pre-registration (batch f1)

Written 2026-10-07, before the frozen pipeline was run on the fresh topics and before any fresh pair was labelled.
Nothing below may be changed after the labels exist. Scoring code: `backend/app/evaluation/fresh_validation.py`
(its SHA-256 at the time the sample is drawn is recorded in `eval/FREEZE.json`).

## Question

Does requiring agreement between independent semantic readings of a claim pair improve the precision of
face-value opposition detection on data that played no part in development, without making recall unusably low?
C4 is development data and E1 was specified after its agreement signal had been previewed, so C4 is not evidence
for this.

## Fresh data

Three topics, chosen by the assistant before any pipeline output existed, as debated machine-learning questions
with several papers on each side. None of the 29 papers is in any of the eight development corpora (checked).

| Topic | Question | Seeds (arXiv) |
|---|---|---|
| lottery_ticket_pruning | Can sparse subnetworks found by pruning be trained in isolation to match the accuracy of the dense network (the lottery ticket hypothesis)? | 1803.03635, 1810.05270, 1903.01611, 1905.01067, 1902.09574, 2009.08576, 2009.11094, 1810.02340, 2003.02389, 1912.05671 |
| batchnorm_covariate_shift | Does batch normalization help neural network training by reducing internal covariate shift? | 1502.03167, 1805.11604, 1806.02375, 1809.00846, 2002.10444, 1901.09321, 2102.06171, 1805.10694, 1803.01814, 1902.08129 |
| attention_explanation | Do attention weights provide faithful explanations of neural model predictions? | 1902.10186, 1908.04626, 1906.03731, 1909.11218, 2010.05607, 1909.07913, 2004.14243, 2005.00928, 2004.03685 |

Corpora were built with the same procedure as the development topics (`build-corpus`, n_papers 10, min_relevance
0.5). All seeds resolved to the intended papers; automatic discovery added no paper to the third topic (a search
API was rate-limited), so it has 9 papers. The corpora are used as built.

Pipeline, unchanged and identical to the development runs: `prepare --run fresh` (max_claims 8, max_pairs 200) and
`llm --run fresh` (pair analysis, relation call with passages, relation call on claims only), same model
(`inclusionai/ling-3.0-flash`), same prompts. A pair whose LLM reading failed stays in the pool with that reading
counted as "no opposition", and the number of such pairs is reported.

## Frozen detectors

* current trigger: the face-value conflict of the present system (`hybrid_no_gate` = stance oppose, or partial + NLI)
* relation only: relation call with passages = CONTRADICTS
* **E1 both agree** (primary): stance = oppose AND relation call with passages = CONTRADICTS
* **E1 2 of 3**: at least two of stance = oppose, relation with passages = CONTRADICTS, relation claims only = CONTRADICTS
* NLI only: frozen comparison, not part of any E1 rule

No calibration, no threshold search, no prompt change, no choice between the two E1 rules after the labels.

## Sample

Every candidate pair of the three topics falls in exactly one stratum, from the frozen readings alone:

* A: stance oppose and relation with passages CONTRADICTS (the both-agree set)
* B: the current trigger fires, not in A
* C: the trigger does not fire; both relation calls say CONTRADICTS
* D: the trigger does not fire; exactly one relation call says CONTRADICTS
* E: no LLM reading signals opposition

Number of pairs to label: N = floor((key balance after the readings − US$0.015) / US$0.0040), at most 100
(US$0.0040 per pair is the measured C4 cost of three annotators).
Allocation (`fresh_validation.allocate`): A, then B, up to 75% of N; the remainder is split 40 / 35 / 25 over
C, D, E, and what a small stratum cannot use passes to the next. Simple random sampling without replacement within
each stratum, pooled over topics, seed 20261007. Design weight = stratum pool / stratum sample. System outputs
choose which pairs are labelled and never appear in an annotation prompt. No pair is removed after sampling; a pair
that ends with fewer than three labels (provider failure, budget stop) is reported and left out.

## Annotation

The C4 protocol exactly: same instructions (`conflict_annotation.SYSTEM`), same three annotator models (Gemini,
Qwen, Haiku), same parser, same per-dimension majority, four states kept, no-consensus pairs kept as such.
Reference opposition = majority answer "oppose" on dimension A, as in C4.

## Estimators

* **Precision** (used by the criteria): design-weighted, i.e. an estimate for the whole candidate pool.
* **Recall** (used by the criteria, "C4-comparable"): computed as C4 computed it, over the labelled sample that
  holds the trigger-fired pairs (A and B, with their design weights if B had to be subsampled) plus the sampled
  other pairs unweighted. Stated before the labels: C4's recall figures were relative to its labelled pool, which
  held every trigger-fired pair but only 56 of the 1,460 other candidates. They are therefore upper estimates of
  recall over all candidates, for every detector including the current trigger. The fresh sample of non-trigger pairs is
  enriched by relation-call signals rather than by earlier annotator signals, so the two recalls are comparable in
  construction, not identical.
* **Full-pool recall** (reported, not used by the criteria): design-weighted over all strata. C4 never measured it.
  It cannot be estimated precisely at this budget, because stratum E is large and thinly sampled.
* Share of the current trigger's true positives that a detector keeps.
* F1 from the C4-comparable quantities.
* Intervals: 95% percentile bootstrap, 2,000 resamples of paper pairs, weights fixed. Raw counts are always given.
* By topic: flagged, true positives, precision, recall for each detector.

## End to end

Constant NONE; current trigger → V1 gate; relation only → V1 gate; both agree → V1 gate; 2 of 3 → V1 gate, on the
labelled pairs with a consensus status. Accuracy (plain and design-weighted), macro-F1, per-class precision and
recall, and said / right counts per class. The V1 gate is not changed.

## Criteria (the user's, fixed before the labels)

* Primary success: both agree or 2 of 3 reaches precision ≥ 75% and recall ≥ 60% and does not collapse on any
  single fresh topic.
* Strong success: precision ≥ 75% and recall ≥ 70%, with the precision gain over the current trigger positive.
* Failure: precision < 65%, or recall < 50%, or the apparent gain comes almost entirely from one topic.

Operational definitions added by the assistant, also before the labels:

* collapse: a topic with at least 5 pairs flagged by the detector and precision below 50%, or with at least 5
  reference-opposed labelled pairs and recall below 30%;
* gain from one topic: the design-weighted precision gain over the current trigger is ≤ 0 once some single topic
  is removed.

## Verdict rule

Each E1 detector gets one outcome from the point estimates: failure (any failure condition), success / strong
(the criteria above), otherwise intermediate.

* **FALSIFIED**: both E1 detectors are failures.
* **SUPPORTED**: an E1 detector is a success or strong, the lower end of its 95% precision interval is at least
  65%, and the interval of its precision gain over the current trigger excludes zero.
* **PROMISING BUT INCONCLUSIVE**: everything else.

The verdict is computed by `fresh_validation.verdict`. The error analysis is done after the metrics are written,
and changes nothing.

## Not done

No E2 or E3 work, no frames, no calibration, no NLI experiment, no prompt variant. Nothing is wired into the
pipeline.
