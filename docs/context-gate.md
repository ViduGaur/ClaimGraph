# The context gate — what the code actually does

An audit of the decision logic as implemented (`backend/app/reasoning/relations.py`,
`alignment.py`, `conflicts.py`), written from the code and pinned by `backend/tests/test_context_gate.py`.
Where the implementation is arbitrary or surprising, this document says so.

## 1. Two levels

| Level | Values | Where it is set |
|---|---|---|
| **relation** | SUPPORTS, EXTENDS, QUALIFIES, CONTRADICTS (or none: the pair is unrelated) | `RelationClassifier.decide` |
| **conflict verdict** (only for face-value conflicts) | genuine, context-dependent, insufficient evidence | `analyze_conflicts` |

QUALIFIES means that one claim narrows, conditions, limits or contextualises the other, so that an
apparent disagreement is not a direct contradiction under equivalent conditions. It arises in two ways:
from a face-value conflict the gate judges context-dependent, and from a `partial` stance with no
conflict at all. Only the first carries a verdict.

A face-value conflict whose context cannot be compared keeps the relation CONTRADICTS, flagged
`uncertain`, with verdict *insufficient evidence*. The system does not assert it. In the evaluation
this is reported as an abstention.

## 2. Inputs to the decision

For a candidate pair (cross-paper, embedding cosine ≥ 0.72):

| Input | Source in the evaluated (hybrid) system | Source without an LLM |
|---|---|---|
| NLI contradiction `con` | mean of the two directions, `nli-deberta-v3-xsmall` | same |
| NLI entailment `ent` | maximum of the two directions | same |
| stance | LLM: `agree` / `oppose` / `partial` / `unrelated` | subject/object entities + polarity cue words |
| factor table | LLM: per factor, each paper's value with a verbatim quote, a status and a `material` flag | lexicon and regex matches compared as sets |

The LLM decides nothing after this point, but **its stance and its per-factor status and
materiality are the inputs the rules act on**. The gate is deterministic given those inputs; it is
not independent of the LLM's reading.

## 3. The twelve factors and how each is compared

| Factor | Weight | Counts as experimental context for the "is anything known" test |
|---|---|---|
| data scale | 1.5 | yes |
| experimental conditions | 1.3 | yes |
| dataset | 1.2 | yes |
| population | 1.2 | yes |
| training & design | 1.0 | yes |
| task | 1.0 | yes |
| evaluation metric | 0.8 | yes |
| model / method | 0.6 | **no** |
| baseline | 0.6 | **no** |
| methodology | 0.5 | yes |
| time period | 0.5 | yes |
| sample size | 0.4 | yes |

Every factor ends up with one status:

- `unknown` — not stated by both papers. This includes a factor stated by one paper only, and a value
  whose quote was not found in that paper's text (it is dropped first).
- `same` — both state it and the values match.
- `overlap` — partly the same, **or** the LLM marked the difference as not material to the findings.
- `different` — both state it, the values differ, and (hybrid) the LLM marked the difference material.

Hybrid: the LLM assigns `same` / `different` / `overlap` and `material`; code only enforces the
grounding rules. Without an LLM: set equality → `same`, non-empty intersection → `overlap`, disjoint →
`different`; model and baseline names are compared by family; and one hand-written rule turns an
overlap of training-regime labels into `different` when one side has a pre-training label and the
other a from-scratch label (`_EXCLUSIVE_CONDITIONS`). **That rule is specific to the vision topic the
project started from.** It only exists on the no-LLM path.

## 4. Are the factors weighted equally?

**No.** The weights above are hand-set and were never fitted or validated. They sum to 10.6.

They are used in one place, the divergence score:

```
divergence = Σ w·s / Σ w   over factors that are not unknown
             s = 0 (same), 0.4 (overlap), 1 (different);  w is halved for a paper-level value
```

What the weights do and do not affect:

| | Hybrid system (evaluated) | Without an LLM |
|---|---|---|
| verdict | **nothing** — one material `different` factor decides, whatever its weight | QUALIFIES needs a `different` factor **and** divergence ≥ 0.3 |
| reported `context_divergence` | yes | yes |
| confidence of a QUALIFIES edge | yes (`+0.2·divergence`) | yes |

So in the evaluated system the twelve factors are, for the verdict, **unweighted and
interchangeable**: any single stated, material mismatch — a different evaluation metric as much as
a different dataset — makes a conflict context-dependent. The weights are effectively dead for the
decision. This is the most important thing to know about the gate, and it is why "always
context-dependent" has to be tested as a degenerate alternative.

## 5. The decision, in order

```
1. stance == unrelated                              → no relation                         (hybrid only)
2. face-value conflict?
      stance == oppose                                                     (hybrid: no NLI needed)
   or con ≥ 0.7 and stance ≠ agree and not same-direction                  (in hybrid: stance == partial)
   if yes → THE GATE:
      a. no factor known among the ten context factors
                                                    → CONTRADICTS, uncertain   → verdict insufficient
      b. else some factor is `different`
         (hybrid: any one; no LLM: and divergence ≥ 0.3)
                                                    → QUALIFIES                → verdict context-dependent
      c. else                                       → CONTRADICTS              → verdict genuine
3. agreement?  stance == agree, or ent ≥ 0.5, or (ent ≥ 0.3 and very similar)
      a `different` factor among dataset / data scale / task / population / experimental conditions
                                                    → EXTENDS
      else                                          → SUPPORTS
4. stance == partial                                → QUALIFIES (no verdict)
5. otherwise                                        → no relation
```

Answers to the specific questions:

- **Missing context.** Never counts as a difference. A one-sided or unquoted factor is `unknown` and
  is ignored by the divergence and by the gate. If nothing is known on both sides, the gate abstains.
- **What is a meaningful mismatch.** Hybrid: both papers state the factor with a quote that is found
  in their text, the LLM says the values differ, and the LLM marks the difference material. No LLM:
  both lexicons fire and the value sets are disjoint.
- **How many mismatches decide.** Hybrid: one. No LLM: at least one, and weighted divergence ≥ 0.3.
- **Which factors can decide.** All twelve, including model / method and baseline.
- **Stance and the gate.** The gate only runs after a face-value conflict. `agree` and `unrelated`
  never reach it. In the hybrid system stance is the gate's trigger; NLI is not needed.
- **NLI and the gate.** In the hybrid system NLI matters only for `partial` pairs: contradiction ≥ 0.7
  turns one into a face-value conflict (then the gate runs), entailment ≥ 0.5 turns one into
  agreement. An `agree` stance blocks an NLI contradiction; an `oppose` stance stands against an NLI
  entailment; `unrelated` ends the analysis whatever NLI says. Without stance, NLI contradiction ≥ 0.7
  is the only trigger.
- **QUALIFIES** is caused by (i) a face-value conflict plus a decisive mismatch, or (ii) a `partial`
  stance without confident NLI either way.
- **Insufficient evidence** is caused by a face-value conflict where no context factor is known on
  both sides. On the no-LLM path it is also set when one claim is a statement attributed to prior
  work; that cannot happen with LLM-selected claims.
- **CONTRADICTS (genuine)** requires a face-value conflict, at least one context factor known on both
  sides, and no `different` factor.

## 6. Quirks found in the audit

1. **Fixed.** A `partial` stance from the LLM was honoured only if embedding similarity was ≥ 0.79, a
   cut-off written for the noisy surface-cue stance. `agree` and `oppose` had no such condition. It is
   removed for LLM stances. Effect on the 1,200 held-out pairs: 7 pairs move from unrelated to
   QUALIFIES; no asserted contradiction changes. Logged in `eval/FREEZE.json`.
2. **Not changed, documented.** Model / method and baseline are treated asymmetrically. They do not
   count as "known context", so a conflict where only they are stated is *insufficient* even if they
   differ (8 of the 17 abstentions on the held-out pairs). But once any other factor is known, a
   model / method or baseline mismatch decides like any other: in 17 of the 102 context-dependent
   verdicts these are the only decisive mismatches. A difference in what the claims are about is
   arguably not a difference in conditions; changing this now would be tuning after the fact.
3. **Not changed, documented.** The factor weights do not influence the hybrid verdict (section 4).
4. **Not changed, documented.** Quote grounding verifies that a quote occurs in the paper. It does not
   verify that the stated value is a fair reading of the quote, nor that the factor label is the
   right one. An independent judge found one such value in the explanation audit.
5. **Not changed, documented.** `_EXCLUSIVE_CONDITIONS` and the lexicons are topic-specific; they
   affect only the no-LLM ablation.
6. **Not changed, documented.** Whether a difference is "material" is the LLM's judgement, with no
   check. On the held-out pairs the LLM reported 3,319 context values; 258 stated differences were
   downgraded to `overlap` as immaterial, and 162 values were dropped because their quote was not found.
7. **Not changed, documented.** On the no-LLM path a value found only in paper-level text does not count
   as known context; on the hybrid path it does, at half weight in the divergence.
