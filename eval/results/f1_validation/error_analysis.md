# Error analysis of the fresh-topic E1 validation (batch f1)

Written after `validation.md` and `validation.json` were produced (sha256 9d45fca7… and 4459021b…). Nothing here
changed a metric, a rule or the verdict, and nothing was rerun.

## E1 both agree

**False positives: 9 of 28 flagged** (6 unanimous "not opposed", 3 by majority; 5 in batchnorm_covariate_shift,
4 in attention_explanation, 0 in lottery_ticket_pruning; three paper pairs account for 6 of them).

| Why the annotators say "not opposed" | Pairs |
|---|---|
| Different method or condition compared: batch-normalised against unnormalised networks, or batch normalisation against Fixup, each reported as working | 4 |
| One claim describes a protocol, a definition or a proposal, the other reports a result | 3 |
| Different outcome: predictive performance against explanatory value | 1 |
| A general benefit against a failure under a narrow condition (annotators split 2 to 1) | 1 |

These are the same errors found on C4 (a different subject or aspect accepted as "the same relationship"). Both
readings come from the same LLM and share this error, so requiring them to agree does not remove it. For 7 of the 9
the third reading (relation on claims only) did not say CONTRADICTS.

**False negatives: 17 of the 36 opposed pairs in the sample** (12 CONTEXT_DEPENDENT, 3 INSUFFICIENT, 2 without a
consensus status; 10 unanimous).

| Readings | Pairs |
|---|---|
| Stance oppose, relation call with passages says QUALIFIES (7) or EXTENDS (1) | 8 |
| Both relation calls say CONTRADICTS, but the pair analysis answered "not the same relationship" | 7 |
| One relation call says CONTRADICTS, pair analysis "not the same relationship" | 1 |
| No reading signals opposition | 1 |

Half of the misses are pairs the relation call reads as QUALIFIES while the annotators read them as opposed at face
value and context-dependent. The other half never reach the current trigger at all: the pair analysis rules them
out at its "same relationship" step (for example, ensembles of batch-normalised networks against normalizer-free
networks on ImageNet).

## E1 2 of 3

**False positives: 15 of 46 flagged** — the 9 above plus 6 of the 13 sampled pairs in which both relation calls
say CONTRADICTS and the pair analysis says "not the same relationship". Those 6 carry a design weight of 3.3 each,
which is why the design-weighted precision (62%) is below the raw 31 of 46.
**False negatives: 5 of 36** — 3 where the stance is oppose and neither relation call agrees, 2 outside the trigger.

## What the sample says about the pool

Of the 13 sampled pairs outside the trigger on which both relation calls say CONTRADICTS, 7 are opposed by the
reference. The current trigger therefore misses a substantial amount of face-value opposition, which C4 could not
show because it labelled few pairs outside the trigger. One of the 7 sampled pairs with no opposition signal is
opposed; with a weight of 54 that single pair dominates every full-pool recall estimate, so those estimates are
not usable.

## Reference

Fleiss κ for stance on the fresh pairs: 0.60 (59 unanimous, 25 by majority). Evidence sufficiency on opposed
pairs: 0.12. No pair has a GENUINE consensus (Gemini alone calls 4 pairs GENUINE), so nothing can be said about
genuine-contradiction detection.
