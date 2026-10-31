# Annotation guide — cross-paper claim relations

You will see pairs of claims taken verbatim from two different papers on the same research
question. For each pair you answer **two questions**. You never see what any system predicted.

Open your file (`<batch>_<your-id>.html`) in a browser. Progress is saved in that browser, so use
the same browser until you are done, then click **Export labels** and send back the downloaded
`<batch>_<your-id>.json`. Budget about one minute per pair.

## Ground rules

- **Judge only from the text shown**: the two claims, their passages and, if needed, the abstracts.
  Do not use what you know about the papers, and do not look them up.
- **Do not discuss items with the other annotators** until everyone has exported.
- If a claim is unintelligible, tick **Not a real claim** for that side. Still label the pair if you
  can; if you cannot, choose *Unrelated*.
- When you are torn between two answers, pick one and say why in the note.

## Question 1 — stance

Look only at what each claim *concludes*. **Ignore differences in setting for this step.**

| Answer | Meaning | Example |
|---|---|---|
| **Agree** | Same conclusion about the same relationship. | A: "X improves accuracy." B: "We also find X helps accuracy." |
| **Oppose** | Conflicting conclusions about the same relationship: opposite direction of effect, *or* one reports an effect or advantage that the other denies or finds absent. | A: "Large batches hurt generalization." B: "Large batches generalize as well as small ones." |
| **Partial** | Same relationship, but neither a clear conflict nor clear agreement: one narrows, weakens or adds a condition to the other, or they agree on one part and differ on another. | A: "X always helps." B: "X helps only for large models." |
| **Unrelated** | Not the same relationship. Sharing a topic, a dataset or a method name is not enough. | A: "X improves accuracy." B: "X is 3× faster to train." |

"The same relationship" = the same kind of method / intervention / factor, assessed on the same
kind of outcome. "Our method beats baseline Z" and "a different method beats baseline Z" are
**unrelated**, not agreement.

## Question 2 — context (skip for *Unrelated*)

Are the conditions under which the two claims were established comparable? Use only what the
claims and passages (and abstracts) state: dataset, data scale, training regime, task, population,
metric, and so on.

| Answer | Meaning |
|---|---|
| **Comparable** | The text states the relevant conditions for both claims and they are essentially the same. |
| **Different** | The text states a difference in conditions that could plausibly explain why the findings differ. For *Agree*: the second claim shows the finding in a different setting. |
| **Not stated** | The text does not say enough about the conditions of at least one claim to tell. |

A difference only counts if it is **stated** and **plausibly matters**. Two papers always differ in
something; an incidental difference (for example a different random seed, or a different but
equivalent benchmark split) is not "Different".

## Four tie-breakers (these cases come up constantly)

1. **Only one claim states its conditions → Not stated.** "X hurts accuracy" versus "With enough
   tuning, X does not hurt accuracy": the second names a condition, the first does not say whether
   it tuned. You cannot tell whether the settings were the same, so do not choose *Different*. Choose
   *Different* only when the text gives the condition of both claims (one passage describing both
   settings counts: "unlike the noise-free setting of earlier work, in noisy settings we find…").
2. **Two explicitly general claims → Comparable.** "There is an inherent trade-off between A and B"
   versus "The trade-off between A and B is not inherent" are both about the general case, so the
   conditions match. A claim tied to one unnamed experiment ("in our experiments…") is not general.
3. **"Our fix removes the problem" versus "the problem exists" → Partial.** A claim that a technique
   *closes*, *reduces* or *eliminates* a gap concedes that the gap exists without the technique, so
   it narrows the other claim rather than opposing it.
4. **Two different named methods → Unrelated.** "Method A reaches state of the art" and "method B
   reaches state of the art" are not about the same relationship, even in the same sub-field. They
   are related only when both are about the same method or factor, or one is explicitly about the
   whole family ("Transformers are not effective for forecasting").

## What your answers mean

| Stance | Context | Derived label |
|---|---|---|
| Agree | Comparable / Not stated | SUPPORTS |
| Agree | Different | EXTENDS |
| Oppose | Comparable | **CONTRADICTS** — a genuine conflict |
| Oppose | Different | **QUALIFIES** — a context-dependent difference |
| Oppose | Not stated | **INSUFFICIENT** — cannot tell from the text |
| Partial | any | QUALIFIES |
| Unrelated | — | UNRELATED |

## Worked examples

1. A: "ViT attains excellent results when pre-trained on large amounts of data."
   B: "ViTs trained on ImageNet alone underperform ResNets of comparable size."
   → **Oppose** at face value (ViT good vs ViT worse) · **Different** (large-scale pre-training vs
   ImageNet only) → QUALIFIES.
2. A: "Method M improves robustness on CIFAR-10." B: "We find M does not improve robustness on
   CIFAR-10 under the same attack." → **Oppose** · **Comparable** → CONTRADICTS.
3. A: "Chain-of-thought prompting improves reasoning." B: "Chain-of-thought does not help."
   Neither passage says which models or tasks. → **Oppose** · **Not stated** → INSUFFICIENT.
4. A: "Our optimiser converges faster than Adam." B: "Adam generalises worse than SGD."
   → **Unrelated** (different comparisons and outcomes).

## After labelling

Items where annotators disagree are listed for adjudication: discuss each one, agree on a final
stance and context, and record it. Agreement (Cohen's κ) is computed on the labels **before**
adjudication, so do not change your exported file afterwards.
