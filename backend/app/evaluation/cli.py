"""Command-line entry point for the evaluation harness (wrapped by scripts/cg_eval.py)."""
from __future__ import annotations

import argparse
import logging

from collections import Counter

from app.core.config import Settings
from app.evaluation import LLM_CACHE_PATH
from app.evaluation import corpus as corpus_mod
from app.evaluation import experiment as exp
from app.providers.llm import BudgetedLLM, CachedProvider, create_provider
from app.providers.models import get_models

DEFAULT_RUN = "main"


def eval_llm(settings: Settings) -> BudgetedLLM:
    """The configured LLM behind the experiment cache, with no practical call budget."""
    provider = create_provider(settings.model_copy(update={"llm_route_strict": True}))
    if provider is None:
        raise SystemExit("no LLM configured: set CLAIMGRAPH_LLM_API_KEY (OpenRouter) in .env")
    return BudgetedLLM(CachedProvider(provider, LLM_CACHE_PATH), max_calls=10**9)


def _topic_ids(arg: str | None, split: str | None) -> list[str]:
    topics, _ = corpus_mod.load_topics()
    if arg:
        return [t.strip() for t in arg.split(",")]
    return [t.id for t in topics if split in (None, "all", t.split)]


def cmd_build_corpus(args: argparse.Namespace) -> None:
    settings = Settings()
    models = get_models(settings)
    _, n_default = corpus_mod.load_topics()
    for tid in _topic_ids(args.topic, args.split):
        if corpus_mod.corpus_path(tid).exists() and not args.force:
            print(f"[skip] {tid}: snapshot exists (use --force to rebuild)")
            continue
        snap = corpus_mod.build_corpus(corpus_mod.get_topic(tid), args.n_papers or n_default, settings, models,
                                       discover_extra=not args.seeds_only)
        for note in snap["notes"]:
            print(f"  note: {note}")
        print(corpus_mod.describe_corpus(tid), flush=True)


def cmd_describe(args: argparse.Namespace) -> None:
    for tid in _topic_ids(args.topic, args.split):
        try:
            print(corpus_mod.describe_corpus(tid))
        except FileNotFoundError as exc:
            print(f"{tid}: {exc}")


def cmd_prepare(args: argparse.ArgumentParser) -> None:
    settings = Settings()
    models = get_models(settings)
    llm = None if args.rule_claims else eval_llm(settings)
    for tid in _topic_ids(args.topic, args.split):
        data = exp.prepare_topic(tid, args.run, settings, models, max_claims=args.max_claims, max_pairs=args.max_pairs, llm=llm)
        types = Counter(c.claim_type.value for c in data.claims.values())
        print(f"{tid}: {data.meta['n_papers']} papers ({data.meta['full_text']} full text), {len(data.claims)} claims "
              f"{dict(types)}, {len(data.records)} candidate pairs (sim >= {data.threshold})", flush=True)
        methods = Counter(c.extraction_method for c in data.claims.values())
        if llm is not None and (llm.failures or methods.get("heuristic")):
            print(f"  WARNING: claim selection fell back to cue rules for some papers: {dict(methods)}; "
                  f"{len(llm.failures)} LLM failures so far", flush=True)


def cmd_llm(args: argparse.Namespace) -> None:
    settings = Settings()
    llm = eval_llm(settings)
    fields = tuple(args.fields.split(",")) if args.fields else exp.LLM_FIELDS
    for tid in _topic_ids(args.topic, args.split):
        data = exp.load_topic(args.run, tid)
        if args.refresh:  # recompute with the current prompts (old responses stay in the cache under their own keys)
            for rec in data.records:
                for f in fields:
                    rec.pop(f, None)
        only = None
        if args.limit:
            only = {r["pair_id"] for r in data.records[: args.limit]}
        failed = exp.add_llm_signals(data, llm, concurrency=args.concurrency or settings.llm_concurrency, fields=fields,
                                     only=only, progress=lambda m: print("  " + m, flush=True))
        exp.save_topic(data, args.run)
        done = {f: sum(1 for r in data.records if r.get(f)) for f in fields}
        print(f"{tid}: {done} of {len(data.records)} pairs; failed this pass: {failed}", flush=True)
    cache = llm.provider
    print(f"LLM {settings.llm_provider_name}/{settings.llm_model_name}: {cache.misses} new calls, {cache.hits} cache hits; "
          f"{len(llm.failures)} failures")
    for f in llm.failures[:5]:
        print("  failure:", f)


def cmd_report(args: argparse.Namespace) -> None:
    from app.evaluation.report import format_topic

    for tid in _topic_ids(args.topic, args.split):
        print(format_topic(exp.load_topic(args.run, tid), args.system, show_unrelated=args.all, limit=args.limit,
                           claims=not args.no_claims))
        print()


def cmd_sheet(args: argparse.Namespace) -> None:
    from app.evaluation import EVAL_DIR
    from app.evaluation import annotation as ann

    datas = {tid: exp.load_topic(args.run, tid) for tid in _topic_ids(args.topic, args.split)}
    quotas = {"full_assert": args.full_assert, "gate_input": args.gate_input, "conflict": args.conflict,
              "agreement": args.agreement, "background": args.background}
    items = ann.sample_items(list(datas.values()), quotas, seed=args.seed)
    out_dir = EVAL_DIR / "annotation" / args.batch
    if any(out_dir.glob(f"{args.batch}_key.json")) and not args.force:
        raise SystemExit(f"{out_dir} already holds batch {args.batch!r}; refusing to overwrite labels in progress (use --force)")
    per = ann.write_sheets(items, datas, args.annotators.split(","), out_dir, per_item=args.per_item, overlap=args.overlap,
                           seed=args.seed, batch=args.batch)
    strata = Counter((i.topic, i.stratum) for i in items)
    for tid in datas:
        print(f"{tid}: " + ", ".join(f"{s}={strata[(tid, s)]}" for s in ann.STRATA))
    print(f"{len(items)} items → {out_dir}")
    for annotator, n in per.items():
        print(f"  {annotator}: {n} items  ({args.batch}_{annotator}.html)")


def cmd_score(args: argparse.Namespace) -> None:
    import csv
    import json

    from app.evaluation import EVAL_DIR
    from app.evaluation import scoring
    from app.evaluation.annotation import derive_label, load_labels

    labels_dir = EVAL_DIR / "annotation" / args.batch
    gs = scoring.load_gold(labels_dir, mode=args.gold)
    if not gs.gold:
        raise SystemExit(f"no usable gold labels found in {labels_dir}")
    topics = sorted({i["topic"] for i in gs.key_items if i["item_id"] in gs.gold})
    if args.split != "all":
        wanted = set(_topic_ids(None, args.split))
        topics = [t for t in topics if t in wanted]
    datas = {tid: exp.load_topic(args.run, tid) for tid in topics}
    suffix = "" if args.gold == "final" else "_" + args.gold.replace(":", "_")
    out_dir = EVAL_DIR / "results" / (args.name or f"{args.batch}_{args.run}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agreement.json").write_text(json.dumps(gs.agreement, indent=1))
    print(f"gold: {gs.description}; {len(gs.gold)} usable, {len(gs.excluded)} excluded")
    if gs.kind == "two_question" and gs.excluded:
        raw = load_labels(labels_dir)
        by_item = {i["item_id"]: i for i in gs.key_items}
        with (labels_dir / "to_adjudicate.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["item_id", "topic", "answers", "stance", "context"])
            for iid in gs.excluded:
                answers = "; ".join(f"{a}: {v['stance']}/{v['context']} → {derive_label(v['stance'], v['context'])}"
                                    for a, v in raw[iid].items())
                w.writerow([iid, by_item.get(iid, {}).get("topic", ""), answers, "", ""])
        print(f"{len(gs.excluded)} items have conflicting labels and are excluded until adjudicated: fill stance/context in "
              f"{labels_dir / 'to_adjudicate.csv'} and save it as adjudicated.csv")
    summary = scoring.score(gs, datas, out_dir, n_boot=args.bootstrap, title=args.title or f"{args.batch} / run {args.run}")
    print((out_dir / "results.md").read_text().split("## Per topic")[0])
    print(f"written: {out_dir}  ({summary['n_items']} items)")


def cmd_explain(args: argparse.Namespace) -> None:
    """Grounded explanations for every face-value conflict of the full system (after the decision; cached)."""
    settings = Settings()
    llm = eval_llm(settings)
    total = {"conflicts": 0, "llm": 0, "template": 0, "new": 0}
    for tid in _topic_ids(args.topic, args.split):
        data = exp.load_topic(args.run, tid)
        stats = exp.add_explanations(data, llm, concurrency=args.concurrency or settings.llm_concurrency, refresh=args.refresh)
        exp.save_topic(data, args.run)
        for k in total:
            total[k] += stats[k]
        print(f"{tid}: {stats}", flush=True)
    cache = llm.provider
    print(f"total: {total}; LLM {settings.llm_model_name}: {cache.misses} new calls, {cache.hits} cache hits, {len(llm.failures)} failures")


def cmd_examples(args: argparse.Namespace) -> None:
    """Candidate paper examples: face-value conflicts the full system resolves as context-dependent, genuine or insufficient."""
    from app.evaluation import EVAL_DIR
    from app.evaluation import scoring
    from app.evaluation.metrics import ABSTAIN
    from app.evaluation.report import format_pair

    gold_by_pair: dict[str, str] = {}
    if args.batch:
        gs = scoring.load_gold(EVAL_DIR / "annotation" / args.batch)
        gold_by_pair = {i["pair_id"]: gs.gold[i["item_id"]]["label"] for i in gs.key_items if i["item_id"] in gs.gold}
    wanted = {"A": "QUALIFIES", "B": "CONTRADICTS", "C": ABSTAIN}
    agrees = {"QUALIFIES": ("QUALIFIES",), "CONTRADICTS": ("CONTRADICTS",), ABSTAIN: ("CONTRADICTS", "QUALIFIES")}
    found: dict[str, list] = {k: [] for k in wanted}
    for tid in _topic_ids(args.topic, args.split):
        data = exp.load_topic(args.run, tid)
        systems = exp.Systems(data)
        for rec in data.records:
            label, rel = exp.predict(systems, "hybrid_full", rec)
            if rel is None or not rel.signals.conflict_signal:
                continue
            gold_label = gold_by_pair.get(rec["pair_id"])
            if args.batch and args.confirmed and gold_label not in agrees.get(label, ()):
                continue
            for key, target in wanted.items():
                if label == target:
                    grounded = sum(1 for f in rel.factors if f.status == "different" and f.a_quote and f.b_quote)
                    found[key].append((grounded, rel.confidence, data, rec, label, rel, gold_label))
    titles = {"A": "Example A — face-value conflict explained by context (relation QUALIFIES, verdict context-dependent)",
              "B": "Example B — conflict under comparable conditions (relation CONTRADICTS, verdict genuine)",
              "C": "Example C — conflict with too little stated context (CONTRADICTS not asserted, verdict insufficient)"}
    for key in wanted:
        print(f"\n{'=' * 100}\n{titles[key]}: {len(found[key])} candidate pairs\n{'=' * 100}")
        for _, _, data, rec, label, rel, gold_label in sorted(found[key], key=lambda x: (-x[0], -x[1]))[: args.k]:
            gold_note = f"   [gold relation: {gold_label}]" if gold_label else ("   [not in the gold set]" if args.batch else "")
            print(f"\n[{data.topic.id}]{gold_note}")
            print(format_pair(data, rec, label, rel))


def _annotation_dir(batch: str):
    from app.evaluation import EVAL_DIR

    return EVAL_DIR / "annotation" / batch


def cmd_annotate_sample(args: argparse.Namespace) -> None:
    """Draw the stratified evaluation sample (from the frozen systems' outputs) and write its key."""
    import json

    from app.evaluation import model_annotation as ma

    out_dir = _annotation_dir(args.batch)
    key_path = out_dir / f"{args.batch}_key.json"
    if key_path.exists() and not args.force:
        raise SystemExit(f"{key_path} exists; refusing to redraw a sample that may already be annotated (use --force)")
    datas = [exp.load_topic(args.run, tid) for tid in _topic_ids(args.topic, args.split)]
    items, design = ma.sample_eval_set(datas, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({"batch": args.batch, "run": args.run, "seed": args.seed, "design": design,
                                    "strata": dict(ma.STRATA), "items": [i.__dict__ for i in items]}, indent=1))
    for name, desc in ma.STRATA:
        print(f"  {name:28} pool {design[name]['pool']:4d}  sampled {design[name]['sampled']:3d}   {desc}")
    print(f"{len(items)} pairs → {key_path}")


def _load_batch(batch: str):
    import json

    from app.evaluation import model_annotation as ma
    from app.evaluation.annotation import Item

    out_dir = _annotation_dir(batch)
    key = json.loads((out_dir / f"{batch}_key.json").read_text())
    items = [Item(**i) for i in key["items"]]
    datas = {tid: exp.load_topic(key["run"], tid) for tid in {i.topic for i in items}}
    return out_dir, key, items, ma.payloads_for(items, datas)


def cmd_annotate_run(args: argparse.Namespace) -> None:
    """Have the three annotator models label the sample, blind and zero-shot. Prints a cost estimate first."""
    from app.evaluation import model_annotation as ma

    settings = Settings()
    if not settings.llm_api_key:
        raise SystemExit("no OpenRouter key: set CLAIMGRAPH_LLM_API_KEY in .env")
    out_dir, _, items, payloads = _load_batch(args.batch)
    chars = sum(len(ma.ANNOTATION_SYSTEM) + len(ma.annotation_prompt(p)) for p in payloads.values())
    in_tok = chars / 4  # rough: 4 characters per token
    print(f"{len(items)} pairs × {len(ma.ANNOTATORS)} annotators = {len(items) * len(ma.ANNOTATORS)} calls; "
          f"about {in_tok / len(items):.0f} input tokens per call")
    est_total = 0.0
    for a in ma.ANNOTATORS:
        price = ma.PRICES_PER_M.get(a.model)
        if price:
            est = (in_tok * price[0] + len(items) * args.est_output_tokens * price[1]) / 1e6
            est_total += est
            print(f"  {a.id:7} {a.model:32} ${price[0]:.2f}/M in, ${price[1]:.2f}/M out → about ${est:.2f}")
    print(f"estimated total: about ${est_total:.2f} (assuming {args.est_output_tokens} output tokens per call, reasoning included)")
    if args.dry_run:
        return
    if est_total > args.max_cost:
        raise SystemExit(f"estimate exceeds --max-cost ${args.max_cost:.2f}; nothing was sent")
    if args.limit:
        payloads = dict(sorted(payloads.items())[: args.limit])
    res = ma.run_annotation(payloads, out_dir, settings.llm_api_key, concurrency=args.concurrency,
                            max_calls=int(len(payloads) * len(ma.ANNOTATORS) * 1.5) + 10, progress=lambda m: print("  " + m, flush=True))
    print(f"labels: {res['labels']} (requested now: {res['requested']}, HTTP calls now: {res['http_calls']}, failures: {len(res['failures'])})")
    for f in res["failures"][:8]:
        print("  failure:", f)
    usage = ma.usage_summary(out_dir)
    for a, u in usage["per_annotator"].items():
        print(f"  {a:7} {u['labels']} labels, {u['http_calls']} HTTP calls, {u['prompt_tokens']} in / {u['completion_tokens']} out tokens, ${u['cost']:.4f}")
    print(f"  total: {usage['total']['http_calls']} HTTP calls, ${usage['total']['cost']:.4f}")


def cmd_annotate_gold(args: argparse.Namespace) -> None:
    """Consensus, disagreement flags, adjudication worklist and the gold file."""
    import json

    from app.evaluation import model_annotation as ma

    out_dir, key, items, payloads = _load_batch(args.batch)
    doc = ma.build_gold(out_dir, key["items"])
    work = ma.write_adjudication_files(out_dir, key["items"], payloads, doc, spot_checks=args.spot_checks, batch=args.batch)
    report = ma.agreement_report(doc)
    (out_dir / "agreement.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"adjudication worklist: {work['flagged']} flagged + {work['spot_checks']} spot checks = {work['worklist']} pairs "
          f"→ {out_dir / 'adjudicate.html'} (export as adjudicated.json into that folder, then run this command again)")


def cmd_explain_audit(args: argparse.Namespace) -> None:
    """Reliability of the explanation layer: construction checks, acceptance statistics, independent judges."""
    import json

    from app.evaluation import EVAL_DIR
    from app.evaluation import explanation_audit as ea

    datas = {tid: exp.load_topic(args.run, tid) for tid in _topic_ids(args.topic, args.split)}
    out_dir = EVAL_DIR / "results" / args.name
    rows = ea.collect(datas)
    if not args.no_judges:
        settings = Settings()
        print(f"{len(rows)} explanations × {len(ea.JUDGES)} judges ({', '.join(j.model for j in ea.JUDGES)})")
        print(ea.run_judges(rows, out_dir, settings.llm_api_key, concurrency=args.concurrency))
    summary = ea.summarise(datas, out_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "flagged_by_both_judges"}, indent=1))
    print(f"flagged by both judges: {len(summary['flagged_by_both_judges'])}; written: {out_dir / 'explanation_audit.json'}")


def cmd_rejections(args: argparse.Namespace) -> None:
    """Why the context gate did not assert the face-value conflicts it saw."""
    from app.evaluation import EVAL_DIR
    from app.evaluation import rejections, scoring

    datas = {tid: exp.load_topic(args.run, tid) for tid in _topic_ids(args.topic, args.split)}
    gs = scoring.load_gold(EVAL_DIR / "annotation" / args.batch) if args.batch else None
    out_dir = EVAL_DIR / "results" / (args.name or (f"{args.batch}_{args.run}" if args.batch else f"rejections_{args.run}"))
    summary = rejections.analyse(datas, out_dir, gs)
    print((out_dir / "rejection_analysis.md").read_text().split("## Representative examples")[0])
    print(f"written: {out_dir / 'rejection_analysis.md'} ({summary['conflicts']} face-value conflicts)")


def cmd_v2_study(args: argparse.Namespace) -> None:
    """Development study for the V2 comparability model (offline; no LLM calls)."""
    from app.evaluation import EVAL_DIR
    from app.evaluation.comparability_study import run_study

    out_dir = EVAL_DIR / "results" / args.name
    res = run_study(args.batch, args.run, out_dir, n_perm=args.shuffles, n_boot=args.bootstrap)
    print((out_dir / "comparability_study.md").read_text().split("## 1.")[0])
    print(f"selected by the rule: {res['selection']['selected']}; written: {out_dir / 'comparability_study.md'}")


def cmd_fourstate_annotate(args: argparse.Namespace) -> None:
    """Blind four-state labels (stance, sufficiency, comparability, status, factors) from three annotator models."""
    import json

    from app.evaluation import EVAL_DIR
    from app.evaluation import conflict_annotation as ca
    from app.evaluation.model_annotation import FORBIDDEN_IN_PROMPT, annotation_prompt

    out = EVAL_DIR / "annotation" / args.batch
    out.mkdir(parents=True, exist_ok=True)
    key_path = out / f"{args.batch}_key.json"
    topics, _ = corpus_mod.load_topics()
    run_of = lambda t: {"dev": "dev", "fresh": "fresh"}.get(t.split, "main")  # noqa: E731
    datas = {(run_of(t), t.id): exp.load_topic(run_of(t), t.id) for t in topics if exp.topic_dir(run_of(t), t.id).exists()}
    if not key_path.exists():
        if args.batch != "c4":
            raise SystemExit(f"no sample for batch {args.batch!r}: draw it first (fresh-sample)")
        prior = EVAL_DIR / "annotation" / "eval2"
        items, design = ca.sample(datas, json.loads((prior / "gold.json").read_text())["items"],
                                  json.loads((prior / "eval2_key.json").read_text())["items"], n_no_signal=args.no_signal)
        key_path.write_text(json.dumps({"batch": args.batch, "protocol": ca.PROTOCOL, "seed": 2027, "strata": ca.STRATA, "design": design, "items": items}, indent=1))
    key = json.loads(key_path.read_text())
    payloads = ca.payloads(key["items"], datas)
    def outside_paper_text(p: dict) -> str:
        """The prompt minus the papers' own words: a paper may itself say 'similarity' or 'explanation'."""
        text = annotation_prompt(p)
        for side in (p["a"], p["b"]):
            for value in sorted((v for v in side.values() if isinstance(v, str) and len(v) > 20), key=len, reverse=True):
                text = text.replace(value, " ").replace(value[:1200], " ")
        return text

    leaks = [w for p in payloads.values() for w in FORBIDDEN_IN_PROMPT if w in outside_paper_text(p) and w != "explanation"]
    assert not leaks and not [w for w in FORBIDDEN_IN_PROMPT if w.lower() in ca.SYSTEM.lower()], "system output in an annotation prompt"
    todo = 3 * len(payloads) - len(ca.read_calls(out))
    print(f"{len(payloads)} pairs, {todo} calls to make; about US${0.0040 * todo / 3:.2f} at the measured US$0.0040 per pair; cap US${args.max_cost}")
    if args.execute and todo:
        print(ca.run(payloads, out, Settings().llm_api_key, max_calls=int(todo * 1.25) + 10, max_cost=args.max_cost, concurrency=args.concurrency))
    doc = ca.build_reference(out, key["items"])
    (out / "reference.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    (out / "agreement.json").write_text(json.dumps(ca.agreement(doc), indent=1))
    print(f"reference labels for {len(doc['items'])} pairs; spent so far: US${ca.spent(out)['total']['cost']:.3f}; written: {out / 'reference.json'}")


def cmd_fourstate_study(args: argparse.Namespace) -> None:
    """Sufficiency, comparability and end-to-end four-state performance against the four-state labels (offline)."""
    from app.evaluation import EVAL_DIR
    from app.evaluation.fourstate_study import run_study

    out_dir = EVAL_DIR / "results" / args.name
    run_study(args.batch, out_dir, n_boot=args.bootstrap, n_perm=args.shuffles)
    print((out_dir / "fourstate_study.md").read_text().split("## 1.")[0])
    print(f"written: {out_dir / 'fourstate_study.md'}")


def cmd_layer_study(args: argparse.Namespace) -> None:
    """E1-E3 on the four-state labels: opposition detection, factor layer, conditional-prediction frames (offline)."""
    from app.evaluation import EVAL_DIR
    from app.evaluation.layer_study import run_study

    out_dir = EVAL_DIR / "results" / args.name
    run_study(args.batch, out_dir, n_boot=args.bootstrap, n_perm=args.shuffles, parts=tuple(args.parts.split(",")))
    print(f"written: {out_dir / 'layer_study.md'}")


def cmd_fresh_sample(args: argparse.Namespace) -> None:
    """Draw the pre-registered stratified sample of fresh-topic pairs for the E1 validation (no LLM call)."""
    import hashlib
    import json
    from pathlib import Path

    from app.evaluation import EVAL_DIR
    from app.evaluation import conflict_annotation as ca
    from app.evaluation import fresh_validation as fv
    from app.evaluation.frame_extraction import key_usage

    out = EVAL_DIR / "annotation" / args.batch
    key_path = out / f"{args.batch}_key.json"
    if key_path.exists():
        raise SystemExit(f"{key_path} exists: the sample is drawn once")
    topics, _ = corpus_mod.load_topics()
    datas = {(fv.RUN, t.id): exp.load_topic(fv.RUN, t.id) for t in topics if t.split == "fresh"}
    used = key_usage(Settings().llm_api_key)
    if used is None:
        raise SystemExit("cannot read the key balance")
    n_max = min(100, int((args.limit_usd - used - fv.RESERVE) / fv.COST_PER_PAIR))
    items, design = fv.sample(datas, n_max, args.batch)
    design.update(n_max=n_max, key_usage_before=used, key_limit=args.limit_usd, seed=fv.SEED,
                  scorer_sha256=hashlib.sha256(Path(fv.__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({"batch": args.batch, "protocol": ca.PROTOCOL, "strata": fv.STRATA, "design": design, "items": items}, indent=1))
    print(json.dumps(design, indent=1))


def cmd_fresh_validate(args: argparse.Namespace) -> None:
    """Score the frozen E1 detectors on the fresh labels (offline; run once)."""
    from app.evaluation import EVAL_DIR
    from app.evaluation.fresh_validation import run_validation

    out_dir = EVAL_DIR / "results" / args.name
    run_validation(args.batch, out_dir, n_boot=args.bootstrap)
    print((out_dir / "validation.md").read_text())


def cmd_frames_extract(args: argparse.Namespace) -> None:
    """E3: one conditional-prediction frame per distinct claim of the annotation batch (cached; one call per claim)."""
    from app.evaluation import frame_extraction as fx

    settings = Settings()
    _, claims = fx.batch_claims(args.batch)
    todo = [c for c in claims if "/".join(c) not in fx.load_raw(args.batch)]
    print(f"{len(claims)} distinct claims, {len(todo)} without a stored frame")
    if args.execute and todo:
        before = fx.key_usage(settings.llm_api_key)
        print(fx.extract(args.batch, eval_llm(settings), limit=args.limit, concurrency=args.concurrency))
        after = fx.key_usage(settings.llm_api_key)
        if before is not None and after is not None:
            print(f"key usage: US${before:.4f} -> US${after:.4f} (this pass: US${after - before:.4f})")
    frames = fx.load_frames(args.batch)
    print(f"valid frames: {len(frames)}; effects: {dict(Counter(f.effect for f in frames.values()))}")


def cmd_v2_freeze(args: argparse.Namespace) -> None:
    """Freeze the development-calibrated parameters of the comparability model for use on held-out pairs."""
    import json

    from app.evaluation import EVAL_DIR
    from app.evaluation.comparability_study import HAND_WEIGHTS
    from app.evaluation.experiment import V2_PARAMS_PATH
    from app.reasoning.comparability import ComparabilityParams, save_params

    study = json.loads((EVAL_DIR / "results" / args.name / "comparability_study.json").read_text())
    prov = {"calibrated_on": f"batch {study['batch']} (run {study['run']}): opposed pairs with shared context, consensus labels",
            "fit": f"penalised logistic regression, ridge {study['l2']}, non-negative weights",
            "threshold_rule": "lowest p at which at least half of the asserted development pairs are consensus contradictions; none -> never asserts",
            "selected_by_rule": study["selection"]["selected"]}
    params = []
    for name in args.models.split(","):
        fit = study["p3_parameters"][name]
        cols = dict(fit["columns"])
        if "diff:hand-weighted" in cols:  # one fitted scale on the V1 hand weights
            scale = cols.pop("diff:hand-weighted")
            cols.update({f"diff:{factor}": scale * w for factor, w in HAND_WEIGHTS.items()})
        params.append(ComparabilityParams.from_fit(name, fit["intercept"], cols, fit["threshold"], {**prov, "model": fit["title"]}))
    save_params(V2_PARAMS_PATH, params, "Comparability-model parameters calibrated on development data. Do not refit on test pairs.")
    for q in params:
        print(f"{q.name}: intercept {q.intercept:+.3f}, same {q.weight_same:.3f}, mismatch {q.weights_different}, threshold {q.as_dict()['threshold']}")
    print(f"written: {V2_PARAMS_PATH}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cg_eval", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def topic_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--topic", help="comma-separated topic ids (default: all topics of --split)")
        sp.add_argument("--split", choices=["dev", "test", "all"], default="all")

    sp = sub.add_parser("build-corpus", help="discover + download papers and freeze a corpus snapshot")
    topic_args(sp)
    sp.add_argument("--n-papers", type=int)
    sp.add_argument("--seeds-only", action="store_true", help="skip automatic discovery")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_build_corpus)

    sp = sub.add_parser("describe", help="list the papers in frozen corpora")
    topic_args(sp)
    sp.set_defaults(func=cmd_describe)

    def run_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--run", default=DEFAULT_RUN, help="name of the experiment run (eval/runs/<run>)")

    sp = sub.add_parser("prepare", help="extract claims and candidate pairs (+ similarity, NLI) from frozen corpora")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("--max-claims", type=int, default=8, help="claims kept per paper")
    sp.add_argument("--max-pairs", type=int, default=200, help="candidate pairs kept per topic (most similar first)")
    sp.add_argument("--rule-claims", action="store_true", help="select claims with the cue rules only (no LLM screening)")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("report", help="print claims, contexts, pairs, signals, decisions and evidence for a topic")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("--system", default="hybrid_full", choices=[x.name for x in exp.SYSTEM_SPECS])
    sp.add_argument("--all", action="store_true", help="also list pairs judged UNRELATED")
    sp.add_argument("--limit", type=int, help="show at most N pairs")
    sp.add_argument("--no-claims", action="store_true", help="skip the claim listing")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("sheet", help="sample pairs and write blind annotation sheets (HTML tool + CSV) per annotator")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("--batch", required=True, help="name of the annotation batch, e.g. test1")
    sp.add_argument("--annotators", required=True, help="comma-separated annotator ids (no spaces or underscores)")
    sp.add_argument("--full-assert", type=int, default=-1, help="per topic: pairs the full system calls CONTRADICTS or "
                    "abstains on (default -1: all of them)")
    sp.add_argument("--gate-input", type=int, default=10, help="per topic: other face-value conflicts seen by the full system")
    sp.add_argument("--conflict", type=int, default=8, help="per topic: pairs only another system flags as a conflict")
    sp.add_argument("--agreement", type=int, default=8, help="per topic: pairs some system calls agreement")
    sp.add_argument("--background", type=int, default=6, help="per topic: pairs no system relates")
    sp.add_argument("--per-item", type=int, default=2, help="annotators per item")
    sp.add_argument("--overlap", type=int, default=30, help="items labelled by every annotator")
    sp.add_argument("--seed", type=int, default=13)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_sheet)

    sp = sub.add_parser("score", help="score all systems against the gold labels of an annotation batch")
    sp.add_argument("--batch", required=True)
    sp.add_argument("--split", choices=["dev", "test", "all"], default="all")
    run_arg(sp)
    sp.add_argument("--gold", default="final",
                    help="model-consensus batches: 'final' = adjudicated + unanimous + provisional majority labels; "
                         "'unanimous' = only unanimous or adjudicated labels; 'annotator:<id>' = one annotator model's "
                         "labels (robustness check)")
    sp.add_argument("--name", help="results folder name (default: <batch>_<run>)")
    sp.add_argument("--title")
    sp.add_argument("--bootstrap", type=int, default=2000)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("explain", help="grounded LLM explanations for the full system's face-value conflicts")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--concurrency", type=int)
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("annotate-sample", help="draw the stratified evaluation sample for model-assisted annotation")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("--batch", required=True)
    sp.add_argument("--seed", type=int, default=2026)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_annotate_sample)

    sp = sub.add_parser("annotate-run", help="blind zero-shot labels from three annotator models (prints a cost estimate first)")
    sp.add_argument("--batch", required=True)
    sp.add_argument("--dry-run", action="store_true", help="only print the call count and cost estimate")
    sp.add_argument("--max-cost", type=float, default=1.5, help="refuse to run if the estimate exceeds this many dollars")
    sp.add_argument("--est-output-tokens", type=int, default=250)
    sp.add_argument("--limit", type=int, help="only the first N pairs (smoke test)")
    sp.add_argument("--concurrency", type=int, default=6)
    sp.set_defaults(func=cmd_annotate_run)

    sp = sub.add_parser("annotate-gold", help="consensus labels, agreement, adjudication worklist and gold.json")
    sp.add_argument("--batch", required=True)
    sp.add_argument("--spot-checks", type=int, default=20, help="unanimous pairs the researcher re-checks")
    sp.set_defaults(func=cmd_annotate_gold)

    sp = sub.add_parser("explain-audit", help="reliability of the explanation layer (checks + two independent judge models)")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("--name", required=True, help="results folder")
    sp.add_argument("--no-judges", action="store_true")
    sp.add_argument("--concurrency", type=int, default=6)
    sp.set_defaults(func=cmd_explain_audit)

    sp = sub.add_parser("rejections", help="why the context gate did not assert the face-value conflicts it saw")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("--batch", help="annotation batch: also relate the gate's verdicts to gold labels")
    sp.add_argument("--name")
    sp.set_defaults(func=cmd_rejections)

    sp = sub.add_parser("v2-study", help="development study of the V2 comparability model on an annotated batch (offline)")
    sp.add_argument("--batch", default="eval2")
    run_arg(sp)
    sp.add_argument("--name", default="v2_dev")
    sp.add_argument("--shuffles", type=int, default=500)
    sp.add_argument("--bootstrap", type=int, default=2000)
    sp.set_defaults(func=cmd_v2_study)

    sp = sub.add_parser("fourstate-annotate", help="blind four-state conflict labels from three annotator models (prints the cost first)")
    sp.add_argument("--batch", default="c4")
    sp.add_argument("--no-signal", type=int, default=30, help="random sample size of pairs nobody saw as opposed")
    sp.add_argument("--max-cost", type=float, default=0.95)
    sp.add_argument("--concurrency", type=int, default=8)
    sp.add_argument("--execute", action="store_true", help="make the calls (without it: estimate and rebuild the reference only)")
    sp.set_defaults(func=cmd_fourstate_annotate)

    sp = sub.add_parser("fourstate-study", help="four-state development study: sufficiency, comparability, end to end (offline)")
    sp.add_argument("--batch", default="c4")
    sp.add_argument("--name", default="v2_fourstate")
    sp.add_argument("--shuffles", type=int, default=1000)
    sp.add_argument("--bootstrap", type=int, default=2000)
    sp.set_defaults(func=cmd_fourstate_study)

    sp = sub.add_parser("layer-study", help="E1-E3 on the four-state labels (opposition, factor layer, frames)")
    sp.add_argument("--batch", default="c4")
    sp.add_argument("--name", default="c4_layers")
    sp.add_argument("--parts", default="e1,e2,e3")
    sp.add_argument("--shuffles", type=int, default=1000)
    sp.add_argument("--bootstrap", type=int, default=2000)
    sp.set_defaults(func=cmd_layer_study)

    sp = sub.add_parser("fresh-sample", help="draw the pre-registered sample of fresh-topic pairs (E1 validation)")
    sp.add_argument("--batch", default="f1")
    sp.add_argument("--limit-usd", type=float, default=2.0, help="spending limit of the API key")
    sp.set_defaults(func=cmd_fresh_sample)

    sp = sub.add_parser("fresh-validate", help="score the frozen E1 detectors on the fresh labels")
    sp.add_argument("--batch", default="f1")
    sp.add_argument("--name", default="f1_validation")
    sp.add_argument("--bootstrap", type=int, default=2000)
    sp.set_defaults(func=cmd_fresh_validate)

    sp = sub.add_parser("frames-extract", help="E3: per-claim conditional-prediction frames for an annotation batch")
    sp.add_argument("--batch", default="c4")
    sp.add_argument("--execute", action="store_true", help="make the LLM calls (otherwise only report what is stored)")
    sp.add_argument("--limit", type=int, help="only the first N missing claims (pilot)")
    sp.add_argument("--concurrency", type=int, default=6)
    sp.set_defaults(func=cmd_frames_extract)

    sp = sub.add_parser("v2-freeze", help="freeze development-calibrated comparability parameters into eval/v2_params.json")
    sp.add_argument("--name", default="v2_dev")
    sp.add_argument("--models", default="uniform_ctx,hand,role")
    sp.set_defaults(func=cmd_v2_freeze)

    sp = sub.add_parser("examples", help="list candidate qualitative examples (context-dependent / genuine / insufficient)")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("-k", type=int, default=3, help="examples per category")
    sp.add_argument("--batch", help="annotation batch: show gold labels next to each example")
    sp.add_argument("--confirmed", action="store_true", help="with --batch: only examples the gold relation bears out")
    sp.set_defaults(func=cmd_examples)

    sp = sub.add_parser("llm", help="add the (cached) LLM signals to prepared pairs")
    topic_args(sp)
    run_arg(sp)
    sp.add_argument("--fields", help=f"comma-separated subset of {','.join(exp.LLM_FIELDS)}")
    sp.add_argument("--limit", type=int, help="only the first N pairs per topic (smoke tests)")
    sp.add_argument("--refresh", action="store_true", help="discard stored LLM outputs of these fields and recompute")
    sp.add_argument("--concurrency", type=int)
    sp.set_defaults(func=cmd_llm)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args.func(args)
