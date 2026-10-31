"""Model-assisted annotation: blinding, sampling strata, consensus and adjudication (no network)."""
from __future__ import annotations

import json

from app.evaluation import model_annotation as ma
from app.evaluation.annotation import Item, item_payload
from app.evaluation.corpus import Topic
from app.evaluation.experiment import TopicData, pair_id
from app.evaluation.metrics import ABSTAIN
from tests.conftest import make_claim, make_paper

CON = {"entailment": 0.02, "contradiction": 0.9, "neutral": 0.08}


def topic_data() -> TopicData:
    a = make_claim("a", "p1", "Method X improves accuracy on benchmark one.")
    b = make_claim("b", "p2", "Method X does not improve accuracy on benchmark two.")
    rec = {"pair_id": pair_id("a", "b"), "topic": "t", "a": "a", "b": "b", "similarity": 0.91, "nli_ab": CON, "nli_ba": CON,
           "llm_pair": {"stance": "oppose", "same_relationship": True, "factors": [], "conclusion_a": "SECRET-A", "conclusion_b": "SECRET-B"},
           "llm_direct_context": {"label": "CONTRADICTS", "reason": "SECRET-REASON"},
           "explanation": {"explanation": "SECRET-EXPLANATION", "conflict_verdict": "insufficient"}}
    return TopicData(Topic("t", "test", "Does X help?", []), {"p1": make_paper("p1"), "p2": make_paper("p2")}, {"a": a, "b": b}, [rec], 0.72)


def test_annotation_prompt_contains_only_claims_and_evidence():
    data = topic_data()
    payload = item_payload(Item("ev_1", data.records[0]["pair_id"], "t", "abstained"), data)
    prompt = ma.annotation_prompt(payload)
    assert "Method X improves accuracy" in prompt and "does not improve" in prompt and "An abstract." in prompt
    blob = (ma.ANNOTATION_SYSTEM + prompt)
    for leaked in ("SECRET", "0.91", "0.9,", "oppose", "abstained", "insufficient", *ma.FORBIDDEN_IN_PROMPT):
        assert leaked not in blob, leaked
    # zero-shot: the instructions define the labels and give no worked pair
    assert "CLAIM A" not in ma.ANNOTATION_SYSTEM and "Example" not in ma.ANNOTATION_SYSTEM


def test_three_different_providers_and_none_is_the_pipeline_model():
    assert len({a.model.split("/")[0] for a in ma.ANNOTATORS}) == 3
    assert all("ling" not in a.model for a in ma.ANNOTATORS)


def test_strata_follow_the_systems_outputs():
    base = {"hybrid_full": "UNRELATED", "hybrid_no_gate": "UNRELATED", "nli": "UNRELATED", "llm_direct_claims": "UNRELATED",
            "llm_direct_context": "UNRELATED", "rules_full": "UNRELATED"}
    cases = {"asserted_contradiction": {"hybrid_full": "CONTRADICTS", "hybrid_no_gate": "CONTRADICTS"},
             "abstained": {"hybrid_full": ABSTAIN, "hybrid_no_gate": "CONTRADICTS"},
             "gate_rejected_nli_contra": {"hybrid_full": "QUALIFIES", "hybrid_no_gate": "CONTRADICTS", "nli": "CONTRADICTS"},
             "gate_rejected": {"hybrid_full": "QUALIFIES", "hybrid_no_gate": "CONTRADICTS"},
             "nli_contra_no_conflict": {"nli": "CONTRADICTS"},
             "direct_contra_no_conflict": {"llm_direct_context": "CONTRADICTS"},
             "qualifies_partial": {"hybrid_full": "QUALIFIES", "hybrid_no_gate": "QUALIFIES"},
             "extends": {"hybrid_full": "EXTENDS"}, "supports": {"hybrid_full": "SUPPORTS"},
             "others_relate": {"llm_direct_claims": "SUPPORTS"}, "all_unrelated": {}}
    assert set(cases) == {name for name, _ in ma.STRATA}
    for name, change in cases.items():
        assert ma.stratum_of({**base, **change}) == name


def write_calls(tmp_path, votes: dict[str, list[str]], fvo: bool = True) -> None:
    with (tmp_path / "model_calls.jsonl").open("w") as fh:
        for iid, labels in votes.items():
            for annotator, label in zip(("gemini", "qwen", "haiku"), labels):
                fh.write(json.dumps({"annotator": annotator, "model": f"m/{annotator}", "item_id": iid, "label": label,
                                     "face_value_opposition": fvo, "justification": "because", "usage": {"http_calls": 1, "cost": 0.001}}) + "\n")


def test_consensus_flags_and_adjudication(tmp_path):
    votes = {"u": ["QUALIFIES"] * 3, "m": ["CONTRADICTS", "CONTRADICTS", "QUALIFIES"], "s": ["SUPPORTS", "EXTENDS", "UNRELATED"]}
    write_calls(tmp_path, votes)
    items = [{"item_id": k} for k in votes]
    gold = ma.build_gold(tmp_path, items)["items"]
    assert (gold["u"]["final"], gold["u"]["final_source"], gold["u"]["flagged"]) == ("QUALIFIES", "unanimous", False)
    assert gold["u"]["conflict_verdict"] == "context_dependent"  # QUALIFIES on a face-value conflict
    assert (gold["m"]["final"], gold["m"]["agreement"], gold["m"]["flagged"]) == ("CONTRADICTS", "majority", True)
    assert gold["m"]["final_source"].startswith("majority") and gold["m"]["conflict_verdict"] == "genuine"
    assert gold["s"]["final"] is None and gold["s"]["final_source"] == "unresolved" and gold["s"]["flagged"]
    # the researcher resolves the split, overrules a majority and re-checks a unanimous pair
    (tmp_path / "adjudicated.csv").write_text("item_id,final_label,note\ns,UNRELATED,different outcomes\nm,QUALIFIES,B states a condition\nu,QUALIFIES,\n")
    doc = ma.build_gold(tmp_path, items)
    gold = doc["items"]
    assert [gold[k]["final"] for k in "usm"] == ["QUALIFIES", "UNRELATED", "QUALIFIES"]
    assert all(gold[k]["final_source"] == "adjudicated" for k in "usm") and gold["m"]["consensus"] == "CONTRADICTS"
    report = ma.agreement_report(doc)
    assert report["method"] == ma.METHOD_NAME and "human" not in report["method"] and "expert" not in report["method"]
    assert report["agreement_levels"] == {"unanimous": 1, "majority": 1, "split": 1}
    assert report["spot_check"] == {"checked": 1, "researcher_agrees": 1}
    assert report["adjudication"] == {"flagged": 2, "adjudicated": 2, "researcher_kept_majority": 0}
    assert ma.usage_summary(tmp_path)["total"]["http_calls"] == 9


def test_fleiss_kappa():
    assert ma.fleiss_kappa([["A", "A", "A"], ["B", "B", "B"]], ["A", "B"]) == 1.0
    assert abs(ma.fleiss_kappa([["A", "A", "B"], ["A", "B", "B"]], ["A", "B"]) - (-1 / 3)) < 1e-9
    assert ma.fleiss_kappa([], ["A", "B"]) is None


def test_adjudication_worklist_mixes_disagreements_with_blind_spot_checks(tmp_path):
    data = topic_data()
    item = Item("ev_1", data.records[0]["pair_id"], "t", "abstained")
    payloads = {"ev_1": item_payload(item, data)}
    write_calls(tmp_path, {"ev_1": ["QUALIFIES"] * 3})
    doc = ma.build_gold(tmp_path, [item.__dict__])
    work = ma.write_adjudication_files(tmp_path, [item.__dict__], payloads, doc, spot_checks=5, batch="b")
    assert work == {"flagged": 0, "spot_checks": 1, "worklist": 1}
    header = (tmp_path / "to_adjudicate.csv").read_text().splitlines()[0]
    assert "kind" not in header and "final_label" in header
    html = (tmp_path / "adjudicate.html").read_text()
    assert "SECRET" not in html and "Method X improves accuracy" in html
