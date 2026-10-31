# Documentation

| Document | Read it for |
|---|---|
| [`architecture.md`](architecture.md) | design goals, code layout, data model, the relation and verdict decision rules, graph, landscape, failure handling |
| [`context-gate.md`](context-gate.md) | the exact logic of the context gate that turns a face-value conflict into a verdict, with its known quirks; pinned by `backend/tests/test_context_gate.py` |
| [`research.md`](research.md) | the literature APIs, PDF tools, local models and LLM providers that were considered, with the limits checked when they were chosen |

Related material elsewhere in the repository:

- [`../README.md`](../README.md): setup, running the app, configuration, API.
- [`../eval/README.md`](../eval/README.md): the evaluation harness, the annotation protocols, the freeze record and how to verify it.
- [`../eval/annotation/f1/PREREGISTRATION.md`](../eval/annotation/f1/PREREGISTRATION.md): the pre-registered fresh-topic validation.
