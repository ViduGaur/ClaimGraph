"""Claim graph construction with NetworkX.

Nodes: papers and claims. Edges: paper -MAKES-> claim, and typed claim→claim
relations (SUPPORTS / CONTRADICTS / QUALIFIES / EXTENDS). NetworkX is used for
graph analytics (centrality, components, bridging papers); the serialised form
is what the frontend renders with Cytoscape.js.
"""
from __future__ import annotations

import networkx as nx

from app.schemas.domain import Claim, ClaimGraphModel, GraphEdge, GraphNode, Paper, Relationship


def build_graph(papers: list[Paper], claims: list[Claim], relationships: list[Relationship]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for p in papers:
        g.add_node(p.paper_id, kind="paper", label=p.citation, title=p.title, year=p.year)
    for c in claims:
        g.add_node(c.claim_id, kind="claim", label=c.normalized_text, paper_id=c.paper_id, claim_type=c.claim_type.value,
                   confidence=c.confidence, polarity=c.polarity.value)
        if c.paper_id in g:
            g.add_edge(c.paper_id, c.claim_id, key="MAKES", kind="MAKES")
    for r in relationships:
        if r.source_claim_id in g and r.target_claim_id in g:
            g.add_edge(r.source_claim_id, r.target_claim_id, key=r.relation_id, kind=r.relation.value,
                       relation_id=r.relation_id, confidence=r.confidence, uncertain=r.uncertain)
    return g


def graph_stats(g: nx.MultiDiGraph) -> dict:
    claim_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "claim"]
    rel_edges = [(u, v, d) for u, v, d in g.edges(data=True) if d["kind"] != "MAKES"]
    rel_graph = nx.Graph()
    rel_graph.add_nodes_from(claim_nodes)
    rel_graph.add_edges_from((u, v) for u, v, _ in rel_edges)
    centrality = nx.degree_centrality(rel_graph) if rel_graph.number_of_nodes() > 1 else {}
    top_claims = [n for n, c in sorted(centrality.items(), key=lambda x: x[1], reverse=True) if c > 0][:5]

    # papers whose claims connect to the most *other* papers (bridges in the literature)
    paper_links: dict[str, set[str]] = {}
    for u, v, _ in rel_edges:
        pu, pv = g.nodes[u].get("paper_id"), g.nodes[v].get("paper_id")
        if pu and pv and pu != pv:
            paper_links.setdefault(pu, set()).add(pv)
            paper_links.setdefault(pv, set()).add(pu)
    bridging = sorted(paper_links, key=lambda p: len(paper_links[p]), reverse=True)[:3]
    counts: dict[str, int] = {}
    for _, _, d in rel_edges:
        counts[d["kind"]] = counts.get(d["kind"], 0) + 1
    connected = [c for c in nx.connected_components(rel_graph) if len(c) > 1]
    return {
        "papers": sum(1 for _, d in g.nodes(data=True) if d["kind"] == "paper"),
        "claims": len(claim_nodes),
        "relationships": len(rel_edges),
        "by_relation": counts,
        "connected_claim_clusters": len(connected),
        "isolated_claims": sum(1 for n in claim_nodes if rel_graph.degree(n) == 0),
        "most_connected_claims": top_claims,
        "bridging_papers": bridging,
        "density": round(nx.density(rel_graph), 4) if rel_graph.number_of_nodes() > 1 else 0.0,
    }


def serialize_graph(g: nx.MultiDiGraph) -> ClaimGraphModel:
    nodes = [GraphNode(id=n, kind=d["kind"], label=d["label"], data={k: v for k, v in d.items() if k not in ("kind", "label")})
             for n, d in g.nodes(data=True)]
    edges = []
    for u, v, k, d in g.edges(keys=True, data=True):
        edges.append(GraphEdge(id=f"makes__{u}__{v}" if d["kind"] == "MAKES" else d["relation_id"], source=u, target=v,
                               kind=d["kind"], data={x: y for x, y in d.items() if x != "kind"}))
    return ClaimGraphModel(nodes=nodes, edges=edges, stats=graph_stats(g))
