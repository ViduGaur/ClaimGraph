import type { AnalysisResult, Claim, ClaimContext } from '../types'
import { authorsShort, ClaimTypeChip, Confidence, doiUrl, EvidenceQuote, RelationLabel } from './ui'

const CTX_FIELDS: [keyof ClaimContext, string][] = [
  ['datasets', 'Dataset'], ['data_scale', 'Data scale'], ['conditions', 'Conditions'], ['training', 'Training'],
  ['models', 'Model / method'], ['baselines', 'Baseline'], ['metrics', 'Metric'], ['task', 'Task'],
  ['population', 'Population'], ['sample_size', 'Sample size'], ['methodology', 'Methodology'], ['time_period', 'Time period'],
]

/** Only the context that was actually found — "not stated" fields are summarised in one line. */
export function ContextChips({ claim }: { claim: Claim }) {
  const rows = CTX_FIELDS.map(([key, label]) => {
    const v = claim.context[key]
    return { key, label, values: Array.isArray(v) ? v : v ? [v] : [], inherited: claim.inherited_context.includes(key) }
  })
  const found = rows.filter((r) => r.values.length)
  const missing = rows.filter((r) => !r.values.length).map((r) => r.label.toLowerCase())
  return (
    <div className="ctx">
      {found.map((r) => (
        <div key={r.key} className="ctx-row">
          <span className="ctx-label">{r.label}</span>
          <span className="ctx-values">
            {r.values.map((x) => <span key={x} className="tag">{x}</span>)}
            {r.inherited && <span className="muted small" title="Not stated in this claim's passage; taken from the paper's abstract/methods">from paper</span>}
          </span>
        </div>
      ))}
      {missing.length > 0 && <p className="muted small">Not stated: {missing.join(', ')}</p>}
    </div>
  )
}

export function ClaimDetail({ result, claimId, colors, onSelectClaim, onSelectRelation }: {
  result: AnalysisResult
  claimId: string
  colors: Record<string, string>
  onSelectClaim: (id: string) => void
  onSelectRelation?: (id: string) => void
}) {
  const claim = result.claims.find((c) => c.claim_id === claimId)
  if (!claim) return <p className="muted">Claim not found.</p>
  const paper = result.papers.find((p) => p.paper_id === claim.paper_id)!
  const claims = Object.fromEntries(result.claims.map((c) => [c.claim_id, c]))
  const rels = result.relationships.filter((r) => r.source_claim_id === claimId || r.target_claim_id === claimId)
  const doi = doiUrl(paper.doi)

  return (
    <div className="detail">
      <div className="paper-line">
        <span className="swatch" style={{ background: colors[paper.paper_id] }} />
        <b>{claim.source_citation}</b>
        <span className="muted">· § {claim.section.replace('_', ' ')}</span>
      </div>
      <p className="detail-claim">{claim.claim_text}</p>
      <div className="claim-meta">
        <ClaimTypeChip type={claim.claim_type} />
        <Confidence value={claim.confidence} />
        {claim.reported_result && <span className="tag mono">{claim.reported_result}</span>}
      </div>

      <section>
        <h4>Evidence in context</h4>
        {claim.evidence.map((p) => <EvidenceQuote key={p.passage_id} passage={p.text} claim={claim.claim_text} />)}
        <p className="muted small" style={{ marginTop: 6 }}>
          {claim.extraction_method === 'llm' ? 'LLM-extracted, verified against the passage' : 'Rule-based extraction'} ·
          text source: {paper.text_source.replace('_', ' ')}
        </p>
      </section>

      {claim.uncertainty_notes.length > 0 && (
        <div className="callout warn">
          {claim.uncertainty_notes.map((n) => <div key={n}>{n}</div>)}
        </div>
      )}

      <section>
        <h4>Relationships <span className="count-badge">{rels.length}</span></h4>
        {rels.length === 0 && <p className="muted small">No cross-paper relationship for this claim.</p>}
        <div className="rel-list">
          {rels.map((r) => {
            const otherId = r.source_claim_id === claimId ? r.target_claim_id : r.source_claim_id
            const other = claims[otherId]
            return (
              <button key={r.relation_id} className="rel-item" onClick={() => (onSelectRelation ? onSelectRelation(r.relation_id) : onSelectClaim(otherId))}>
                <span className="rel-item-head">
                  <RelationLabel relation={r.relation} uncertain={r.uncertain} />
                  <span className="paper-line small">
                    <span className="swatch sm" style={{ background: colors[other.paper_id] }} />{other.source_citation}
                  </span>
                </span>
                <span className="rel-item-text">{other.claim_text}</span>
              </button>
            )
          })}
        </div>
      </section>

      <section>
        <h4>Extracted context</h4>
        <ContextChips claim={claim} />
      </section>

      <section>
        <h4>Source</h4>
        <p style={{ fontWeight: 600 }}>{paper.title}</p>
        <p className="secondary small">{authorsShort(paper.authors, 5)} · {paper.year ?? 'n.d.'}{paper.venue ? ` · ${paper.venue}` : ''}</p>
        <p className="small links">
          {doi && <a href={doi} target="_blank" rel="noreferrer">DOI ↗</a>}
          {paper.url && <a href={paper.url} target="_blank" rel="noreferrer">Source page ↗</a>}
          {paper.pdf_url && <a href={paper.pdf_url} target="_blank" rel="noreferrer">PDF ↗</a>}
        </p>
      </section>
    </div>
  )
}
