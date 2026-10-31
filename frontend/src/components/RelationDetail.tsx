import type { AnalysisResult } from '../types'
import { Confidence, EvidenceQuote, RelationLabel, VerdictChip } from './ui'

export function RelationDetail({ result, relationId, colors, onSelectClaim }: {
  result: AnalysisResult
  relationId: string
  colors: Record<string, string>
  onSelectClaim: (id: string) => void
}) {
  const r = result.relationships.find((x) => x.relation_id === relationId)
  if (!r) return <p className="muted">Relationship not found.</p>
  const claims = Object.fromEntries(result.claims.map((c) => [c.claim_id, c]))
  const a = claims[r.target_claim_id], b = claims[r.source_claim_id]
  const conflict = result.conflicts.find((c) => c.relation_id === relationId)
  const differing = r.factors.filter((f) => f.status === 'different')
  const shared = r.factors.filter((f) => f.status === 'same' || f.status === 'overlap')

  return (
    <div className="detail">
      <div className="claim-meta" style={{ marginTop: 0 }}>
        <RelationLabel relation={r.relation} uncertain={r.uncertain} />
        {conflict && <VerdictChip verdict={conflict.verdict} />}
        <Confidence value={r.confidence} />
      </div>
      {conflict && <p className="detail-headline">{conflict.headline}</p>}

      {[{ label: 'Later claim', c: b }, { label: 'Earlier claim', c: a }].map(({ label, c }) => (
        <button key={c.claim_id} className="side-claim" style={{ borderLeftColor: colors[c.paper_id] }} onClick={() => onSelectClaim(c.claim_id)}>
          <span className="eyebrow">{label} · {c.source_citation}</span>
          <EvidenceQuote passage={c.claim_text} claim={c.claim_text} />
        </button>
      ))}

      <section>
        <h4>Why this label</h4>
        <p className="small secondary">{conflict ? conflict.explanation : r.reasoning}</p>
      </section>

      {(differing.length > 0 || shared.length > 0) && (
        <section>
          <h4>Context</h4>
          {differing.length > 0 && (
            <div className="factor-chips">
              <span className="small muted">Differs:</span>
              {differing.map((f) => <span key={f.factor} className="tag bad" title={`${f.a.join(', ')} vs ${f.b.join(', ')}`}>{f.factor}</span>)}
            </div>
          )}
          {shared.length > 0 && (
            <div className="factor-chips">
              <span className="small muted">Shared:</span>
              {shared.map((f) => <span key={f.factor} className="tag ok" title={`${f.a.join(', ')} / ${f.b.join(', ')}`}>{f.factor}</span>)}
            </div>
          )}
        </section>
      )}

      <section>
        <h4>Signals</h4>
        <div className="signal-row">
          <div><b>{r.signals.similarity.toFixed(2)}</b>similarity</div>
          <div><b>{r.signals.nli_entailment.toFixed(2)}</b>entailment</div>
          <div><b>{r.signals.nli_contradiction.toFixed(2)}</b>contradiction</div>
          <div><b>{r.signals.context_divergence < 0 ? '—' : r.signals.context_divergence.toFixed(2)}</b>context diff.</div>
        </div>
        <p className="muted small" style={{ marginTop: 6 }}>decided by {r.decided_by}{r.signals.polarity_conflict ? ' · opposing stance' : ''}</p>
      </section>
    </div>
  )
}
