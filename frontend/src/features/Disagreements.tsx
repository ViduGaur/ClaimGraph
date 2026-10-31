import { useState } from 'react'
import type { AnalysisResult, Claim, FactorComparison, Verdict } from '../types'
import { Confidence, EvidenceQuote, RelationLabel } from '../components/ui'

const VERDICTS: { key: Verdict; label: string; blurb: string; cls: string; icon: string }[] = [
  { key: 'genuine_disagreement', label: 'Genuine disagreement', blurb: 'comparable conditions, opposing findings', cls: 'bad', icon: '≠' },
  { key: 'context_dependent', label: 'Context-dependent', blurb: 'likely explained by different conditions', cls: 'warn', icon: '≈' },
  { key: 'insufficient_evidence', label: 'Insufficient evidence', blurb: 'too little context reported to judge', cls: 'neutral', icon: '?' },
]
const STATUS_TEXT: Record<FactorComparison['status'], string> = { same: 'same', overlap: 'partly shared', different: 'different', unknown: 'not stated' }

/** The factor pills already list the details; the card shows just the verdict sentence. */
function firstSentence(text: string) {
  const m = text.match(/^.+?[.!?](?=\s+[A-Z(]|$)/)
  return m ? m[0] : text
}

function Side({ claim, color, onClaim, label }: { claim: Claim; color: string; onClaim: (id: string) => void; label: string }) {
  return (
    <button className="versus-side" style={{ borderTopColor: color }} onClick={() => onClaim(claim.claim_id)}>
      <span className="paper-line small"><span className="swatch sm" style={{ background: color }} /><b>{label} · {claim.source_citation}</b></span>
      <span className="versus-claim">{claim.claim_text}</span>
    </button>
  )
}

export function Disagreements({ result, colors, onClaim }: {
  result: AnalysisResult
  colors: Record<string, string>
  onClaim: (id: string) => void
}) {
  const [verdict, setVerdict] = useState<Verdict | 'all'>('all')
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const claims = Object.fromEntries(result.claims.map((c) => [c.claim_id, c]))
  const rels = Object.fromEntries(result.relationships.map((r) => [r.relation_id, r]))
  const list = result.conflicts.filter((c) => verdict === 'all' || c.verdict === verdict)

  if (!result.conflicts.length) return <div className="view"><div className="empty">No apparent disagreements were found among the extracted claims.</div></div>

  return (
    <div className="view">
      <div className="verdict-tiles">
        {VERDICTS.map((v) => {
          const n = result.conflicts.filter((c) => c.verdict === v.key).length
          return (
            <button key={v.key} className={`verdict-tile ${v.cls} ${verdict === v.key ? 'on' : ''}`} disabled={!n}
                    aria-pressed={verdict === v.key} onClick={() => setVerdict(verdict === v.key ? 'all' : v.key)}>
              <span className="verdict-icon">{v.icon}</span>
              <span><b className="verdict-num">{n}</b> {v.label}<span className="small muted block">{v.blurb}</span></span>
            </button>
          )
        })}
      </div>
      <p className="muted small" style={{ margin: '4px 0 16px' }}>
        ClaimGraph only calls a disagreement genuine when both passages state comparable conditions. {verdict !== 'all' && <button className="link-btn" onClick={() => setVerdict('all')}>Show all {result.conflicts.length}</button>}
      </p>

      <div className="conflict-list">
        {list.map((c) => {
          const a = claims[c.claim_a_id], b = claims[c.claim_b_id], r = rels[c.relation_id]
          const v = VERDICTS.find((x) => x.key === c.verdict)!
          const expanded = open[c.conflict_id]
          return (
            <article key={c.conflict_id} className={`conflict-card ${v.cls}`}>
              <header>
                <span className={`verdict-badge ${v.cls}`}>{v.icon} {v.label}</span>
                <h3>{c.headline}</h3>
                <Confidence value={c.confidence} />
              </header>
              <div className="versus">
                <Side claim={a} color={colors[a.paper_id]} onClaim={onClaim} label="A" />
                <div className="versus-mid">vs</div>
                <Side claim={b} color={colors[b.paper_id]} onClaim={onClaim} label="B" />
              </div>
              {c.differing_factors.length > 0 && (
                <div className="factor-strip">
                  {c.differing_factors.map((f) => (
                    <span key={f.factor} className="factor-pill">
                      <b>{f.factor}</b> {f.a.join(', ') || '—'} <span className="muted">vs</span> {f.b.join(', ') || '—'}
                    </span>
                  ))}
                </div>
              )}
              {!c.differing_factors.length && <p className="explanation">{firstSentence(c.explanation)}</p>}
              {c.llm_explanation && <p className="small secondary" style={{ padding: '0 20px 12px' }}>{c.llm_explanation}</p>}
              <footer>
                <RelationLabel relation={r.relation} uncertain={r.uncertain} />
                <span className="muted small">contradiction score {r.signals.nli_contradiction.toFixed(2)}{r.signals.polarity_conflict ? ' · opposing stance' : ''}</span>
                <button className="link-btn" onClick={() => setOpen({ ...open, [c.conflict_id]: !expanded })}>{expanded ? 'Hide details' : 'Evidence & all factors'}</button>
              </footer>
              {expanded && (
                <div className="conflict-details">
                  <p className="explanation" style={{ paddingTop: 0 }}>{c.explanation}</p>
                  <div className="versus">
                    <div><span className="eyebrow">Evidence A</span>{a.evidence[0] ? <EvidenceQuote passage={a.evidence[0].text} claim={a.claim_text} /> : <span className="muted small">no passage stored</span>}</div>
                    <div className="versus-mid" />
                    <div><span className="eyebrow">Evidence B</span>{b.evidence[0] ? <EvidenceQuote passage={b.evidence[0].text} claim={b.claim_text} /> : <span className="muted small">no passage stored</span>}</div>
                  </div>
                  <table className="factor-table">
                    <thead><tr><th>Factor</th><th>Claim A</th><th>Claim B</th><th>Comparison</th></tr></thead>
                    <tbody>
                      {r.factors.map((f) => (
                        <tr key={f.factor}>
                          <td>{f.factor}</td>
                          <td>{f.a.join(', ') || <span className="muted">not stated</span>}</td>
                          <td>{f.b.join(', ') || <span className="muted">not stated</span>}</td>
                          <td className={`status-${f.status}`}>{STATUS_TEXT[f.status]}{f.inherited && <span className="muted"> · paper-level</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </article>
          )
        })}
      </div>
    </div>
  )
}
