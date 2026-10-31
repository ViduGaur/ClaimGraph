import { useMemo, useState } from 'react'
import type { AnalysisResult, ClaimType, RelationType } from '../types'
import { claimTypeLabel, Confidence, REL_META, RelationGlyph } from '../components/ui'

const RELS: RelationType[] = ['SUPPORTS', 'CONTRADICTS', 'QUALIFIES', 'EXTENDS']

const TYPES: ClaimType[] = ['finding', 'comparison', 'limitation', 'background', 'method', 'future_work']

export function Claims({ result, colors, selected, onSelect }: {
  result: AnalysisResult
  colors: Record<string, string>
  selected: string | null
  onSelect: (id: string) => void
}) {
  const [type, setType] = useState<ClaimType | 'all'>('all')
  const [paper, setPaper] = useState('all')
  const [q, setQ] = useState('')
  const [onlyLinked, setOnlyLinked] = useState(false)

  const relsOf = useMemo(() => {
    const m: Record<string, typeof result.relationships> = {}
    for (const r of result.relationships) {
      ;(m[r.source_claim_id] ??= []).push(r)
      ;(m[r.target_claim_id] ??= []).push(r)
    }
    return m
  }, [result])

  const shown = result.claims
    .filter((c) => type === 'all' || c.claim_type === type)
    .filter((c) => paper === 'all' || c.paper_id === paper)
    .filter((c) => !onlyLinked || relsOf[c.claim_id]?.length)
    .filter((c) => !q || c.claim_text.toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => (relsOf[b.claim_id]?.length ?? 0) - (relsOf[a.claim_id]?.length ?? 0) || b.confidence - a.confidence)

  const typeCount = (t: ClaimType) => result.claims.filter((c) => c.claim_type === t).length

  return (
    <div className="view">
      <div className="filterbar">
        <input type="text" className="search" placeholder="Search claims…" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="seg">
          <button className={type === 'all' ? 'on' : ''} onClick={() => setType('all')}>All <span>{result.claims.length}</span></button>
          {TYPES.filter(typeCount).map((t) => (
            <button key={t} className={type === t ? 'on' : ''} onClick={() => setType(t)}>{claimTypeLabel(t)} <span>{typeCount(t)}</span></button>
          ))}
        </div>
        <select value={paper} onChange={(e) => setPaper(e.target.value)}>
          <option value="all">All papers</option>
          {result.papers.map((p) => <option key={p.paper_id} value={p.paper_id}>{p.authors[0]?.split(' ').pop()} ({p.year}) — {p.title.slice(0, 50)}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={onlyLinked} onChange={(e) => setOnlyLinked(e.target.checked)} />Linked only</label>
      </div>

      {shown.length === 0 && <div className="empty">No claims match these filters.</div>}
      <div className="claim-grid">
        {shown.map((c) => {
          const rels = relsOf[c.claim_id] ?? []
          return (
            <button key={c.claim_id} className={`claim-card ${selected === c.claim_id ? 'selected' : ''}`}
                    style={{ borderLeftColor: colors[c.paper_id] }} onClick={() => onSelect(c.claim_id)}>
              <span className="claim-card-top">
                <span className="paper-line small"><span className="swatch sm" style={{ background: colors[c.paper_id] }} /><b>{c.source_citation}</b></span>
                <span className={`type-tag t-${c.claim_type}`}>{claimTypeLabel(c.claim_type)}</span>
              </span>
              <span className="claim-text">{c.claim_text}</span>
              <span className="claim-meta">
                <Confidence value={c.confidence} />
                {c.context.datasets.length > 0 && <span className="tag">{c.context.datasets.slice(0, 2).join(', ')}</span>}
                {c.context.data_scale && <span className="tag">{c.context.data_scale} data</span>}
                {RELS.map((r) => {
                  const n = rels.filter((x) => x.relation === r).length
                  return n > 0 && <span key={r} className="links-inline" title={`${n} ${REL_META[r].label.toLowerCase()} link${n > 1 ? 's' : ''}`}><RelationGlyph relation={r} />{n}</span>
                })}
                {c.uncertainty_notes.length > 0 && <span className="caveat" title={c.uncertainty_notes.join('\n')}>⚠ {c.uncertainty_notes.length} caveat{c.uncertainty_notes.length > 1 ? 's' : ''}</span>}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
