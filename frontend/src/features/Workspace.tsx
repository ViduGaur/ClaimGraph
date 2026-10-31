import { useMemo } from 'react'
import { ClaimDetail } from '../components/ClaimDetail'
import { RelationDetail } from '../components/RelationDetail'
import { REL_META, RelationGlyph, VerdictChip } from '../components/ui'
import { ClaimGraphView, RELS } from '../graph/ClaimGraphView'
import type { AnalysisResult } from '../types'

export interface Selection {
  claim: string | null
  relation: string | null
  paper: string | null
}

function Glance({ result, colors, onRelation, onTab }: {
  result: AnalysisResult
  colors: Record<string, string>
  onRelation: (id: string) => void
  onTab: (t: string) => void
}) {
  const claims = Object.fromEntries(result.claims.map((c) => [c.claim_id, c]))
  const total = result.relationships.length || 1
  const consensus = result.landscape.consensus
  return (
    <div className="detail">
      <section>
        <h4>Relationships</h4>
        <div className="relbar" role="img" aria-label="relationship mix">
          {RELS.map((r) => {
            const n = result.relationships.filter((x) => x.relation === r).length
            return n ? <span key={r} title={`${r}: ${n}`} style={{ width: `${(n / total) * 100}%`, background: REL_META[r].color }} /> : null
          })}
        </div>
        <div className="mix-legend">
          {RELS.map((r) => (
            <span key={r}><RelationGlyph relation={r} /> {REL_META[r].label} <b>{result.relationships.filter((x) => x.relation === r).length}</b></span>
          ))}
        </div>
      </section>

      <section>
        <h4>
          Disagreements <span className="count-badge">{result.conflicts.length}</span>
          <button className="link-btn" onClick={() => onTab('disagreements')}>All →</button>
        </h4>
        {result.conflicts.length === 0 && <p className="muted small">No apparent disagreements.</p>}
        <div className="rel-list">
          {result.conflicts.slice(0, 4).map((c) => (
            <button key={c.conflict_id} className="rel-item" onClick={() => onRelation(c.relation_id)}>
              <span className="rel-item-head">
                <span className="paper-line small">
                  <span className="swatch sm" style={{ background: colors[claims[c.claim_a_id].paper_id] }} />
                  {claims[c.claim_a_id].source_citation}
                  <span className="muted">vs</span>
                  <span className="swatch sm" style={{ background: colors[claims[c.claim_b_id].paper_id] }} />
                  {claims[c.claim_b_id].source_citation}
                </span>
              </span>
              <VerdictChip verdict={c.verdict} />
              <span className="rel-item-text">{c.headline}</span>
            </button>
          ))}
        </div>
      </section>

      {consensus.length > 0 && (
        <section>
          <h4>Consensus <button className="link-btn" onClick={() => onTab('landscape')}>Landscape →</button></h4>
          {consensus.slice(0, 2).map((t) => (
            <div key={t.theme_id} className="mini-theme ok">
              <b>{t.label}</b>
              <span className="small secondary">{t.paper_ids.length} papers · {t.support_edges} supporting links</span>
            </div>
          ))}
        </section>
      )}

      {result.landscape.gaps.length > 0 && (
        <section>
          <h4>Potential gaps <span className="count-badge">{result.landscape.gaps.length}</span>
            <button className="link-btn" onClick={() => onTab('landscape')}>All →</button></h4>
          <div className="mini-theme gap"><span className="small">{result.landscape.gaps[0].title.replace('Potential research gap: ', '')}</span></div>
        </section>
      )}

      <p className="hint">Click any claim or edge in the graph to see its evidence and reasoning here.</p>
    </div>
  )
}

export function Workspace({ result, colors, selection, setSelection, onTab }: {
  result: AnalysisResult
  colors: Record<string, string>
  selection: Selection
  setSelection: (s: Partial<Selection>) => void
  onTab: (t: string) => void
}) {
  const claimCount = useMemo(() => {
    const m: Record<string, number> = {}
    for (const c of result.claims) m[c.paper_id] = (m[c.paper_id] ?? 0) + 1
    return m
  }, [result])
  const papers = [...result.papers].sort((a, b) => (a.year ?? 0) - (b.year ?? 0))
  const selectClaim = (id: string) => setSelection({ claim: id, relation: null })
  const selectRelation = (id: string) => setSelection({ relation: id, claim: null })
  const clear = () => setSelection({ claim: null, relation: null, paper: null })
  const title = selection.claim ? 'Claim' : selection.relation ? 'Relationship' : 'At a glance'

  return (
    <div className="workspace">
      <aside className="rail">
        <div className="rail-head">Papers <span className="count-badge">{papers.length}</span></div>
        {papers.map((p) => (
          <button key={p.paper_id} className={`paper-chip ${selection.paper === p.paper_id ? 'active' : ''}`}
                  onClick={() => setSelection({ paper: selection.paper === p.paper_id ? null : p.paper_id, claim: null, relation: null })}
                  title={p.title}>
            <span className="swatch" style={{ background: colors[p.paper_id] }} />
            <span className="paper-chip-body">
              <b>{p.authors[0]?.split(' ').pop() ?? 'Unknown'}{p.authors.length > 1 ? ' et al.' : ''} <span className="muted">{p.year}</span></b>
              <span className="paper-chip-title">{p.title}</span>
            </span>
            <span className="paper-chip-count">{claimCount[p.paper_id] ?? 0}</span>
          </button>
        ))}
        {selection.paper && <button className="link-btn" style={{ marginTop: 8 }} onClick={() => setSelection({ paper: null })}>Show all papers</button>}
      </aside>

      <ClaimGraphView result={result} colors={colors} selection={selection}
                      onSelectClaim={selectClaim} onSelectRelation={selectRelation} onClear={clear} />

      <aside className="panel">
        <div className="panel-head">
          <span>{title}</span>
          {(selection.claim || selection.relation) && <button className="link-btn" onClick={() => setSelection({ claim: null, relation: null })}>✕ Close</button>}
        </div>
        <div className="panel-body">
          {selection.claim
            ? <ClaimDetail result={result} claimId={selection.claim} colors={colors} onSelectClaim={selectClaim} onSelectRelation={selectRelation} />
            : selection.relation
              ? <RelationDetail result={result} relationId={selection.relation} colors={colors} onSelectClaim={selectClaim} />
              : <Glance result={result} colors={colors} onRelation={selectRelation} onTab={onTab} />}
        </div>
      </aside>
    </div>
  )
}
