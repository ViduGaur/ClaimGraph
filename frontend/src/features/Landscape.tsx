import type { AnalysisResult, Theme } from '../types'

function ThemeCard({ t, result, colors, onClaim }: { t: Theme; result: AnalysisResult; colors: Record<string, string>; onClaim: (id: string) => void }) {
  const claims = Object.fromEntries(result.claims.map((c) => [c.claim_id, c]))
  return (
    <div className="theme-card">
      <div className="theme-card-head">
        <b>{t.label}</b>
        <span className="swatches">{t.paper_ids.map((p) => <span key={p} className="swatch sm" style={{ background: colors[p] }} />)}</span>
      </div>
      <p className="small secondary">{t.summary}</p>
      <ul>
        {t.claim_ids.slice(0, 3).map((id) => (
          <li key={id}>
            <button className="text-btn" onClick={() => onClaim(id)}>
              <span className="swatch sm" style={{ background: colors[claims[id].paper_id] }} />
              <span>{claims[id].claim_text} <span className="muted">— {claims[id].source_citation}</span></span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

const STRENGTH: Record<string, string> = { strong: 'bad', moderate: 'warn', weak: 'neutral' }

export function Landscape({ result, colors, onClaim }: { result: AnalysisResult; colors: Record<string, string>; onClaim: (id: string) => void }) {
  const l = result.landscape
  const claims = Object.fromEntries(result.claims.map((c) => [c.claim_id, c]))
  const cols: { key: string; title: string; hint: string; cls: string; themes: Theme[] }[] = [
    { key: 'c', title: 'Consensus', hint: 'several papers agree, no conflicts', cls: 'ok', themes: l.consensus },
    { key: 'd', title: 'Disputed', hint: 'themes containing apparent conflicts', cls: 'warn', themes: l.disagreements },
    { key: 'e', title: 'Emerging', hint: 'not yet replicated in this set', cls: 'info', themes: l.emerging },
  ]
  return (
    <div className="view">
      <div className="summary-card"><p className="summary-text">{l.summary}</p></div>

      <div className="landscape-cols">
        {cols.map((c) => (
          <section key={c.key} className={`landscape-col ${c.cls}`}>
            <h3>{c.title} <span className="count-badge">{c.themes.length}</span></h3>
            <p className="muted small">{c.hint}</p>
            {c.themes.length === 0 && <p className="muted small" style={{ marginTop: 12 }}>None detected.</p>}
            {c.themes.map((t) => <ThemeCard key={t.theme_id} t={t} result={result} colors={colors} onClaim={onClaim} />)}
          </section>
        ))}
      </div>

      <div className="section-title"><h2>Potential research gaps</h2><span className="muted small">signals within the analysed papers — verify before relying on them</span></div>
      <div className="gap-grid">
        {l.gaps.length === 0 && <p className="muted">No gap signals were strong enough to report.</p>}
        {l.gaps.map((g) => (
          <div key={g.gap_id} className={`gap-card ${STRENGTH[g.strength]}`}>
            <div className="gap-card-head">
              <span className="tag">{g.signal.replace(/_/g, ' ')}</span>
              <span className={`strength ${STRENGTH[g.strength]}`}>{g.strength}</span>
            </div>
            <b>{g.title.replace(/^Potential research gap( \(author-stated\))?: /, '')}</b>
            <p className="small secondary">{g.rationale}</p>
            <div className="gap-evidence">
              {g.evidence_claim_ids.slice(0, 4).map((id) => claims[id] && (
                <button key={id} className="text-btn small" onClick={() => onClaim(id)}>
                  <span className="swatch sm" style={{ background: colors[claims[id].paper_id] }} />{claims[id].source_citation}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {l.unresolved_questions.length > 0 && (
        <>
          <div className="section-title"><h2>Unresolved questions</h2></div>
          <ol className="questions">
            {l.unresolved_questions.map((q) => <li key={q}>{q}</li>)}
          </ol>
        </>
      )}
    </div>
  )
}
