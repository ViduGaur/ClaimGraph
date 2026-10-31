import { Fragment, useState } from 'react'
import type { AnalysisResult } from '../types'
import { authorsShort, doiUrl } from '../components/ui'

export function Papers({ result, colors }: { result: AnalysisResult; colors: Record<string, string> }) {
  const [open, setOpen] = useState<string | null>(null)
  const claimCount = (id: string) => result.claims.filter((c) => c.paper_id === id).length
  const maxRel = Math.max(...result.papers.map((p) => p.relevance), 0.0001)
  return (
    <div className="view"><div className="card" style={{ overflowX: 'auto' }}>
      <table className="table">
        <thead>
          <tr>
            <th style={{ width: '42%' }}>Title</th>
            <th>Year</th>
            <th>Venue</th>
            <th>DOI</th>
            <th>Relevance</th>
            <th>Access</th>
            <th>Claims</th>
          </tr>
        </thead>
        <tbody>
          {result.papers.map((p) => {
            const doi = doiUrl(p.doi)
            return (
              <Fragment key={p.paper_id}>
                <tr className="clickable" onClick={() => setOpen(open === p.paper_id ? null : p.paper_id)}>
                  <td>
                    <div className="paper-line" style={{ alignItems: 'flex-start' }}>
                      <span className="swatch" style={{ background: colors[p.paper_id], marginTop: 4 }} />
                      <span style={{ fontWeight: 500 }}>{p.title}</span>
                    </div>
                    <div className="muted small" style={{ marginTop: 2, paddingLeft: 20 }}>{authorsShort(p.authors)}</div>
                  </td>
                  <td className="tnum">{p.year ?? '—'}</td>
                  <td className="small secondary">{p.venue ?? '—'}</td>
                  <td className="small">{doi ? <a href={doi} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()} className="mono">{p.doi}</a> : <span className="muted">—</span>}</td>
                  <td>
                    <span className="confidence"><span className="bar"><span style={{ width: `${(p.relevance / maxRel) * 100}%` }} /></span>{p.relevance.toFixed(2)}</span>
                  </td>
                  <td>
                    <span className={`chip ${p.text_source === 'full_text' ? 'ok' : ''}`} title={p.acquisition_note ?? ''}>
                      {p.text_source === 'full_text' ? 'Full text' : p.text_source === 'abstract' ? 'Abstract' : 'None'}
                    </span>
                    {p.is_open_access && <div className="muted small" style={{ marginTop: 4 }}>OA{p.oa_status ? ` · ${p.oa_status}` : ''}</div>}
                  </td>
                  <td className="mono">{claimCount(p.paper_id)}</td>
                </tr>
                {open === p.paper_id && (
                  <tr>
                    <td colSpan={7} style={{ background: 'var(--surface-2)' }}>
                      <p className="small secondary" style={{ maxWidth: 900 }}>{p.abstract ?? 'No abstract available.'}</p>
                      <p className="small muted" style={{ marginTop: 8 }}>
                        Found via {p.sources.join(', ')} · {p.acquisition_note}
                        {p.citation_count != null && ` · ${p.citation_count} citations`}
                        {p.url && <> · <a href={p.url} target="_blank" rel="noreferrer">source ↗</a></>}
                        {p.pdf_url && <> · <a href={p.pdf_url} target="_blank" rel="noreferrer">PDF ↗</a></>}
                      </p>
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div></div>
  )
}
