import { lazy, Suspense, useCallback, useMemo } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { ClaimDrawer } from '../components/ClaimDrawer'
import { StageIcon } from '../components/ui'
import { Claims } from '../features/Claims'
import { Disagreements } from '../features/Disagreements'
import { Landscape } from '../features/Landscape'
import { Papers } from '../features/Papers'
import type { Selection } from '../features/Workspace'
import { useAnalysis } from '../hooks/useAnalysis'
import { paperColorMap } from '../lib/paperColors'
import type { AnalysisDetail } from '../types'

// Cytoscape is ~400 kB; only load it with the graph workspace
const Workspace = lazy(() => import('../features/Workspace').then((m) => ({ default: m.Workspace })))

const TABS = [
  { key: 'graph', label: 'Claim graph', icon: '◉' },
  { key: 'claims', label: 'Claims', icon: '❝' },
  { key: 'disagreements', label: 'Disagreements', icon: '≠' },
  { key: 'landscape', label: 'Landscape', icon: '▦' },
  { key: 'papers', label: 'Papers', icon: '▤' },
] as const
type Tab = (typeof TABS)[number]['key']

function Progress({ data }: { data: AnalysisDetail }) {
  const done = data.stages.filter((s) => s.status === 'done').length
  return (
    <div className="progress-wrap">
      <div className="progress-bar"><span style={{ width: `${(done / data.stages.length) * 100}%` }} /></div>
      <ol className="stages">
        {data.stages.map((s, i) => (
          <li key={s.key} data-status={s.status}>
            <span className="idx">{i + 1}</span>
            <span className="label">{s.label}</span>
            <span className="detail">{s.detail}</span>
            <StageIcon status={s.status} />
          </li>
        ))}
      </ol>
      {data.status === 'failed' && <div className="callout bad" style={{ marginTop: 16 }}><b>Analysis failed.</b> {data.error}</div>}
      {data.status !== 'failed' && (
        <p className="muted small" style={{ marginTop: 12 }}>
          Runs locally on CPU. Live searches respect API rate limits (arXiv: one request every 3 s), so analysing 8 full-text papers takes about 1–4 minutes.
        </p>
      )}
      {data.warnings.length > 0 && <Warnings data={data} />}
    </div>
  )
}

function Warnings({ data }: { data: AnalysisDetail }) {
  return (
    <details className="notes">
      <summary>{data.warnings.length} pipeline note{data.warnings.length > 1 ? 's' : ''} — degraded sources, papers without full text</summary>
      <ul>
        {data.warnings.map((w, i) => <li key={i}><span className="tag">{w.stage}</span> {w.message}</li>)}
      </ul>
    </details>
  )
}

export function AnalysisPage() {
  const { id } = useParams()
  const { data, error } = useAnalysis(id)
  const [params, setParams] = useSearchParams()
  const tabParam = params.get('tab')
  const tab: Tab = TABS.some((t) => t.key === tabParam) ? (tabParam as Tab) : 'graph'
  const selection: Selection = { claim: params.get('claim'), relation: params.get('rel'), paper: params.get('paper') }
  const result = data?.result ?? null
  const colors = useMemo(() => (result ? paperColorMap(result.papers) : {}), [result])

  const setTab = (t: string) => setParams((p) => {
    p.set('tab', t)
    p.delete('claim'); p.delete('rel')
    return p
  })
  const setSelection = useCallback((s: Partial<Selection>) => setParams((p) => {
    const map: [keyof Selection, string][] = [['claim', 'claim'], ['relation', 'rel'], ['paper', 'paper']]
    for (const [k, q] of map) {
      if (!(k in s)) continue
      const v = s[k]
      if (v) p.set(q, v)
      else p.delete(q)
    }
    return p
  }, { replace: true }), [setParams])
  const openClaim = useCallback((cid: string) => setSelection({ claim: cid, relation: null }), [setSelection])
  const closeClaim = useCallback(() => setSelection({ claim: null }), [setSelection])

  if (error) return <div className="page"><div className="callout bad">{error}</div></div>
  if (!data) return <div className="page"><p className="muted">Loading…</p></div>

  const counts: Record<Tab, number | null> = {
    graph: result?.relationships.length ?? null,
    claims: result?.claims.length ?? null,
    disagreements: result?.conflicts.length ?? null,
    landscape: null,
    papers: result?.papers.length ?? null,
  }

  return (
    <div className={`page analysis ${result && tab === 'graph' ? 'wide' : ''}`}>
      <header className="analysis-head">
        <div className="eyebrow">{data.demo_mode ? 'Analysis · demo corpus' : 'Analysis'}{result && ` · ${result.run.llm_provider === 'none' ? 'local models only' : `LLM: ${result.run.llm_model}`} · ${result.run.duration_s?.toFixed(0)} s`}</div>
        <h1>{data.question}</h1>
      </header>

      {!result && <Progress data={data} />}

      {result && (
        <>
          <nav className="tabbar" role="tablist">
            {TABS.map((t) => (
              <button key={t.key} role="tab" aria-selected={tab === t.key} onClick={() => setTab(t.key)}>
                <span className="tab-icon" aria-hidden="true">{t.icon}</span>{t.label}
                {counts[t.key] !== null && <span className="tab-count">{counts[t.key]}</span>}
              </button>
            ))}
          </nav>

          {tab === 'graph' && (
            <Suspense fallback={<p className="muted" style={{ padding: 40 }}>Loading graph…</p>}>
              <Workspace result={result} colors={colors} selection={selection} setSelection={setSelection} onTab={setTab} />
            </Suspense>
          )}
          {tab === 'claims' && <Claims result={result} colors={colors} selected={selection.claim} onSelect={openClaim} />}
          {tab === 'disagreements' && <Disagreements result={result} colors={colors} onClaim={openClaim} />}
          {tab === 'landscape' && <Landscape result={result} colors={colors} onClaim={openClaim} />}
          {tab === 'papers' && <Papers result={result} colors={colors} />}
          {data.warnings.length > 0 && tab === 'papers' && <div className="view"><Warnings data={data} /></div>}
          {tab !== 'graph' && selection.claim && (
            <ClaimDrawer result={result} claimId={selection.claim} colors={colors} onClose={closeClaim} onSelect={openClaim} />
          )}
        </>
      )}
    </div>
  )
}
