import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import type { AnalysisSummary, Health } from '../types'

const EXAMPLES = [
  'Do vision transformers need large-scale pre-training to outperform convolutional networks?',
  'Does chain-of-thought prompting improve reasoning in small language models?',
  'Is data augmentation effective for low-resource speech recognition?',
  'Does intermittent fasting improve insulin sensitivity compared to calorie restriction?',
]
const DEMO_QUESTION = EXAMPLES[0]

const STEPS = ['Query understanding', 'Literature discovery', 'Open-access acquisition', 'Claim extraction', 'Claim comparison & NLI', 'Conflict analysis', 'Claim graph & landscape']

const STEP_COLORS = ['var(--accent)', 'var(--accent)', 'var(--accent)', 'var(--rel-extends)', 'var(--rel-supports)', 'var(--rel-contradicts)', 'var(--rel-qualifies)']

/** Decorative: three papers, their claims, and the four relation types. */
function HeroArt() {
  return (
    <svg className="hero-art" viewBox="0 0 220 170" aria-hidden="true">
      <g fill="none" strokeWidth="2.2" strokeLinecap="round">
        <path d="M58 58 C 100 40, 130 44, 160 62" stroke="var(--rel-supports)" />
        <path d="M60 70 C 90 100, 100 110, 106 122" stroke="var(--rel-contradicts)" strokeWidth="3" />
        <path d="M164 74 C 150 100, 130 112, 116 124" stroke="var(--rel-qualifies)" strokeDasharray="6 4" />
        <path d="M52 66 C 40 100, 70 130, 100 130" stroke="var(--rel-extends)" strokeDasharray="1 5" />
      </g>
      <g stroke="var(--border-strong)" strokeWidth="1">
        <line x1="30" y1="30" x2="52" y2="60" /><line x1="190" y1="30" x2="166" y2="66" /><line x1="110" y1="160" x2="110" y2="130" />
      </g>
      <rect x="6" y="16" width="52" height="20" rx="5" fill="#eda100" fillOpacity="0.18" stroke="#eda100" />
      <rect x="162" y="16" width="52" height="20" rx="5" fill="#e87ba4" fillOpacity="0.18" stroke="#e87ba4" />
      <rect x="84" y="146" width="52" height="20" rx="5" fill="#008300" fillOpacity="0.14" stroke="#008300" />
      <circle cx="54" cy="64" r="8" fill="#eda100" stroke="var(--surface)" strokeWidth="2" />
      <circle cx="164" cy="68" r="8" fill="#e87ba4" stroke="var(--surface)" strokeWidth="2" />
      <circle cx="110" cy="128" r="8" fill="#008300" stroke="var(--surface)" strokeWidth="2" />
    </svg>
  )
}

export function Home({ health }: { health: Health | null }) {
  const nav = useNavigate()
  const [question, setQuestion] = useState('')
  const [maxPapers, setMaxPapers] = useState(8)
  const [useLlm, setUseLlm] = useState(false)
  const [demo, setDemo] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [recent, setRecent] = useState<AnalysisSummary[]>([])

  useEffect(() => { api.list().then(setRecent).catch(() => setRecent([])) }, [])
  useEffect(() => { if (health?.demo_mode) setDemo(true) }, [health])

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault()
    const q = question.trim() || (demo ? DEMO_QUESTION : '')
    if (q.length < 8) { setError('Please enter a research question (at least a few words).'); return }
    setBusy(true); setError(null)
    try {
      const s = await api.analyze({ question: q, max_papers: maxPapers, use_llm: useLlm, demo })
      nav(`/analysis/${s.id}`)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
      setBusy(false)
    }
  }

  return (
    <div className="page narrow">
      <section className="hero">
        <div>
          <div className="eyebrow">Cross-paper scientific claim analysis</div>
          <h1 style={{ marginTop: 10 }}>What does the literature actually agree on — and where, and why, does it disagree?</h1>
          <p className="lede">
            ClaimGraph finds open-access papers for a research question, extracts their claims with evidence, compares claims
            across papers, and explains apparent contradictions by the datasets, conditions and methods behind them.
          </p>
        </div>
        <HeroArt />
      </section>

      <form className="card query-box" onSubmit={submit}>
        <label htmlFor="q" className="eyebrow">Research question</label>
        <textarea id="q" placeholder={demo ? DEMO_QUESTION : 'What are the current findings on …?'} value={question}
                  onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit() }} />
        <div className="query-row">
          <label className="check">Papers
            <select value={maxPapers} onChange={(e) => setMaxPapers(Number(e.target.value))}>
              {[4, 6, 8, 10, 12].map((n) => <option key={n}>{n}</option>)}
            </select>
          </label>
          <label className="check" title={health?.llm_available ? '' : `LLM provider "${health?.llm_provider}" is not reachable`}>
            <input type="checkbox" checked={useLlm} disabled={!health?.llm_available} onChange={(e) => setUseLlm(e.target.checked)} />
            Use LLM adjudication {health && !health.llm_available && <span className="muted">(not configured)</span>}
          </label>
          <label className="check" title="Analyse the bundled snapshot of 10 real arXiv abstracts instead of searching live">
            <input type="checkbox" checked={demo} onChange={(e) => setDemo(e.target.checked)} />
            Demo corpus
          </label>
          <span className="spacer" />
          <button className="btn" type="submit" disabled={busy}>{busy ? 'Starting…' : 'Analyze literature'}</button>
        </div>
        {error && <div className="notice bad" style={{ marginTop: 12 }}>{error}</div>}
      </form>

      <div className="examples">
        {EXAMPLES.map((ex) => <button key={ex} type="button" onClick={() => setQuestion(ex)}>{ex}</button>)}
      </div>
      {demo && <p className="muted small" style={{ marginTop: 10 }}>Demo corpus is a fixed set of papers on vision transformers vs CNNs — use the first example question with it.</p>}

      <div className="pipeline-strip" aria-label="pipeline">
        {STEPS.map((s, i) => (
          <div key={s} style={{ '--c': STEP_COLORS[i] } as React.CSSProperties}><b>{String(i + 1).padStart(2, '0')}</b>{s}</div>
        ))}
      </div>

      {recent.length > 0 && (
        <>
          <div className="section-title"><h2>Recent analyses</h2></div>
          <div className="card card-pad">
            <ul className="recent">
              {recent.map((r) => (
                <li key={r.id}>
                  <a href={`/analysis/${r.id}`} onClick={(e) => { e.preventDefault(); nav(`/analysis/${r.id}`) }}>{r.question}</a>
                  <span className="muted small mono" style={{ whiteSpace: 'nowrap' }}>
                    {r.status === 'completed' ? `${r.counts.papers}p · ${r.counts.claims}c · ${r.counts.relationships}r` : r.status}
                    {r.demo_mode ? ' · demo' : ''}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  )
}
