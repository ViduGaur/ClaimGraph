import type { ClaimType, RelationType, StageStatus, Verdict } from '../types'

export const REL_META: Record<RelationType, { label: string; color: string; dash: string; width: number; help: string }> = {
  SUPPORTS: { label: 'Supports', color: 'var(--rel-supports)', dash: '', width: 2, help: 'compatible finding' },
  CONTRADICTS: { label: 'Contradicts', color: 'var(--rel-contradicts)', dash: '', width: 3, help: 'opposing finding, comparable conditions' },
  QUALIFIES: { label: 'Qualifies', color: 'var(--rel-qualifies)', dash: '6 4', width: 2, help: 'narrows or conditions the other claim' },
  EXTENDS: { label: 'Extends', color: 'var(--rel-extends)', dash: '2 3', width: 2, help: 'consistent, in a new setting' },
}

export function RelationGlyph({ relation, uncertain = false }: { relation: RelationType; uncertain?: boolean }) {
  const m = REL_META[relation]
  return (
    <svg width="22" height="8" viewBox="0 0 22 8" aria-hidden="true" style={{ opacity: uncertain ? 0.55 : 1 }}>
      <line x1="1" y1="4" x2="21" y2="4" stroke={m.color} strokeWidth={m.width} strokeDasharray={m.dash} strokeLinecap="round" />
    </svg>
  )
}

export function RelationLabel({ relation, uncertain = false }: { relation: RelationType; uncertain?: boolean }) {
  return (
    <span className="rel" title={REL_META[relation].help}>
      <RelationGlyph relation={relation} uncertain={uncertain} />
      {relation}
      {uncertain && <span className="muted" style={{ fontWeight: 400, letterSpacing: 0 }}>· uncertain</span>}
    </span>
  )
}

export function Confidence({ value, label }: { value: number; label?: string }) {
  const pct = Math.round(value * 100)
  const word = value >= 0.75 ? 'high' : value >= 0.5 ? 'moderate' : 'low'
  return (
    <span className="confidence" title={`${label ?? 'confidence'}: ${pct}% (${word})`}>
      <span className="bar"><span style={{ width: `${pct}%` }} /></span>
      {pct}%
    </span>
  )
}

const TYPE_LABEL: Record<ClaimType, string> = {
  finding: 'Finding',
  comparison: 'Comparison',
  method: 'Method',
  limitation: 'Limitation',
  future_work: 'Future work',
  background: 'Prior belief',
}
export const claimTypeLabel = (t: ClaimType) => TYPE_LABEL[t]

export function ClaimTypeChip({ type }: { type: ClaimType }) {
  return <span className="chip outline">{TYPE_LABEL[type]}</span>
}

const VERDICT: Record<Verdict, { label: string; cls: string }> = {
  context_dependent: { label: 'Context-dependent', cls: 'warn' },
  genuine_disagreement: { label: 'Genuine disagreement', cls: 'bad' },
  insufficient_evidence: { label: 'Insufficient evidence', cls: '' },
}
export function VerdictChip({ verdict }: { verdict: Verdict }) {
  const v = VERDICT[verdict]
  const icon = verdict === 'genuine_disagreement' ? '≠' : verdict === 'context_dependent' ? '≈' : '?'
  return <span className={`chip ${v.cls}`}><span aria-hidden="true" className="mono">{icon}</span>{v.label}</span>
}

export function StageIcon({ status }: { status: StageStatus }) {
  if (status === 'running')
    return (
      <svg className="stage-icon spin" viewBox="0 0 16 16" aria-label="running">
        <circle cx="8" cy="8" r="6" fill="none" stroke="var(--border-strong)" strokeWidth="2" />
        <path d="M8 2 a6 6 0 0 1 6 6" fill="none" stroke="var(--ink)" strokeWidth="2" strokeLinecap="round" />
      </svg>
    )
  if (status === 'done')
    return (
      <svg className="stage-icon" viewBox="0 0 16 16" aria-label="done">
        <circle cx="8" cy="8" r="7" fill="var(--ok-bg)" />
        <path d="M5 8.2 7.1 10.3 11 6" fill="none" stroke="var(--ok)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  if (status === 'failed')
    return (
      <svg className="stage-icon" viewBox="0 0 16 16" aria-label="failed">
        <circle cx="8" cy="8" r="7" fill="var(--bad-bg)" />
        <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="var(--bad)" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    )
  return (
    <svg className="stage-icon" viewBox="0 0 16 16" aria-label={status}>
      <circle cx="8" cy="8" r="6" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" strokeDasharray={status === 'skipped' ? '2 2' : ''} />
    </svg>
  )
}

/** Renders the evidence passage with the claim sentence highlighted. */
export function EvidenceQuote({ passage, claim }: { passage: string; claim: string }) {
  const i = passage.indexOf(claim)
  if (i < 0) return <blockquote className="quote" style={{ margin: 0 }}>{passage}</blockquote>
  return (
    <blockquote className="quote" style={{ margin: 0 }}>
      {passage.slice(0, i)}
      <mark>{claim}</mark>
      {passage.slice(i + claim.length)}
    </blockquote>
  )
}

export function Logo() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="4" cy="5" r="2.3" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="16" cy="5" r="2.3" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="10" cy="15.5" r="2.3" fill="currentColor" />
      <path d="M6.2 5.6h7.6M5.2 7l3.6 6.5M14.8 7l-3.6 6.5" stroke="currentColor" strokeWidth="1.2" fill="none" />
    </svg>
  )
}

export function doiUrl(doi: string | null) {
  return doi ? `https://doi.org/${doi}` : null
}

export function authorsShort(authors: string[], n = 3) {
  if (!authors.length) return 'Unknown authors'
  return authors.length > n ? `${authors.slice(0, n).join(', ')} et al.` : authors.join(', ')
}

export function paperCitation(p: { authors: string[]; year: number | null }) {
  const last = p.authors[0]?.split(' ').pop() ?? 'Unknown'
  return `${last}${p.authors.length > 1 ? ' et al.' : ''} (${p.year ?? 'n.d.'})`
}
