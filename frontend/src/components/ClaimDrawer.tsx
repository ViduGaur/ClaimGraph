import { useEffect } from 'react'
import type { AnalysisResult } from '../types'
import { ClaimDetail } from './ClaimDetail'

/** Slide-over used outside the graph workspace (claims list, disagreements, landscape). */
export function ClaimDrawer({ result, claimId, colors, onClose, onSelect }: {
  result: AnalysisResult
  claimId: string
  colors: Record<string, string>
  onClose: () => void
  onSelect: (id: string) => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label="Claim provenance">
        <div className="panel-head">
          <span>Why did ClaimGraph say this?</span>
          <button className="link-btn" onClick={onClose}>✕ Close <span className="mono muted">esc</span></button>
        </div>
        <div className="panel-body">
          <ClaimDetail result={result} claimId={claimId} colors={colors} onSelectClaim={onSelect} />
        </div>
      </aside>
    </>
  )
}
