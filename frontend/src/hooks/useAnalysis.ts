import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { AnalysisDetail } from '../types'

/** Polls an analysis until it completes or fails, then fetches the full result once. */
export function useAnalysis(id: string | undefined) {
  const [data, setData] = useState<AnalysisDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => {
    if (!id) return
    let cancelled = false
    setData(null)
    setError(null)

    const tick = async () => {
      try {
        const summary = await api.get(id, false)
        if (cancelled) return
        if (summary.status === 'completed') {
          const full = await api.get(id, true)
          if (!cancelled) setData(full)
          return
        }
        setData(summary)
        if (summary.status !== 'failed') timer.current = window.setTimeout(tick, 1500)
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError ? e.message : String(e))
      }
    }
    tick()
    return () => {
      cancelled = true
      window.clearTimeout(timer.current)
    }
  }, [id])

  return { data, error }
}
