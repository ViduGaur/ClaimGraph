import type { AnalysisDetail, AnalysisSummary, Health } from '../types'

const BASE = '/api/v1'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(BASE + path, { headers: { 'Content-Type': 'application/json' }, ...init })
  } catch {
    throw new ApiError(0, 'Cannot reach the ClaimGraph backend. Is it running on port 8765?')
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') msg = body.detail
      else if (Array.isArray(body.detail)) msg = body.detail.map((d: { msg: string }) => d.msg).join('; ')
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, msg)
  }
  return res.json() as Promise<T>
}

export interface AnalyzeInput {
  question: string
  max_papers?: number
  use_llm?: boolean
  demo?: boolean
}

export const api = {
  health: () => request<Health>('/health'),
  analyze: (input: AnalyzeInput) => request<AnalysisSummary>('/research/analyze', { method: 'POST', body: JSON.stringify(input) }),
  list: () => request<AnalysisSummary[]>('/research?limit=12'),
  get: (id: string, includeResult = true) => request<AnalysisDetail>(`/research/${id}?include_result=${includeResult}`),
}
