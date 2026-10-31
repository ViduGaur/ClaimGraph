// Mirrors backend/app/schemas/domain.py and api.py
export type ClaimType = 'finding' | 'comparison' | 'method' | 'limitation' | 'future_work' | 'background'
export type Polarity = 'positive' | 'negative' | 'neutral' | 'none'
export type RelationType = 'SUPPORTS' | 'CONTRADICTS' | 'QUALIFIES' | 'EXTENDS'
export type Verdict = 'context_dependent' | 'genuine_disagreement' | 'insufficient_evidence'

export interface Paper {
  paper_id: string
  title: string
  authors: string[]
  year: number | null
  venue: string | null
  doi: string | null
  url: string | null
  pdf_url: string | null
  abstract: string | null
  is_open_access: boolean
  oa_status: string | null
  citation_count: number | null
  sources: string[]
  source_ids: Record<string, string>
  relevance: number
  text_source: 'full_text' | 'abstract' | 'none'
  acquisition_note: string | null
}

export interface Passage {
  passage_id: string
  paper_id: string
  section: string
  text: string
  relevance: number
}

export interface ClaimContext {
  datasets: string[]
  models: string[]
  metrics: string[]
  population: string[]
  sample_size: string | null
  conditions: string[]
  training: string[]
  baselines: string[]
  task: string[]
  data_scale: string | null
  time_period: string | null
  methodology: string | null
}

export interface Claim {
  claim_id: string
  paper_id: string
  claim_text: string
  normalized_text: string
  claim_type: ClaimType
  polarity: Polarity
  subject: string | null
  evidence: Passage[]
  section: string
  context: ClaimContext
  inherited_context: string[]
  reported_result: string | null
  confidence: number
  extraction_method: 'heuristic' | 'llm_selected' | 'llm'
  grounding_score: number
  source_citation: string
  relevance: number
  uncertainty_notes: string[]
}

export interface FactorComparison {
  factor: string
  a: string[]
  b: string[]
  status: 'same' | 'overlap' | 'different' | 'unknown'
  inherited: boolean
  a_quote?: string | null
  b_quote?: string | null
  note?: string | null
}

export interface Relationship {
  relation_id: string
  source_claim_id: string
  target_claim_id: string
  relation: RelationType
  confidence: number
  reasoning: string
  signals: {
    similarity: number
    nli_entailment: number
    nli_contradiction: number
    nli_neutral: number
    polarity_conflict: boolean
    context_divergence: number
    conflict_signal?: boolean
  }
  factors: FactorComparison[]
  decided_by: 'rules' | 'llm' | 'rules+llm'
  llm_note?: string | null
  uncertain: boolean
}

export interface ConflictAnalysis {
  conflict_id: string
  relation_id: string
  claim_a_id: string
  claim_b_id: string
  verdict: Verdict
  headline: string
  explanation: string
  differing_factors: FactorComparison[]
  shared_factors: FactorComparison[]
  unknown_factors: string[]
  evidence_passage_ids: string[]
  confidence: number
  llm_explanation: string | null
  apparent_conflict?: string | null
  explanation_source?: 'template' | 'llm'
  explanation_note?: string | null
}

export interface Theme {
  theme_id: string
  label: string
  keywords: string[]
  claim_ids: string[]
  paper_ids: string[]
  support_edges: number
  conflict_edges: number
  status: 'consensus' | 'disputed' | 'emerging' | 'single_source'
  summary: string
}

export interface ResearchGap {
  gap_id: string
  title: string
  signal: string
  rationale: string
  evidence_claim_ids: string[]
  strength: 'weak' | 'moderate' | 'strong'
}

export interface Landscape {
  summary: string
  consensus: Theme[]
  disagreements: Theme[]
  emerging: Theme[]
  unresolved_questions: string[]
  gaps: ResearchGap[]
}

export interface GraphNode {
  id: string
  kind: 'paper' | 'claim'
  label: string
  data: Record<string, unknown>
}
export interface GraphEdge {
  id: string
  source: string
  target: string
  kind: 'MAKES' | RelationType
  data: Record<string, unknown>
}

export interface AnalysisResult {
  query_plan: { question: string; search_queries: string[]; key_terms: string[]; entities: string[]; method: string }
  papers: Paper[]
  claims: Claim[]
  relationships: Relationship[]
  conflicts: ConflictAnalysis[]
  graph: { nodes: GraphNode[]; edges: GraphEdge[]; stats: Record<string, unknown> }
  landscape: Landscape
  run: {
    demo_mode: boolean
    llm_provider: string
    llm_model: string | null
    llm_calls: number
    embedding_backend: string
    nli_backend: string
    sources: string[]
    duration_s: number | null
  }
}

export type StageStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped'
export interface StageProgress {
  key: string
  label: string
  status: StageStatus
  detail: string | null
  started_at: string | null
  finished_at: string | null
}

export interface AnalysisSummary {
  id: string
  question: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  stages: StageProgress[]
  warnings: { stage: string; message: string }[]
  error: string | null
  counts: { papers: number; claims: number; relationships: number; conflicts: number; gaps: number }
  demo_mode: boolean
}

export interface AnalysisDetail extends AnalysisSummary {
  result: AnalysisResult | null
}

export interface Health {
  status: 'ok'
  version: string
  demo_mode: boolean
  llm_provider: string
  llm_available: boolean
  sources: string[]
  keys_configured: Record<string, boolean>
}
