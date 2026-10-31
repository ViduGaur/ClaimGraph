import cytoscape, { type Core, type ElementDefinition } from 'cytoscape'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { AnalysisResult, RelationType } from '../types'
import { REL_META, RelationGlyph } from '../components/ui'

export const RELS: RelationType[] = ['SUPPORTS', 'CONTRADICTS', 'QUALIFIES', 'EXTENDS']

const FORCE_LAYOUT = {
  name: 'cose', animate: false, randomize: true, padding: 50, numIter: 2500, gravity: 0.35, componentSpacing: 80,
  nodeRepulsion: () => 60000, nodeOverlap: 20,
  idealEdgeLength: (e: cytoscape.EdgeSingular) => (e.data('kind') === 'MAKES' ? 55 : 150),
} as cytoscape.LayoutOptions

/** Papers on an outer ring (by year); each paper's claims fan inward, so relationship
 *  edges read Paper → Claim → relation → Claim → Paper across the middle. */
function radialPositions(elements: ElementDefinition[]): Record<string, { x: number; y: number }> {
  const papers = elements.filter((e) => e.data.kind === 'paper').sort((a, b) => (a.data.year as number) - (b.data.year as number))
  const claimsOf: Record<string, string[]> = {}
  for (const e of elements) if (e.data.kind === 'MAKES') (claimsOf[e.data.source as string] ??= []).push(e.data.target as string)
  const n = Math.max(papers.length, 1)
  const R = Math.max(260, n * 62)
  const pos: Record<string, { x: number; y: number }> = {}
  papers.forEach((p, i) => {
    const theta = (2 * Math.PI * i) / n - Math.PI / 2
    const px = R * Math.cos(theta), py = R * Math.sin(theta)
    pos[p.data.id as string] = { x: px, y: py }
    const claims = claimsOf[p.data.id as string] ?? []
    const m = claims.length
    const spread = Math.min(0.55, 2.6 / Math.max(m, 1))
    const d = 95 + 10 * m
    claims.forEach((cid, k) => {
      const phi = theta + Math.PI + spread * (k - (m - 1) / 2)
      pos[cid] = { x: px + d * Math.cos(phi), y: py + d * Math.sin(phi) }
    })
  })
  return pos
}

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export interface GraphSelection {
  claim?: string | null
  relation?: string | null
  paper?: string | null
}

export function ClaimGraphView({ result, colors, selection, onSelectClaim, onSelectRelation, onClear }: {
  result: AnalysisResult
  colors: Record<string, string>
  selection: GraphSelection
  onSelectClaim: (id: string) => void
  onSelectRelation: (id: string) => void
  onClear: () => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const cy = useRef<Core | null>(null)
  const [visible, setVisible] = useState<Set<RelationType>>(new Set(RELS))
  const [showUncertain, setShowUncertain] = useState(true)
  const [hideIsolated, setHideIsolated] = useState(true)
  const [layout, setLayout] = useState<'paper' | 'force'>('paper')
  const [tip, setTip] = useState<{ text: string; x: number; y: number; color?: string; sub?: string } | null>(null)
  // handlers live in a ref so selecting something never rebuilds the graph (and never resets zoom/pan)
  const handlers = useRef({ onSelectClaim, onSelectRelation, onClear })
  handlers.current = { onSelectClaim, onSelectRelation, onClear }
  const counts = useMemo(() => Object.fromEntries(RELS.map((r) => [r, result.relationships.filter((x) => x.relation === r).length])), [result])

  const elements = useMemo<ElementDefinition[]>(() => {
    const rels = result.graph.edges.filter((e) => e.kind !== 'MAKES')
      .filter((e) => visible.has(e.kind as RelationType))
      .filter((e) => showUncertain || !e.data.uncertain)
    const linked = new Set(rels.flatMap((e) => [e.source, e.target]))
    const claims = result.graph.nodes.filter((n) => n.kind === 'claim' && (!hideIsolated || linked.has(n.id)))
    const claimIds = new Set(claims.map((c) => c.id))
    const cites = Object.fromEntries(result.claims.map((c) => [c.claim_id, c.source_citation]))
    const papers = result.graph.nodes.filter((n) => n.kind === 'paper' &&
      result.graph.edges.some((e) => e.kind === 'MAKES' && e.source === n.id && claimIds.has(e.target)))
    return [
      ...papers.map((p) => ({ data: { id: p.id, label: p.label, kind: 'paper', year: (p.data.year as number | null) ?? 0, color: colors[p.id] ?? '#888' } })),
      ...claims.map((c) => ({ data: { id: c.id, full: c.label, cite: cites[c.id], kind: 'claim', ctype: c.data.claim_type as string,
        conf: c.data.confidence as number, color: colors[c.data.paper_id as string] ?? '#888' } })),
      ...result.graph.edges.filter((e) => e.kind === 'MAKES' && claimIds.has(e.target)).map((e) => ({ data: { id: e.id, source: e.source, target: e.target, kind: 'MAKES' } })),
      ...rels.map((e) => ({ data: { id: e.id, source: e.source, target: e.target, kind: e.kind, uncertain: e.data.uncertain ? 1 : 0 } })),
    ]
  }, [result, colors, visible, hideIsolated, showUncertain])

  // build the graph
  useEffect(() => {
    if (!box.current) return
    const ink = cssVar('--ink'), ink2 = cssVar('--ink-2'), surface = cssVar('--surface'), border = cssVar('--border-strong')
    const instance = cytoscape({
      container: box.current,
      elements,
      minZoom: 0.25,
      maxZoom: 3,
      style: [
        { selector: 'node[kind = "paper"]', style: {
          shape: 'round-rectangle', 'background-color': 'data(color)', 'background-opacity': 0.16,
          'border-width': 1.5, 'border-color': 'data(color)',
          label: 'data(label)', 'font-family': 'IBM Plex Sans, sans-serif', 'font-size': 16, 'font-weight': 600, color: ink,
          'text-valign': 'center', 'text-halign': 'center', height: 36,
          width: (n: cytoscape.NodeSingular) => Math.max(90, String(n.data('label')).length * 9 + 26),
        } },
        { selector: 'node[kind = "claim"]', style: {
          shape: 'ellipse', width: 'mapData(conf, 0, 1, 12, 22)', height: 'mapData(conf, 0, 1, 12, 22)',
          'background-color': 'data(color)', 'border-width': 2, 'border-color': surface,
        } },
        { selector: 'node[kind = "claim"]:active, node[kind = "claim"].hover', style: { 'overlay-opacity': 0, 'border-color': ink, 'border-width': 3 } },
        { selector: 'node[ctype = "background"]', style: { 'background-opacity': 0.45, 'border-style': 'dashed', 'border-color': ink2, 'border-width': 1.5 } },
        { selector: 'node[ctype = "limitation"]', style: { shape: 'diamond' } },
        { selector: 'edge', style: { 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.9, opacity: 0.85 } },
        { selector: 'edge[kind = "MAKES"]', style: { 'line-color': border, width: 1, 'target-arrow-shape': 'none', opacity: 0.55 } },
        ...RELS.map((r) => ({
          selector: `edge[kind = "${r}"]`,
          style: {
            'line-color': cssVar(`--rel-${r.toLowerCase()}`), 'target-arrow-color': cssVar(`--rel-${r.toLowerCase()}`),
            width: REL_META[r].width + 0.5,
            'line-style': (r === 'QUALIFIES' ? 'dashed' : r === 'EXTENDS' ? 'dotted' : 'solid') as 'dashed' | 'dotted' | 'solid',
          },
        })),
        { selector: 'edge[uncertain = 1]', style: { opacity: 0.4 } },
        { selector: 'edge.hover', style: { opacity: 1, 'z-index': 15 } },
        { selector: '.faded', style: { opacity: 0.1 } },
        { selector: 'node.sel', style: { 'border-width': 4, 'border-color': ink, 'z-index': 20 } },
        { selector: 'edge.sel', style: { width: 6, opacity: 1, 'z-index': 20 } },
      ],
      layout: layout === 'paper'
        ? ({ name: 'preset', positions: radialPositions(elements), padding: box.current.clientWidth < 500 ? 24 : 70 } as cytoscape.LayoutOptions)
        : FORCE_LAYOUT,
    })
    instance.on('tap', 'node[kind = "claim"]', (evt) => handlers.current.onSelectClaim(evt.target.id()))
    instance.on('tap', 'edge[kind != "MAKES"]', (evt) => handlers.current.onSelectRelation(evt.target.id()))
    instance.on('tap', (evt) => { if (evt.target === instance) handlers.current.onClear() })
    instance.on('mouseover', 'node[kind = "claim"]', (evt) => {
      const p = evt.renderedPosition
      setTip({ text: evt.target.data('full'), sub: evt.target.data('cite'), color: evt.target.data('color'), x: p.x, y: p.y })
    })
    instance.on('mouseover', 'edge[kind != "MAKES"]', (evt) => {
      const p = evt.renderedPosition
      setTip({ text: `${REL_META[evt.target.data('kind') as RelationType].label} — click to see why`, x: p.x, y: p.y })
    })
    instance.on('mouseover', 'node[kind = "claim"], edge[kind != "MAKES"]', (evt) => { evt.target.addClass('hover'); box.current!.style.cursor = 'pointer' })
    instance.on('mouseout', 'node, edge', (evt) => { evt.target.removeClass('hover'); box.current!.style.cursor = '' })
    instance.on('mouseout', () => setTip(null))
    instance.on('viewport', () => setTip(null))
    cy.current = instance
    // the canvas can change size after first paint (lazy chunk, stacked mobile layout, window resize); refit when it does
    let lastW = box.current.clientWidth, lastH = box.current.clientHeight
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (Math.abs(width - lastW) < 2 && Math.abs(height - lastH) < 2) return
      lastW = width; lastH = height
      instance.resize()
      instance.fit(undefined, width < 500 ? 24 : 70)
    })
    ro.observe(box.current)
    return () => { ro.disconnect(); instance.destroy() }
  }, [elements, layout])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') handlers.current.onClear() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // reflect the current selection
  useEffect(() => {
    const c = cy.current
    if (!c) return
    c.elements().removeClass('faded sel')
    let focus: cytoscape.CollectionReturnValue | null = null
    if (selection.claim && c.getElementById(selection.claim).nonempty()) {
      const n = c.getElementById(selection.claim)
      n.addClass('sel')
      focus = n.closedNeighborhood().union(n.closedNeighborhood().nodes('[kind = "claim"]').closedNeighborhood('node[kind = "paper"]'))
    } else if (selection.relation && c.getElementById(selection.relation).nonempty()) {
      const e = c.getElementById(selection.relation)
      e.addClass('sel')
      focus = e.union(e.connectedNodes()).union(e.connectedNodes().closedNeighborhood('node[kind = "paper"], edge[kind = "MAKES"]'))
    } else if (selection.paper && c.getElementById(selection.paper).nonempty()) {
      const p = c.getElementById(selection.paper)
      const own = p.closedNeighborhood()
      focus = own.union(own.nodes('[kind = "claim"]').closedNeighborhood())
    }
    if (focus) {
      c.elements().not(focus).addClass('faded')
      // bring the selection into view if it sits outside the current viewport (e.g. picked from the side panel)
      const ext = c.extent(), bb = focus.boundingBox()
      if (bb.x1 < ext.x1 || bb.x2 > ext.x2 || bb.y1 < ext.y1 || bb.y2 > ext.y2) {
        c.animate({ center: { eles: focus } }, { duration: 250 })
      }
    }
  }, [selection, elements, layout])

  const toggle = (r: RelationType) => {
    const next = new Set(visible)
    if (next.has(r)) next.delete(r)
    else next.add(r)
    setVisible(next)
  }

  return (
    <div className="graph-canvas">
      <div className="graph-toolbar">
        {RELS.map((r) => (
          <button key={r} className={`pill ${visible.has(r) ? 'on' : ''}`} onClick={() => toggle(r)} aria-pressed={visible.has(r)}
                  title={REL_META[r].help}>
            <RelationGlyph relation={r} /> {REL_META[r].label} <span className="pill-count">{counts[r]}</span>
          </button>
        ))}
      </div>
      <div className="graph-toolbar right">
        <button className={`pill ${showUncertain ? 'on' : ''}`} onClick={() => setShowUncertain(!showUncertain)} aria-pressed={showUncertain}>Uncertain</button>
        <button className={`pill ${!hideIsolated ? 'on' : ''}`} onClick={() => setHideIsolated(!hideIsolated)} aria-pressed={!hideIsolated}>Unlinked claims</button>
        <button className="pill" onClick={() => setLayout(layout === 'paper' ? 'force' : 'paper')}>{layout === 'paper' ? 'Force layout' : 'By paper'}</button>
        <button className="pill" onClick={() => cy.current?.fit(undefined, 70)}>Fit</button>
      </div>
      <div ref={box} className="graph-surface" aria-label="Claim graph. The Claims and Disagreements tabs list the same information as text." />
      {tip && (
        <div className="graph-tip" style={{ left: Math.min(tip.x + 14, (box.current?.clientWidth ?? 600) - 330), top: tip.y + 14 }}>
          {tip.sub && <div className="paper-line small"><span className="swatch sm" style={{ background: tip.color }} />{tip.sub}</div>}
          {tip.text}
        </div>
      )}
      <div className="graph-legend">
        <span><i className="dot" /> claim · colour = paper</span>
        <span><i className="dot dashed" /> prior belief</span>
        <span><i className="diamond" /> limitation</span>
        <span>arrow: later → earlier claim</span>
      </div>
    </div>
  )
}
