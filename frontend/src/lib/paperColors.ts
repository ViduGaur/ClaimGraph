import type { Paper } from '../types'

/* Paper identity palette: the validated 8-slot categorical set, ordered so the first papers get hues
   that relation edges don't use (yellow, magenta, green, red), and re-validated for CVD in both modes.
   Identity is never colour-only: papers are always labelled by citation next to their colour. */
const LIGHT = ['#eda100', '#e87ba4', '#008300', '#e34948', '#2a78d6', '#eb6834', '#4a3aa7', '#1baf7a']
const DARK = ['#c98500', '#d55181', '#008300', '#e66767', '#3987e5', '#d95926', '#9085e9', '#199e70']

function isDark() {
  const forced = document.documentElement.dataset.theme
  if (forced) return forced === 'dark'
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

/** Stable colour per paper, assigned by publication order (colour follows the paper, never its rank). */
export function paperColorMap(papers: Paper[]): Record<string, string> {
  const pal = isDark() ? DARK : LIGHT
  const ordered = [...papers].sort((a, b) => (a.year ?? 0) - (b.year ?? 0) || a.paper_id.localeCompare(b.paper_id))
  return Object.fromEntries(ordered.map((p, i) => [p.paper_id, i < pal.length ? pal[i] : '#8f8e87']))
}
