import { useEffect, useState } from 'react'
import { Link, Route, Routes } from 'react-router-dom'
import { api } from './api/client'
import { Logo } from './components/ui'
import { AnalysisPage } from './pages/AnalysisPage'
import { Home } from './pages/Home'
import type { Health } from './types'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [down, setDown] = useState(false)
  useEffect(() => { api.health().then(setHealth).catch(() => setDown(true)) }, [])

  return (
    <>
      <header className="topbar">
        <Link to="/" className="wordmark"><Logo />ClaimGraph</Link>
        <nav>
          <Link to="/">New analysis</Link>
          <a href="/api/docs" target="_blank" rel="noreferrer">API docs</a>
        </nav>
        <span className="spacer" />
        {down && <span className="status-dot"><i className="warn" />backend unreachable</span>}
        {health && (
          <span className="status-dot" title={`sources: ${health.sources.join(', ')}`}>
            <i className="ok" />{health.demo_mode ? 'demo mode' : 'live sources'} · LLM {health.llm_available ? health.llm_provider : 'off'}
          </span>
        )}
      </header>
      <Routes>
        <Route path="/" element={<Home health={health} />} />
        <Route path="/analysis/:id" element={<AnalysisPage />} />
      </Routes>
    </>
  )
}
