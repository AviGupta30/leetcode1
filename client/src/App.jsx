import { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import './index.css'

const API = '/api'
const PER_PAGE = 50

// ── Utilities ────────────────────────────────────────────────────────────────

function formatDuration(secs) {
  if (!secs || secs <= 0) return '--:--'
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  const s = secs % 60
  if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
  return `${m}:${String(s).padStart(2,'0')}`
}

function formatChange(val) {
  if (val === 0) return '0.00'
  return val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2)
}

// Deterministic avatar colour from username
const AVATAR_COLORS = [
  '#3b82f6','#8b5cf6','#ec4899','#14b8a6','#f59e0b',
  '#22c55e','#ef4444','#6366f1','#0ea5e9','#84cc16',
]
function avatarColor(name) {
  let h = 0
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) & 0xffffffff
  return AVATAR_COLORS[Math.abs(h) % AVATAR_COLORS.length]
}

// ── Sub-components ────────────────────────────────────────────────────────────

function RankCell({ rank }) {
  const r = Math.round(rank)
  if (r === 1) return <div className="rank-cell"><span className="medal">🥇</span><span className="rank-number">1</span></div>
  if (r === 2) return <div className="rank-cell"><span className="medal">🥈</span><span className="rank-number">2</span></div>
  if (r === 3) return <div className="rank-cell"><span className="medal">🥉</span><span className="rank-number">3</span></div>
  return <div className="rank-cell"><span className="rank-number">#{r}</span></div>
}

function UserCell({ username, isNew }) {
  const initial = username[0]?.toUpperCase() || '?'
  return (
    <div className="user-cell">
      <div className="avatar" style={{ background: avatarColor(username) }}>{initial}</div>
      <div className="username-wrap">
        <a
          className="username"
          href={`https://leetcode.com/u/${username}/`}
          target="_blank"
          rel="noreferrer"
        >
          {username}
        </a>
        {isNew && <span className="badge-new">New</span>}
      </div>
    </div>
  )
}

function ChangeCell({ val }) {
  const cls = val > 0 ? 'positive' : val < 0 ? 'negative' : 'zero'
  return <td className={`change-cell ${cls}`}>{formatChange(val)}</td>
}

function TrendIcon({ val }) {
  if (val > 0) return <span className="trend up">↑</span>
  if (val < 0) return <span className="trend down">↓</span>
  return <span className="trend flat">—</span>
}

function LeaderboardRow({ row, highlight }) {
  return (
    <tr className={highlight ? 'highlight-user' : ''}>
      <td><RankCell rank={row.actual_rank} /></td>
      <td><UserCell username={row.username} isNew={row.is_new} /></td>
      <td className="num"><span className="score-val">{row.score}</span></td>
      <td className="num col-duration"><span className="duration-val">{formatDuration(row.finish_time_seconds)}</span></td>
      <td className="num col-expected"><span className="exp-rating">{row.expected_rating?.toFixed(2)}</span></td>
      <td className="num">
        <span className={`old-rating${row.is_new ? ' default' : ''}`}>{row.old_rating?.toFixed(2)}</span>
      </td>
      <ChangeCell val={row.change} />
      <td className="num">
        <span className={`new-rating ${row.change > 0 ? 'up' : row.change < 0 ? 'down' : ''}`}>
          {row.new_rating?.toFixed(2)}
        </span>
      </td>
      <td className="col-trend"><TrendIcon val={row.change} /></td>
    </tr>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [contests, setContests]         = useState([])
  const [selectedSlug, setSelectedSlug] = useState('')
  const [search, setSearch]             = useState('')
  const [searchInput, setSearchInput]   = useState('')
  const [page, setPage]                 = useState(1)

  const [rows, setRows]         = useState([])
  const [total, setTotal]       = useState(0)
  const [totalPages, setTotalPages] = useState(1)

  const [loading, setLoading]   = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)
  const [error, setError]       = useState(null)

  const stepTimerRef = useRef(null)
  const LOAD_STEPS = [
    "Downloading snapshot of 300k+ user histories from GitHub…",
    "Scraping " + "~1,000 leaderboard pages asynchronously (Semaphore 50)…",
    "Merging live ranks with historical ratings…",
    "Pre-computing rank table and running math engine on all users…",
  ]

  // ── Fetch contest list on mount ────────────────────────────────────────────
  useEffect(() => {
    axios.get(`${API}/contests`).then(res => {
      const list = res.data || []
      setContests(list)
      if (list.length > 0) setSelectedSlug(list[0].slug)
    }).catch(() => {
      // Fallback: show an empty dropdown
      setContests([])
    })
  }, [])

  // ── Cycle through loading steps while loading ──────────────────────────────
  useEffect(() => {
    if (loading) {
      stepTimerRef.current = setInterval(() => {
        setLoadingStep(p => (p + 1) % LOAD_STEPS.length)
      }, 10000)
    } else {
      clearInterval(stepTimerRef.current)
      setLoadingStep(0)
    }
    return () => clearInterval(stepTimerRef.current)
  }, [loading])

  // ── Fetch leaderboard page ─────────────────────────────────────────────────
  const fetchLeaderboard = useCallback(async (slug, pg, srch) => {
    if (!slug) return
    setLoading(true)
    setError(null)
    try {
      const resp = await axios.get(`${API}/leaderboard/${slug}`, {
        params: { page: pg, per_page: PER_PAGE, search: srch || undefined },
        timeout: 600000,  // 10 min
      })
      setRows(resp.data.rows || [])
      setTotal(resp.data.total || 0)
      setTotalPages(resp.data.total_pages || 1)
      if (resp.data.page) setPage(resp.data.page)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Unknown error'
      setError(msg)
      setRows([])
    } finally {
      setLoading(false)
    }
  }, [])

  // ── Contest selector change ────────────────────────────────────────────────
  const handleContestChange = (e) => {
    const slug = e.target.value
    setSelectedSlug(slug)
    setRows([])
    setTotal(0)
    setPage(1)
    setSearch('')
    setSearchInput('')
    setError(null)
    if (slug) fetchLeaderboard(slug, 1, '')
  }

  // ── Search submit ──────────────────────────────────────────────────────────
  const handleSearch = (e) => {
    e.preventDefault()
    setSearch(searchInput)
    setPage(1)
    fetchLeaderboard(selectedSlug, 1, searchInput)
  }

  // ── Pagination ─────────────────────────────────────────────────────────────
  const goToPage = (p) => {
    setPage(p)
    fetchLeaderboard(selectedSlug, p, search)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // ── Page numbers to show ───────────────────────────────────────────────────
  const pageNumbers = () => {
    const pages = []
    const start = Math.max(1, page - 2)
    const end   = Math.min(totalPages, page + 2)
    if (start > 1) pages.push(1, '…')
    for (let i = start; i <= end; i++) pages.push(i)
    if (end < totalPages) pages.push('…', totalPages)
    return pages
  }

  const hasData = rows.length > 0

  // ── UI ─────────────────────────────────────────────────────────────────────
  return (
    <>
      <header className="header">
        <div className="header-logo">⚡ LeetCode Rating Predictor</div>
        <div className="header-subtitle">
          Live predictions powered by async REST scraping + O(N·U) frequency map engine
        </div>
      </header>

      <main className="main">

        {/* ── Controls ── */}
        <div className="controls">
          <div className="control-group">
            <span className="control-label">Contest</span>
            <select
              className="select"
              value={selectedSlug}
              onChange={handleContestChange}
              disabled={loading}
              id="contest-select"
            >
              <option value="">— Select a Contest —</option>
              {contests.map(c => (
                <option key={c.slug} value={c.slug}>{c.title}</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <span className="control-label">Search Username</span>
            <form onSubmit={handleSearch} style={{ display: 'flex', gap: '8px' }}>
              <input
                className="input"
                placeholder="Filter by username…"
                value={searchInput}
                onChange={e => setSearchInput(e.target.value)}
                disabled={loading || !selectedSlug}
                id="search-input"
              />
              <button
                className="btn"
                type="submit"
                disabled={loading || !selectedSlug}
                id="search-btn"
              >
                Search
              </button>
            </form>
          </div>
        </div>

        {/* ── Info Banner ── */}
        {hasData && !loading && (
          <div className="banner">
            <span className="icon">ℹ️</span>
            Rankings may update until LeetCode finalises ratings. More recent predictions are more accurate.
          </div>
        )}

        {/* ── Error ── */}
        {error && !loading && (
          <div className="error-box">
            <strong>Error: {error}</strong>
            <p>Check the contest slug is correct and the leaderboard is available.</p>
          </div>
        )}

        {/* ── Loading ── */}
        {loading && (
          <div className="loading-wrap">
            <div className="spinner" />
            <div className="loading-title">Building Predictions…</div>
            <div className="loading-step">{LOAD_STEPS[loadingStep]}</div>
            <div className="loading-note">
              The first request for this contest scrapes ~25,000 users concurrently from the LeetCode REST API (50 parallel requests, ~45–60 s).
              All subsequent lookups are instant from cache.
            </div>
          </div>
        )}

        {/* ── Stats Row ── */}
        {hasData && !loading && (
          <div className="stats">
            <div className="stat-badge">
              <strong>{total.toLocaleString()}</strong> participants
              {search && ` matching "${search}"`}
            </div>
            <div className="stat-badge">Page <strong>{page}</strong> of <strong>{totalPages}</strong></div>
            <div className="stat-grow" />
          </div>
        )}

        {/* ── Empty (no data, no search) ── */}
        {!hasData && !loading && !error && selectedSlug && !search && (
          <div className="empty-state">
            <div className="icon">🏆</div>
            <h3>Ready to predict</h3>
            <p>Select a contest above and the leaderboard will load automatically.</p>
          </div>
        )}

        {/* ── Empty Search Results ── */}
        {!hasData && !loading && !error && selectedSlug && search && (
          <div className="empty-state">
            <div className="icon">🔍</div>
            <h3>No participants found</h3>
            <p>No one matching "{search}" was found in this contest. Ghost users (score=0) are excluded from rating algorithms.</p>
          </div>
        )}

        {/* ── Leaderboard Table ── */}
        {hasData && !loading && (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>User</th>
                    <th className="num">Score</th>
                    <th className="num col-duration">Duration</th>
                    <th className="num col-expected">Expected Rating</th>
                    <th className="num">Old Rating</th>
                    <th className="num">Change</th>
                    <th className="num">New Rating</th>
                    <th className="col-trend"></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => (
                    <LeaderboardRow
                      key={row.username + i}
                      row={row}
                      highlight={search && row.username.toLowerCase() === search.toLowerCase()}
                    />
                  ))}
                </tbody>
              </table>

              {/* ── Pagination ── */}
              {totalPages > 1 && (
                <div className="pagination">
                  <button
                    className="page-btn"
                    onClick={() => goToPage(page - 1)}
                    disabled={page <= 1}
                  >← Prev</button>

                  {pageNumbers().map((p, i) =>
                    p === '…'
                      ? <span key={`e${i}`} className="page-info">…</span>
                      : <button
                          key={p}
                          className={`page-btn ${p === page ? 'active' : ''}`}
                          onClick={() => goToPage(p)}
                        >{p}</button>
                  )}

                  <button
                    className="page-btn"
                    onClick={() => goToPage(page + 1)}
                    disabled={page >= totalPages}
                  >Next →</button>
                </div>
              )}
            </div>
          </>
        )}
      </main>
    </>
  )
}
