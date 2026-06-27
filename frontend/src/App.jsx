import { useState, useRef } from 'react'
import './App.css'

const EXAMPLES = [
  { domain: '🏨 Hotel', text: 'Phòng khách sạn rất sạch sẽ và thoáng mát', target: 'Phòng' },
  { domain: '🏨 Hotel', text: 'Khăn tắm cũ và có vết bẩn, rất mất vệ sinh', target: 'Khăn tắm' },
  { domain: '🍴 Nhà hàng', text: 'Món ăn rất ngon, phục vụ nhanh và nhiệt tình', target: 'Món ăn' },
  { domain: '🍴 Nhà hàng', text: 'Nhân viên phục vụ thái độ kém, không nhiệt tình', target: 'Nhân viên' },
  { domain: '📱 Mobile', text: 'Thiết kế đẹp, cầm rất vừa tay và sang trọng', target: 'Thiết kế' },
  { domain: '📱 Mobile', text: 'Loa ngoài nhỏ, nghe không rõ khi xem video', target: 'Loa ngoài' },
]

const SENTIMENT_COLOR = { POSITIVE: '#22c55e', NEGATIVE: '#ef4444', NEUTRAL: '#94a3b8' }

export default function App() {
  const [text, setText] = useState('')
  const [target, setTarget] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const textareaRef = useRef(null)

  function handleSelect() {
    const el = textareaRef.current
    if (!el) return
    const selected = el.value.slice(el.selectionStart, el.selectionEnd).trim()
    if (selected) setTarget(selected)
  }

  async function handlePredict() {
    if (!text.trim() || !target.trim()) return
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.normalize('NFC'), target: target.normalize('NFC') }),
        signal: AbortSignal.timeout(15000),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Lỗi ${res.status}`)
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function loadExample(ex) {
    setText(ex.text); setTarget(ex.target); setResult(null); setError(null)
  }

  return (
    <div className="app">
      <header>
        <h1>Vietnamese ABSA Demo</h1>
        <p>Aspect-Based Sentiment Analysis · PhoBERT + Multi-task Learning</p>
        <a href="https://github.com/dikey4405/targeted_ABSA" target="_blank" rel="noreferrer">GitHub ↗</a>
      </header>

      <main>
        <section className="input-section">
          <label>Review text <span className="hint">(bôi đen để chọn target)</span></label>
          <textarea ref={textareaRef} value={text} onChange={e => setText(e.target.value)}
            onMouseUp={handleSelect} onKeyUp={handleSelect}
            placeholder="Nhập review tiếng Việt vào đây..." rows={4} />

          <label>Target phrase</label>
          <input type="text" value={target} onChange={e => setTarget(e.target.value)}
            placeholder="Từ/cụm từ cần phân tích..." />

          <button onClick={handlePredict} disabled={!text.trim() || !target.trim() || loading} className="predict-btn">
            {loading ? 'Đang phân tích...' : 'Phân tích →'}
          </button>

          {error && <div className="error">⚠ {error}</div>}
        </section>

        {result && (
          <section className="result-section">
            <div className="result-main">
              <div className="badge aspect">{result.aspect}</div>
              <div className="badge sentiment" style={{ background: SENTIMENT_COLOR[result.sentiment] }}>
                {result.sentiment}
              </div>
              <span className="latency">{result.latency_ms}ms</span>
            </div>
            <div className="probs">
              <div className="prob-col">
                <h4>Top aspects</h4>
                {result.aspect_probs.map(p => (
                  <div key={p.label} className="prob-row">
                    <span className="prob-label">{p.label}</span>
                    <div className="prob-bar-bg"><div className="prob-bar" style={{ width: `${(p.score*100).toFixed(0)}%` }} /></div>
                    <span className="prob-score">{(p.score*100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <div className="prob-col">
                <h4>Sentiment</h4>
                {result.sentiment_probs.map(p => (
                  <div key={p.label} className="prob-row">
                    <span className="prob-label">{p.label}</span>
                    <div className="prob-bar-bg"><div className="prob-bar" style={{ width: `${(p.score*100).toFixed(0)}%`, background: SENTIMENT_COLOR[p.label] }} /></div>
                    <span className="prob-score">{(p.score*100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        <section className="examples">
          <h3>Ví dụ nhanh</h3>
          <div className="example-grid">
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="example-card" onClick={() => loadExample(ex)}>
                <span className="domain">{ex.domain}</span>
                <span className="ex-text">{ex.text}</span>
                <span className="ex-target">→ "{ex.target}"</span>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}
