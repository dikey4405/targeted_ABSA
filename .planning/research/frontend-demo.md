# Frontend Demo Research: Vietnamese Targeted ABSA

**Project:** Targeted ABSA Interactive Demo  
**Researched:** 2026-06-27  
**Mode:** Ecosystem + Feasibility  
**Overall confidence:** HIGH (browser APIs + React patterns are stable; recommendations drawn from production patterns)

---

## Executive Summary

The demo is a single-page app with one core interaction: paste text → highlight a span → get a prediction. Everything else (examples, model selector, result display) serves that loop. The implementation is straightforward because the browser's native textarea selection API (`selectionStart`/`selectionEnd`) handles Vietnamese UTF-16 correctly with zero dependencies, and the FastAPI backend contract is a single `POST /predict` endpoint.

The key architectural decision is **what the user interacts with to pick a target phrase**. Three approaches exist; `textarea` with native selection wins for this demo — it's the simplest, works universally across browsers and IMEs, and requires no libraries. The selection data (`textarea.value.slice(selectionStart, selectionEnd)`) is the target string sent to the API — **no character offsets sent to backend** — which avoids the entire class of byte/code-unit offset bugs with Vietnamese text.

---

## 1. Component Architecture

### Recommended File Structure

```
frontend/
├── index.html
├── vite.config.ts
├── tailwind.config.js       # optional — see §4
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── ReviewInput.tsx      # textarea + selection capture
│   │   ├── TargetBadge.tsx      # pill showing selected target phrase
│   │   ├── PredictButton.tsx    # submit + loading spinner
│   │   ├── ExamplePicker.tsx    # horizontal row of one-click example cards
│   │   ├── ResultCard.tsx       # container for the full prediction result
│   │   ├── AspectLabel.tsx      # colored chip: "SERVICE#GENERAL"
│   │   ├── SentimentBadge.tsx   # traffic-light badge: POSITIVE / NEGATIVE / NEUTRAL
│   │   ├── ProbabilityBars.tsx  # top-N horizontal bars for both heads
│   │   └── ModelSelector.tsx    # dropdown for phobert_base / phobert_large / xlm
│   ├── hooks/
│   │   └── usePredict.ts        # single fetch hook with loading/error state
│   ├── data/
│   │   └── examples.ts          # hard-coded example sentences per domain
│   └── types.ts                 # PredictionResult, ExampleEntry, ModelVariant
```

### Component Responsibilities

| Component | Owns | Does NOT own |
|-----------|------|-------------|
| `ReviewInput` | textarea value, selectionStart/End | prediction state |
| `TargetBadge` | renders target string | selection logic |
| `ExamplePicker` | hardcoded example list | API calls |
| `usePredict` | fetch, loading, error | UI rendering |
| `ResultCard` | result layout | data fetching |
| `ProbabilityBars` | bar widths via CSS | sorting logic |

### State Shape (App.tsx)

```typescript
// All state lives in App.tsx — no context provider needed
const [reviewText, setReviewText] = useState('');
const [targetPhrase, setTargetPhrase] = useState('');
const [selectedModel, setSelectedModel] = useState<ModelVariant>('phobert_large');
const { predict, result, isLoading, error } = usePredict();
```

`targetPhrase` is set from `ReviewInput` via an `onTargetSelect` callback whenever the user releases the mouse or keyboard after selecting text. The parent passes it down to `TargetBadge` and to the predict call. Keep it flat — no reducer needed.

---

## 2. Text Span Selection: The Core UX Pattern

### Recommendation: `textarea` + `selectionStart`/`selectionEnd`

**Use this, not contenteditable.**

```typescript
// ReviewInput.tsx
function ReviewInput({ value, onChange, onTargetSelect }) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSelectionChange = () => {
    const el = textareaRef.current;
    if (!el) return;
    const selected = el.value.slice(el.selectionStart, el.selectionEnd).trim();
    if (selected.length > 0) {
      onTargetSelect(selected);
    }
  };

  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={e => onChange(e.target.value)}
      onMouseUp={handleSelectionChange}
      onKeyUp={handleSelectionChange}
      rows={6}
      placeholder="Dán đoạn review tiếng Việt vào đây..."
    />
  );
}
```

**Why `textarea` wins:**
- `selectionStart`/`selectionEnd` are integer indices into `textarea.value` (UTF-16 code units). Vietnamese characters are all in BMP (U+0000–U+FFFF), so one code unit = one character = one Python `str` index. No offset translation needed.
- Paste, undo, IME composition, mobile keyboards all work natively.
- No z-index stacking or absolute positioning hacks.
- Browsers on iOS/Android handle it correctly.

**Why contenteditable loses:**
- `window.getSelection()` returns a `Range` object with `startOffset` inside a text node — requires careful node traversal to get a flat string offset.
- Normalization of pasted content is browser-dependent (Firefox pastes `<br>` vs Chrome pastes `\n`).
- Vietnamese IME (input method editor) composition events fire in inconsistent order across platforms.
- `execCommand` is deprecated; custom formatting logic is fragile.
- **Avoid** `slate.js` / `draft-js` / `lexical` — they are paragraph editors, not span selectors.

### Visual Selection Feedback

After selection, show the captured phrase in a badge below the textarea — **do not try to highlight the selection inside the textarea with a colored overlay** (this requires a pixel-perfect mirror-div technique that breaks with scrolling, zoom, and variable-width fonts).

```
┌──────────────────────────────────────┐
│ textarea (user selects text here)    │
└──────────────────────────────────────┘
  Target đã chọn: [ nhân viên  ×  ]
  [         Dự đoán         ]
```

The badge has a clear `×` button to deselect. The Predict button is disabled when `targetPhrase` is empty.

---

## 3. Data Fetching: `usePredict` Hook

### Recommendation: Native `fetch` — skip Axios and TanStack Query

For a single endpoint demo, `fetch` is sufficient. TanStack Query adds ~30KB gzipped and a mental model overhead that buys nothing when there is no caching, pagination, or background refetching.

```typescript
// hooks/usePredict.ts
import { useState, useCallback } from 'react';
import type { PredictionResult } from '../types';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

export function usePredict() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (text: string, target: string, model: string) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
        body: JSON.stringify({ text, target, model }),
        signal: AbortSignal.timeout(15_000), // 15s — PhoBERT-large on CPU is ~1–3s
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'TimeoutError') {
        setError('Model took too long to respond. Try a shorter text.');
      } else {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { predict, result, isLoading, error };
}
```

**`AbortSignal.timeout`** (baseline 2023) is supported in all modern browsers and handles the PhoBERT-large CPU latency case cleanly.

**`VITE_API_URL`**: Empty string `''` is intentional — it makes fetch use the Vite dev proxy (see §5), which avoids CORS entirely during development. In production, set to `https://your-api-host.com`.

### API Contract

```typescript
// types.ts
export interface PredictRequest {
  text: string;
  target: string;
  model?: string; // 'phobert_base' | 'phobert_large' | 'xlm_roberta_base'
}

export interface PredictionResult {
  aspect: string;        // e.g. "SERVICE#GENERAL"
  sentiment: string;     // "POSITIVE" | "NEGATIVE" | "NEUTRAL"
  probabilities: {
    aspect: Record<string, number>;    // top-N aspect labels → probability
    sentiment: Record<string, number>; // {"POSITIVE": 0.94, ...}
  };
}
```

---

## 4. Vite Project Setup

### Bootstrap

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

**That's it.** The demo needs no additional runtime dependencies beyond React itself.

### Optional: Tailwind CSS (recommended for speed, not required)

```bash
npm install -D tailwindcss @tailwindcss/vite
```

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

The Vite dev proxy eliminates all CORS issues during development. `fetch('/predict', ...)` in the browser hits Vite's dev server which proxies to FastAPI on port 8000. **FastAPI does NOT need CORSMiddleware for local development** with this setup. (You will need CORS middleware for production deployments where frontend and API are on separate origins.)

### Build Output

```bash
npm run build  # outputs to frontend/dist/
```

`dist/` is a static bundle that can be served by FastAPI itself with `StaticFiles` or any static host (Netlify, Vercel, GitHub Pages). The `index.html` is ~1KB; the JS bundle for this demo will be ~150–200KB gzipped (mostly React).

### `.env` Files

```
# .env.development (committed)
VITE_API_URL=

# .env.production (committed)
VITE_API_URL=https://your-production-api.com
```

All `VITE_` prefixed vars are inlined at build time by Vite. Never put secrets here.

---

## 5. Pre-Loaded Examples: One-Click UX Pattern

### Recommendation: Hard-coded examples in `data/examples.ts`

Do not fetch examples from the backend. Hard-code 2–3 examples per domain in the frontend bundle.

```typescript
// data/examples.ts
export interface ExampleEntry {
  domain: 'hotel' | 'restaurant' | 'mobile';
  domainLabel: string;
  text: string;
  target: string;
  expected: { aspect: string; sentiment: string }; // for display only
}

export const EXAMPLES: ExampleEntry[] = [
  {
    domain: 'hotel',
    domainLabel: '🏨 Khách sạn',
    text: 'nhân viên rất nhiệt tình thân thiện, luôn giúp đỡ mỗi khi tôi cần. giường ngủ êm, thoải mái, không gian phòng rộng rãi.',
    target: 'nhân viên',
    expected: { aspect: 'SERVICE#GENERAL', sentiment: 'POSITIVE' },
  },
  {
    domain: 'hotel',
    domainLabel: '🏨 Khách sạn',
    text: 'chưa có thang máy. chưa chấp nhận thanh toán bằng thẻ. địa điểm dễ tìm, bày trí bằng tre nứa rất mát mẻ.',
    target: 'thang máy',
    expected: { aspect: 'FACILITIES#DESIGN&FEATURES', sentiment: 'NEGATIVE' },
  },
  {
    domain: 'restaurant',
    domainLabel: '🍜 Nhà hàng',
    text: 'hương vị thơm ngon, ăn cay cay rất thích, nêm nếm vừa miệng. ngoài ra menu quán khá đa dạng.',
    target: 'hương vị',
    expected: { aspect: 'FOOD#QUALITY', sentiment: 'POSITIVE' },
  },
  {
    domain: 'mobile',
    domainLabel: '📱 Điện thoại',
    text: 'camera chụp ảnh sắc nét, màu sắc trung thực, chụp ban đêm cũng khá tốt.',
    target: 'camera',
    expected: { aspect: 'CAMERA', sentiment: 'POSITIVE' },
  },
];
```

### Example Card Component

```tsx
// ExamplePicker.tsx
function ExamplePicker({ onSelect }) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      {EXAMPLES.map((ex, i) => (
        <button
          key={i}
          onClick={() => onSelect(ex.text, ex.target)}
          title={`Target: "${ex.target}" → ${ex.expected.aspect}`}
        >
          {ex.domainLabel}
          <span style={{ fontSize: 11, opacity: 0.7 }}>"{ex.target}"</span>
        </button>
      ))}
    </div>
  );
}
```

When a user clicks an example card:
1. `reviewText` is set to `ex.text`
2. `targetPhrase` is set to `ex.target`
3. `predict()` fires immediately (auto-submit)

Auto-predict on example selection is better UX than requiring an extra click — the user can see the result instantly and understand the interaction pattern.

---

## 6. Result Display Components

### AspectLabel: Domain Color-Coded Chip

The aspect label format is `CATEGORY#SUBCATEGORY`. Color by top-level category:

```typescript
const ASPECT_COLORS: Record<string, string> = {
  SERVICE:        '#3b82f6', // blue
  ROOMS:          '#8b5cf6', // purple
  FACILITIES:     '#f59e0b', // amber
  FOOD:           '#10b981', // emerald
  'FOOD&DRINKS':  '#10b981',
  LOCATION:       '#06b6d4', // cyan
  AMBIENCE:       '#ec4899', // pink
  HOTEL:          '#6366f1', // indigo
  RESTAURANT:     '#14b8a6', // teal
  CAMERA:         '#f97316', // orange (mobile)
  BATTERY:        '#84cc16', // lime
  DESIGN:         '#a855f7', // purple
};

function AspectLabel({ aspect }: { aspect: string }) {
  const category = aspect.split('#')[0];
  const color = ASPECT_COLORS[category] ?? '#6b7280';
  return (
    <span style={{ background: color, color: 'white', borderRadius: 4, padding: '2px 10px', fontWeight: 600 }}>
      {aspect}
    </span>
  );
}
```

### SentimentBadge: Traffic Light

```typescript
const SENTIMENT_STYLES = {
  POSITIVE: { background: '#dcfce7', color: '#166534', icon: '😊' },
  NEGATIVE: { background: '#fee2e2', color: '#991b1b', icon: '😞' },
  NEUTRAL:  { background: '#f3f4f6', color: '#374151', icon: '😐' },
};
```

### ProbabilityBars: CSS Width Trick (No Chart Library Needed)

```tsx
// ProbabilityBars.tsx
function ProbabilityBars({ probs, title, topN = 5 }: {
  probs: Record<string, number>;
  title: string;
  topN?: number;
}) {
  const sorted = Object.entries(probs)
    .sort(([, a], [, b]) => b - a)
    .slice(0, topN);

  return (
    <div>
      <p style={{ fontWeight: 600, marginBottom: 6 }}>{title}</p>
      {sorted.map(([label, prob]) => (
        <div key={label} style={{ marginBottom: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
            <span>{label}</span>
            <span>{(prob * 100).toFixed(1)}%</span>
          </div>
          <div style={{ background: '#e5e7eb', borderRadius: 4, height: 6 }}>
            <div style={{
              width: `${prob * 100}%`,
              background: '#3b82f6',
              height: '100%',
              borderRadius: 4,
              transition: 'width 0.4s ease',
            }} />
          </div>
        </div>
      ))}
    </div>
  );
}
```

The `transition: width 0.4s ease` on the bar gives a satisfying animation as results appear. No animation library needed.

### Layout: Two-Panel with Mobile Stack

```
Desktop (≥ 768px):
┌──────────────────────┬──────────────────────┐
│  Input Panel         │  Result Panel        │
│  [ModelSelector]     │                      │
│  [ReviewInput]       │  (empty state:       │
│  [TargetBadge]       │   "Paste a review    │
│  [PredictButton]     │    and select a      │
│  [ExamplePicker]     │    target phrase")   │
└──────────────────────┴──────────────────────┘

Mobile (< 768px): stacked vertically, result below input
```

```css
.demo-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  max-width: 1100px;
  margin: 0 auto;
}
@media (max-width: 768px) {
  .demo-layout { grid-template-columns: 1fr; }
}
```

---

## 7. Vietnamese Text: Pitfalls and UTF-8 Handling

### Critical Pitfall 1: BOM Character (U+FEFF)

**What goes wrong:** The project's data files contain a BOM (`﻿`) at the start (visible in the data: `﻿chưa có thang máy...`). If a user opens a file in Notepad and pastes, the BOM travels with the text. The PhoBERT tokenizer may produce a garbage first token.

**Prevention:** Strip BOM on the frontend before sending:
```typescript
const cleanText = text.replace(/^\uFEFF/, '').trim();
```
Also strip in the FastAPI backend as a second defense.

### Critical Pitfall 2: Sending Character Offsets to Backend

**What goes wrong:** `selectionStart`/`selectionEnd` return UTF-16 code unit indices. Python `str` also uses code points (not bytes). For characters in BMP (U+0000–U+FFFF), one code unit = one code point, so JS indices match Python. **But** if the backend normalizes the text (e.g., strips BOM, normalizes whitespace), the offset from the frontend points to the wrong character in the backend's normalized string.

**Prevention (this project):** **Do not send offsets at all.** The backend API accepts `{text, target}` as strings. The model's `tokenize_text_pair` takes text and target as separate strings — it does not need character positions. This design sidesteps all offset translation bugs.

### Pitfall 3: Unicode Normalization Mismatch (NFC vs NFD)

**What goes wrong:** Vietnamese diacritics can be encoded as:
- NFC (precomposed): `ộ` = single code point U+1ED9
- NFD (decomposed): `ộ` = `o` + combining breve + combining dot below (3 code points)

A user pasting from macOS (default NFC) vs some legacy software (NFD) can send different byte representations of the same visual text. If the backend tokenizer and the frontend disagree, the selected `target` string may not be found as a substring of `text`.

**Prevention:**
```typescript
// Normalize both text and target to NFC before sending
const normalizedText   = text.normalize('NFC').replace(/^\uFEFF/, '').trim();
const normalizedTarget = target.normalize('NFC').trim();
```
Add the same normalization in Python: `text = unicodedata.normalize('NFC', text)`.

### Pitfall 4: Zero-Width and Invisible Characters from Paste

**What goes wrong:** Copying from Word, Google Docs, or PDFs introduces zero-width joiners (U+200D), zero-width non-breaking spaces (U+FEFF), soft hyphens (U+00AD), and other invisible characters that confuse the tokenizer.

**Prevention:** Add a visible warning if the pasted text contains characters outside expected Unicode ranges, or strip them silently:
```typescript
const stripped = text.replace(/[\u200B-\u200D\uFEFF\u00AD]/g, '');
```

### Pitfall 5: `textarea` vs. `<div contenteditable>` for Vietnamese IME

**What goes wrong:** On Android with Vietnamese keyboard (Gboard, SwiftKey), `contenteditable` divs sometimes emit incorrect `compositionend` events, dropping diacritics or duplicating characters. `textarea` elements do not have this problem — they are handled by the OS input stack directly.

**Prevention:** Use `textarea`. Never use `contenteditable` for the main text input area in this demo.

### Pitfall 6: Whitespace Normalization Destroying Target Match

**What goes wrong:** If a user selects a target phrase that contains a non-breaking space (`\u00A0`) from the review text, but the backend normalizes spaces to regular spaces, `target not found in text` errors occur silently (the model still runs but target_mask is all zeros → falls back to CLS pooling).

**Prevention:** Normalize whitespace in both frontend and backend:
```typescript
const normalizeSpaces = (s: string) => s.replace(/\u00A0/g, ' ').replace(/\s+/g, ' ').trim();
```

### Pitfall 7: Font Rendering for Rare Vietnamese Characters

**What goes wrong:** Obscure stacked diacritics (e.g., `ặ`, `ữ`) may not render in all monospace fonts used in code-adjacent UIs.

**Prevention:** Set the textarea font to a system serif or sans-serif that guarantees Vietnamese coverage:
```css
textarea {
  font-family: 'Segoe UI', 'Noto Sans', Arial, sans-serif;
  font-size: 16px;         /* below 16px triggers iOS auto-zoom */
  line-height: 1.6;
}
```
**Always set `font-size: 16px` or larger on iOS** to prevent automatic zoom-in on textarea focus.

---

## 8. Libraries: Use vs. Avoid

### Use

| Package | Why |
|---------|-----|
| `vite` + `@vitejs/plugin-react` | Fast HMR, small builds, TS out of box |
| `typescript` | Catches the `probabilities.aspect` key-access bugs at compile time |
| `@tailwindcss/vite` (optional) | Utility classes make the single-page layout fast to write without a CSS file |

### Avoid

| Package | Why Not |
|---------|---------|
| `axios` | +14KB gzipped for one fetch call. Native `fetch` with `AbortSignal.timeout` is equivalent. |
| `@tanstack/react-query` | Caching, deduplication, background refetch — none relevant for one demo endpoint. |
| `redux` / `zustand` / `jotai` | Three `useState` calls in `App.tsx` are sufficient. |
| `material-ui` / `@chakra-ui/react` | 70–150KB gzipped. Overkill for a demo page. |
| `react-router-dom` | Single page, no navigation. |
| `chart.js` / `recharts` / `d3` | CSS width animation is all that's needed for probability bars. |
| `slate.js` / `draft-js` / `lexical` | Rich text editors — this is a plain text paste field. |
| `react-contenteditable` | Fragile cross-browser, IME issues on Vietnamese keyboards. |
| `lodash` | No utility functions needed beyond `slice` and `sort`. |

---

## 9. Model Selector UX

The project has three relevant encoder variants:

| Key | Display Name | Notes |
|-----|--------------|-------|
| `phobert_large` | PhoBERT Large ⭐ | Best F1 (aspect 43%, sentiment 73%). Default. ~1–3s CPU. |
| `phobert_base` | PhoBERT Base | Faster (~0.5s CPU), slightly lower F1. |
| `xlm_roberta_base` | XLM-RoBERTa Base | Multilingual baseline; weaker on Vietnamese. |

Implementation: a `<select>` dropdown in the header. The selected model key is passed as a query param or body field to `POST /predict`. The backend loads all three models at startup and routes by key (avoids per-request load time).

**UX note:** Show the current model's performance metrics next to the selector as a footnote:
```
PhoBERT Large: Aspect F1 43.2% · Sentiment F1 72.5%
```
This contextualizes why the aspect prediction might sometimes be wrong at ~57% error rate.

---

## 10. Error and Empty States

| State | Display |
|-------|---------|
| No text entered | Textarea placeholder only |
| Text entered, no target | TargetBadge shows "Chưa chọn — hãy bôi đen một cụm từ"; Predict button disabled |
| Loading | Button shows spinner; result area shows skeleton bars |
| API error | Red banner below button: `"Lỗi: {error.message}"` |
| API timeout (>15s) | Specific message: `"Model phản hồi quá lâu. Thử đoạn văn ngắn hơn."` |
| Empty selection (whitespace only) | Ignore mouseup; don't update target badge |
| Result received | Fade-in ResultCard replacing empty state |

---

## 11. FastAPI Backend Requirements (for frontend integration)

The frontend assumes this contract. Deviations cause bugs:

```python
# main.py (FastAPI)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # needed for production deploys
from pydantic import BaseModel

app = FastAPI()

# Only needed when frontend and API are on different origins (production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-frontend.com"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class PredictRequest(BaseModel):
    text: str
    target: str
    model: str = "phobert_large"

class PredictionResult(BaseModel):
    aspect: str
    sentiment: str
    probabilities: dict  # {"aspect": {...}, "sentiment": {...}}

@app.post("/predict", response_model=PredictionResult)
async def predict(req: PredictRequest):
    # Strip BOM and normalize
    text = req.text.replace('\uFEFF', '').strip()
    target = req.target.replace('\uFEFF', '').strip()
    # ... run inference ...
```

**CORS is not needed for local development** if the Vite proxy is configured (§4). Add it for production.

**Response must include `probabilities` with BOTH `aspect` and `sentiment` keys** — the frontend `ProbabilityBars` component renders both. If the backend omits probabilities (e.g., returns only top-1 logits), the frontend will show empty bars silently. Emit softmax probabilities for all classes from the inference function.

---

## 12. Recommended Phase Sequence for Implementation

1. **Vite scaffold** — `npm create vite@latest`, configure proxy, verify dev server runs
2. **Static UI** — `ReviewInput` + `TargetBadge` + `PredictButton` + `ResultCard` with hardcoded mock data
3. **Selection logic** — Wire `onMouseUp`/`onKeyUp` to capture target phrase; verify with Vietnamese text samples from `Data/hotel.jsonl`
4. **`usePredict` hook** — Connect to FastAPI `/predict`; test with curl first
5. **`ExamplePicker`** — Hardcode 4 examples; wire auto-predict on click
6. **`ProbabilityBars`** — Render both aspect and sentiment distributions
7. **Model selector** — Dropdown, pass to API
8. **Error/loading states** — All branches in `usePredict`
9. **Mobile layout** — Test at 375px viewport
10. **Vietnamese text normalization** — Add BOM strip + NFC normalize; test by pasting from Word/file

---

## Sources

- MDN Web Docs: `HTMLTextAreaElement.selectionStart`, `selectionEnd` — stable API, all browsers
- MDN: `AbortSignal.timeout()` — Baseline 2023, available in Chrome 103+, Firefox 100+, Safari 16+
- Vite documentation: `server.proxy` configuration — v5.x stable
- Unicode Standard: BMP range, NFC/NFD normalization, U+FEFF BOM behavior
- Project data: `/Users/baovle/Code/Personal/targeted_ABSA/Data/hotel.jsonl` — verified label format and Vietnamese text samples
- Project metrics: `/Users/baovle/Code/Personal/targeted_ABSA/reports/test_metrics.json` — F1 scores for model selector display
- Project config: `/Users/baovle/Code/Personal/targeted_ABSA/config/label_structures.yaml` — confirmed multitask_aspect_sentiment output structure
