# Domain Context: Vietnamese ABSA Demo Website

**Researched:** 2026-06-27  
**Confidence:** HIGH (direct codebase analysis + dataset inspection)  
**Source:** Codebase analysis (`vocabulary.py`, `config/`, `Data/*.jsonl`, `reports/test_metrics.json`) + domain knowledge

---

## 1. Model Performance Reality Check

Before selecting demo examples, understand what the model actually achieves:

| Metric | Score | Framing |
|--------|-------|---------|
| Sentiment F1 (macro) | **72.5%** | ✅ Solid — lead with this |
| Aspect F1 (macro) | **43.2%** | ⚠️ Moderate — ~44 fine-grained categories, task is hard |
| Joint F1 (macro) | **41.0%** | ⚠️ Moderate — product of two tasks |
| Sentiment accuracy | **91.2%** | ✅ Impressive — lead with accuracy for non-ML users |
| Aspect accuracy | **66.9%** | ✅ Presentable |

**Key insight:** The model is genuinely good at sentiment (72% F1, 91% accuracy). Aspect prediction is harder due to 40+ fine-grained categories and class imbalance. Frame the demo accordingly: *sentiment detection is the hero, aspect categorization is the bonus*.

---

## 2. Curated Demo Examples

Selection criteria: short texts (≤ 120 chars), unambiguous target phrases, explicit sentiment words, common aspect categories, sourced from actual training distribution. All texts are raw Vietnamese (no pre-segmentation needed — see §4).

### Domain: Hotel

#### Example H-1 — Service, clear positive
```json
{
  "text": "nhân viên tiếp đón ân cần, rất vui vẻ và thân thiện.",
  "target": "nhân viên",
  "expected_aspect": "SERVICE#GENERAL",
  "expected_sentiment": "POSITIVE",
  "domain": "hotel",
  "why_good": "Most common aspect category. 'ân cần', 'vui vẻ', 'thân thiện' are strong positive markers. Short sentence with no distractor."
}
```

#### Example H-2 — Mixed in same review (great for showing per-target granularity)
```json
{
  "text": "khăn tắm không được thay hằng ngày. phòng có mùi thơm tinh dầu rất dễ chịu, thoải mái.",
  "target_a": "khăn tắm",
  "expected_aspect_a": "SERVICE#CLEANLINESS",
  "expected_sentiment_a": "NEGATIVE",
  "target_b": "phòng",
  "expected_aspect_b": "ROOMS#COMFORT",
  "expected_sentiment_b": "POSITIVE",
  "domain": "hotel",
  "why_good": "Demonstrates the core value prop: same review, different targets → different aspect + sentiment. Negation ('không được thay') is explicit."
}
```

#### Example H-3 — Room cleanliness, clear positive
```json
{
  "text": "khách sạn đẹp, sang trọng. nhân viên thân thiện, nhiệt tình. phòng rất sạch sẽ.",
  "target": "phòng",
  "expected_aspect": "ROOMS#CLEANLINESS",
  "expected_sentiment": "POSITIVE",
  "domain": "hotel",
  "why_good": "'sạch sẽ' is an unambiguous positive cleanliness marker. Classic hotel review pattern."
}
```

#### Example H-4 — Facilities, clear negative
```json
{
  "text": "chúng tôi không cảm thấy thoải mái vì ở chỉ 1 ngày mà cúp điện 3,4 lần. thang máy thì quá nóng và quá chậm chạp.",
  "target": "thang máy",
  "expected_aspect": "FACILITIES#COMFORT",
  "expected_sentiment": "NEGATIVE",
  "domain": "hotel",
  "why_good": "'quá nóng', 'quá chậm chạp' are strong negatives. Elevator (thang máy) is a concrete facility noun."
}
```

---

### Domain: Restaurant

#### Example R-1 — Food quality, clear positive
```json
{
  "text": "hương vị thơm ngon, ăn cay cay rất thích, nêm nếm vừa miệng.",
  "target": "hương vị",
  "expected_aspect": "FOOD#QUALITY",
  "expected_sentiment": "POSITIVE",
  "domain": "restaurant",
  "why_good": "'thơm ngon', 'rất thích', 'vừa miệng' pile on positive signals. FOOD#QUALITY is the single most frequent restaurant aspect."
}
```

#### Example R-2 — Ambience, positive
```json
{
  "text": "quán rộng rãi, view khá đẹp và cũng thoáng lắm.",
  "target": "quán",
  "expected_aspect": "AMBIENCE#GENERAL",
  "expected_sentiment": "POSITIVE",
  "domain": "restaurant",
  "why_good": "'rộng rãi', 'đẹp', 'thoáng' are clear positive ambience descriptors."
}
```

#### Example R-3 — Service, negative (slowness)
```json
{
  "text": "khách của quán đông nên nhiều khi nhân viên phục vụ không được nhanh cho lắm.",
  "target": "nhân viên",
  "expected_aspect": "SERVICE#GENERAL",
  "expected_sentiment": "NEGATIVE",
  "domain": "restaurant",
  "why_good": "Explicit causal chain ('vì đông khách → không nhanh') → clear negative. Matches service patterns model saw heavily in training."
}
```

---

### Domain: Mobile Phone

#### Example M-1 — Performance + Battery, both positive (extreme brevity)
```json
{
  "text": "máy mượt, pin trâu.",
  "target_a": "máy",
  "expected_aspect_a": "PERFORMANCE",
  "expected_sentiment_a": "POSITIVE",
  "target_b": "pin",
  "expected_aspect_b": "BATTERY",
  "expected_sentiment_b": "POSITIVE",
  "domain": "mobile",
  "why_good": "Classic colloquial Vietnamese mobile review. 'Mượt' (smooth/fluid) and 'trâu' (literally 'buffalo' → sturdy/long-lasting) are ultra-common positive markers that the model has seen thousands of times."
}
```

#### Example M-2 — Mixed: Design positive, Features negative
```json
{
  "text": "kiểu dáng thì đẹp, cầm chắc tay, nhưng loa nhỏ quá, nhân viên phục vụ rất nhiệt tình.",
  "target_a": "kiểu dáng",
  "expected_aspect_a": "DESIGN",
  "expected_sentiment_a": "POSITIVE",
  "target_b": "loa",
  "expected_aspect_b": "FEATURES",
  "expected_sentiment_b": "NEGATIVE",
  "domain": "mobile",
  "why_good": "Shows target-sensitivity: 'nhưng loa nhỏ quá' (but the speaker is too small) flips to NEGATIVE despite surrounding positives. Great for demonstrating targeted vs. sentence-level SA."
}
```

#### Example M-3 — Price, positive
```json
{
  "text": "gía này thì quá ổn so với cấu hình!",
  "target": "gía",
  "expected_aspect": "PRICE",
  "expected_sentiment": "POSITIVE",
  "domain": "mobile",
  "why_good": "Short, price-focused sentence. 'Quá ổn so với cấu hình' (very reasonable for the specs) is unambiguous price-positive. Also note intentional misspelling 'gía' → proves model handles noisy input."
}
```

#### Example M-4 — Camera, neutral
```json
{
  "text": "camera ổn chụp lúc sáng đẹp nhưng ban đêm chỉ tạm thôi.",
  "target": "camera",
  "expected_aspect": "CAMERA",
  "expected_sentiment": "NEUTRAL",
  "domain": "mobile",
  "why_good": "Demonstrates NEUTRAL: good in daylight, bad at night → balanced assessment. The model's sentiment accuracy is high (91%), making neutral detection a credible showcase."
}
```

---

### Summary Table

| ID | Domain | Text (excerpt) | Target | Aspect | Sentiment |
|----|--------|----------------|--------|--------|-----------|
| H-1 | Hotel | nhân viên tiếp đón ân cần... | nhân viên | SERVICE#GENERAL | POSITIVE |
| H-2a | Hotel | khăn tắm không được thay... | khăn tắm | SERVICE#CLEANLINESS | NEGATIVE |
| H-2b | Hotel | phòng có mùi thơm tinh dầu... | phòng | ROOMS#COMFORT | POSITIVE |
| H-3 | Hotel | phòng rất sạch sẽ | phòng | ROOMS#CLEANLINESS | POSITIVE |
| H-4 | Hotel | thang máy thì quá nóng... | thang máy | FACILITIES#COMFORT | NEGATIVE |
| R-1 | Restaurant | hương vị thơm ngon... | hương vị | FOOD#QUALITY | POSITIVE |
| R-2 | Restaurant | quán rộng rãi, view đẹp... | quán | AMBIENCE#GENERAL | POSITIVE |
| R-3 | Restaurant | nhân viên phục vụ không nhanh | nhân viên | SERVICE#GENERAL | NEGATIVE |
| M-1a | Mobile | máy mượt... | máy | PERFORMANCE | POSITIVE |
| M-1b | Mobile | pin trâu | pin | BATTERY | POSITIVE |
| M-2a | Mobile | kiểu dáng thì đẹp... | kiểu dáng | DESIGN | POSITIVE |
| M-2b | Mobile | loa nhỏ quá | loa | FEATURES | NEGATIVE |
| M-3 | Mobile | gía này quá ổn... | gía | PRICE | POSITIVE |
| M-4 | Mobile | camera ổn chụp ban đêm chỉ tạm | camera | CAMERA | NEUTRAL |

**Coverage:** 3 domains × all 3 sentiments × diverse aspects. H-2 and M-2 are the anchor examples for the demo homepage (mixed sentiment in one review = the model's unique value).

---

## 3. Recommended Encoder for Demo

### Key finding: Word segmentation is NOT required

**Inspect the evidence:**
- `encoders.yaml` marks `requires_word_segmentation: true` for PhoBERT — this is a design documentation flag
- `vocabulary.py` does NOT apply any pre-segmentation step before calling `AutoTokenizer`
- All raw JSONL data is **unsegmented colloquial Vietnamese**: "máy mượt . pin trâu", "hương vị thơm ngon" — no underscore-joined compounds, no VnCoreNLP pre-processing visible
- The trained `best_model.pt` checkpoint (`phobert_large_target_attn_weighted`) was therefore trained on raw Vietnamese text processed directly through PhoBERT's BPE tokenizer

**Conclusion:** To match training conditions, the demo must feed raw Vietnamese text — no segmentation pipeline. PhoBERT's BPE tokenizer handles raw text adequately; the model learned from exactly this input format.

### Recommendation: **Use PhoBERT-large directly, no segmentation step**

```
Encoder:  vinai/phobert-large   (best_model.pt checkpoint)  
Input:    raw Vietnamese text → AutoTokenizer → inference
Segment:  SKIP — would mismatch training distribution
```

| Option | F1 (joint) | Input needed | Setup complexity | Verdict |
|--------|-----------|--------------|-----------------|---------|
| **PhoBERT-large + raw text** (trained) | **41%** | Raw Vietnamese | Low — just load checkpoint | ✅ **Recommended** |
| PhoBERT-large + VnCoreNLP segmented | Unknown | Segmented | High — Java/JAR download | ❌ Mismatches training |
| XLM-RoBERTa-base/large | ~35-38% est. | Raw text | Low — no checkpoint available | ❌ No trained checkpoint |

### If you ever retrain with pre-segmented data (future work)

If a future training run explicitly pre-segments with VnCoreNLP, the demo API would use:

```python
# Option A — py_vncorenlp (recommended if segmentation needed)
# Pure Python wrapper, downloads JAR automatically from HuggingFace
# No separate Java install required by user
pip install py_vncorenlp
import py_vncorenlp
model = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
segmented = " ".join(model.word_segment(text))

# Option B — underthesea (simpler, no Java dependency)
pip install underthesea
from underthesea import word_tokenize
segmented = word_tokenize(text, format="text")
# Note: different segmentation quality vs VnCoreNLP, may introduce train/serve skew
```

**Avoid for demo:** The original VnCoreNLP Java toolkit (requires JDK, complex setup, not user-friendly for a demo environment).

---

## 4. Vietnamese Text Handling (Web)

### Font Recommendations

Vietnamese requires full Unicode diacritic support (6 tones × consonant combinations = hundreds of unique glyphs). All modern browsers render Vietnamese correctly with the right font stack.

**Recommended font stack:**
```css
font-family: 'Be Vietnam Pro', 'Noto Sans', 'Noto Sans Vietnamese', 
             system-ui, -apple-system, sans-serif;
```

Google Fonts import:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap">
```

| Font | Pros | Cons |
|------|------|------|
| **Be Vietnam Pro** | Purpose-built for Vietnamese, modern, clean | Requires Google Fonts CDN |
| **Noto Sans Vietnamese** | Complete glyph coverage, Google fallback | Less aesthetic |
| **system-ui** | Zero load time | Inconsistent quality across OS |

### Input/Output Rendering

```css
/* Textarea for Vietnamese input */
.vi-input {
  font-family: 'Be Vietnam Pro', system-ui, sans-serif;
  font-size: 16px;        /* never below 16px — triggers iOS zoom */
  line-height: 1.6;       /* Vietnamese diacritics need vertical room */
  unicode-bidi: normal;   /* Left-to-right, no special handling needed */
}

/* Highlighted target span */
.target-highlight {
  background: rgba(59, 130, 246, 0.2);
  border-radius: 3px;
  padding: 0 2px;
}
```

### Backend Text Normalization

```python
import unicodedata

def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text before tokenization."""
    # Normalize Unicode to NFC (composed form) — handles copy-paste from Word/iOS
    text = unicodedata.normalize('NFC', text)
    # Collapse multiple whitespace (common in copy-pasted reviews)
    text = ' '.join(text.split())
    return text.strip()
```

**Why NFC matters:** Vietnamese diacritics can be encoded as pre-composed (NFC) or decomposed (NFD) characters. iOS/macOS tend to produce NFD on copy-paste from some apps; the tokenizer expects NFC. Without normalization, "ngon" (with NFC ơ) and "ngon" (with NFD ơ) tokenize differently.

### Character-offset Safety

The raw JSONL annotations use byte/character offsets (`[start, end]`) to locate the target span. When displaying highlighted text in the browser, use the same character indices:

```javascript
// Safe span extraction for highlighted display
function highlightTarget(text, start, end) {
  return [
    text.slice(0, start),
    `<mark class="target">${text.slice(start, end)}</mark>`,
    text.slice(end)
  ].join('');
}
// No special handling needed — JS strings are UTF-16, Vietnamese is BMP, safe
```

---

## 5. Confidence & Uncertainty Display for Non-ML Users

### Design Principles

1. **Never hide uncertainty** — show confidence for every prediction
2. **Use plain language** — "The model is fairly confident this is about..." not "softmax probability 0.72"
3. **Sentiment is the reliable signal** — lead with it (91% accuracy); present aspect as "likely category"
4. **Show the alternatives** — top-3 aspects with bars makes the model feel transparent and trustworthy

### Confidence Tiers

```
Probability → User-facing label     → Visual treatment
≥ 0.80      → "Confident"           → solid color chip
0.55–0.79   → "Likely"              → standard color chip
0.40–0.54   → "Uncertain"           → dashed border, yellow tint
< 0.40      → "Low confidence"      → grey chip + info tooltip
```

### Recommended UI Components

```
┌─────────────────────────────────────────────────────────┐
│  PREDICTION RESULT                                       │
│                                                          │
│  Target phrase: [nhân viên]                             │
│                                                          │
│  ╔═══════════════╗   ╔══════════════════════╗           │
│  ║   SENTIMENT   ║   ║       ASPECT         ║           │
│  ║               ║   ║                      ║           │
│  ║  ● POSITIVE   ║   ║  ● SERVICE#GENERAL   ║           │
│  ║    Confident  ║   ║    Likely            ║           │
│  ╚═══════════════╝   ╚══════════════════════╝           │
│                                                          │
│  Sentiment breakdown:    Aspect top-3:                  │
│  POSITIVE  ████████░░  82%   SERVICE#GENERAL  ██████░░  68%  │
│  NEGATIVE  ██░░░░░░░░  13%   HOTEL#GENERAL    ████░░░░  31%  │
│  NEUTRAL   █░░░░░░░░░   5%   ROOMS#DESIGN     █░░░░░░░   8%  │
│                                                          │
│  ℹ️ Aspect F1 is ~43% on test data. When confidence is  │
│     below 60%, treat the aspect label as a suggestion.   │
└─────────────────────────────────────────────────────────┘
```

### Color System

```
POSITIVE  → green-500  (#22c55e) bg, green-100 chip background
NEGATIVE  → red-500    (#ef4444) bg, red-100 chip background
NEUTRAL   → slate-500  (#64748b) bg, slate-100 chip background
Low conf  → amber-400  (#fbbf24) border indicator
```

### Inline Limitations Disclosure

Show once per session (collapsible), not on every prediction:

```
ℹ️ About this model
This is a research prototype trained on Vietnamese hotel, restaurant, 
and mobile phone reviews. Performance varies by input:
  • Sentiment detection: ~91% accuracy (reliable)
  • Aspect category: ~67% accuracy (directional)
Hard cases: sarcasm, very short phrases, mixed-language text
Model: PhoBERT-large, trained 2025 on Vietnamese VLSP/ViSFD-style data
```

### When Prediction Fails Gracefully

```python
# In the API response
{
  "aspect": "SERVICE#GENERAL",
  "aspect_confidence": 0.68,
  "sentiment": "POSITIVE",
  "sentiment_confidence": 0.82,
  "aspect_top3": [
    {"label": "SERVICE#GENERAL", "probability": 0.68},
    {"label": "HOTEL#GENERAL",   "probability": 0.21},
    {"label": "ROOMS#DESIGN",    "probability": 0.11}
  ],
  "sentiment_probs": {
    "POSITIVE": 0.82,
    "NEGATIVE": 0.13,
    "NEUTRAL":  0.05
  },
  "low_confidence": false  // true when max sentiment prob < 0.55
}
```

---

## 6. What Makes the Model Look Good (and Bad)

### High-confidence cases (prioritize in demos)

| Pattern | Example | Why model succeeds |
|---------|---------|-------------------|
| Explicit sentiment adjective + clear noun | "dịch vụ tốt", "camera đẹp" | Strong lexical signal aligned with aspect |
| Classic Vietnamese review phrases | "nhân viên nhiệt tình", "pin trâu", "phòng sạch" | Seen thousands of times in training |
| Single-sentence, single-target | "giường rất êm và thoải mái" | No conflicting context |
| Short target + clear surrounding | "thang máy quá chậm" | Target mask isolates signal well |

### Low-confidence / hard cases (avoid in primary demo flow)

| Pattern | Example | Why model struggles |
|---------|---------|-------------------|
| Implicit sentiment (no explicit adjective) | "phòng này hơi... khác" | Vague, requires inference |
| Negation chains | "không phải là không tốt" | Double negation |
| Mixed-language slang | "pin ez, loa zin tàu" | OOV / low-frequency tokens |
| Very generic targets | "cái này", "nó" | Pronouns, no lexical signal |
| Rare aspect categories | `HOTEL#STYLE&OPTIONS` | Listed as unseen in test_metrics.json |
| Aspect about absent features | "không có thang máy" → FACILITIES#NEGATIVE | Model tends to mis-predict aspect when feature is absent |

### Known failure modes from test_metrics.json

```
Unseen labels in test set:
  - LOCATION#MISCELLANEOUS#POSITIVE    (seen but low-frequency)

Completely skipped (label never in training):
  - HOTEL#STYLE&OPTIONS#NEGATIVE
  - SERVICE#DESIGN&FEATURES#POSITIVE

Implication: These aspect categories will always predict the nearest
high-frequency category. Don't include these in demo examples.
```

---

## 7. Demo API Pre-processing Checklist

```python
def preprocess_for_inference(text: str, target: str) -> tuple[str, str]:
    """
    Pre-process (text, target) pair to match training conditions.
    Training used raw Vietnamese text + PhoBERT BPE tokenizer, no segmentation.
    """
    import unicodedata
    
    # 1. Unicode NFC normalization (handle iOS/macOS paste artifacts)
    text   = unicodedata.normalize('NFC', text)
    target = unicodedata.normalize('NFC', target)
    
    # 2. Collapse whitespace
    text   = ' '.join(text.split())
    target = ' '.join(target.split())
    
    # 3. Verify target is a substring of text (exact match required for target_mask)
    if target not in text:
        # Try case-insensitive match + restore original case
        lower_text = text.lower()
        lower_target = target.lower()
        idx = lower_text.find(lower_target)
        if idx >= 0:
            target = text[idx:idx+len(target)]  # use cased version from text
        else:
            raise ValueError(f"Target '{target}' not found in text")
    
    return text, target
```

**Critical constraint:** `_build_target_mask` in `vocabulary.py` uses a reverse-scan sliding window to locate the target's token IDs within the full encoded sequence. If the target string is not present in the text, the mask is all-zeros → the model falls back to CLS-only representation and may give degraded predictions. Always validate target membership before calling the model.

---

## 8. Aspect Label Display Mapping

Raw model outputs look technical to non-ML users. Map them to human-readable descriptions:

```python
ASPECT_DISPLAY = {
    # Hotel
    "SERVICE#GENERAL":          "Service",
    "SERVICE#QUALITY":          "Service Quality",
    "SERVICE#CLEANLINESS":      "Cleanliness (Service)",
    "ROOMS#COMFORT":            "Room Comfort",
    "ROOMS#CLEANLINESS":        "Room Cleanliness",
    "ROOMS#DESIGN&FEATURES":    "Room Design",
    "ROOM_AMENITIES#COMFORT":   "Amenity Comfort",
    "ROOM_AMENITIES#QUALITY":   "Amenity Quality",
    "ROOM_AMENITIES#CLEANLINESS": "Amenity Cleanliness",
    "FACILITIES#COMFORT":       "Facility Comfort",
    "FACILITIES#DESIGN&FEATURES": "Facility Design",
    "FACILITIES#QUALITY":       "Facility Quality",
    "HOTEL#GENERAL":            "Overall Hotel",
    "HOTEL#DESIGN&FEATURES":    "Hotel Design",
    "HOTEL#CLEANLINESS":        "Hotel Cleanliness",
    "LOCATION#GENERAL":         "Location",
    "LOCATION#COMFORT":         "Location Convenience",
    "FOOD&DRINKS#GENERAL":      "Food & Drinks",
    "FOOD&DRINKS#QUALITY":      "Food Quality",
    "FOOD&DRINKS#STYLE&OPTIONS": "Food Variety",
    # Restaurant
    "FOOD#QUALITY":             "Food Quality",
    "FOOD#STYLE_OPTIONS":       "Food Variety",
    "FOOD#PRICE":               "Food Price",
    "FOOD#GENERAL":             "Food (General)",
    "SERVICE#GENERAL":          "Service",
    "AMBIENCE#GENERAL":         "Atmosphere",
    "AMBIENCE#QUALITY":         "Ambience Quality",
    "RESTAURANT#GENERAL":       "Restaurant (Overall)",
    "RESTAURANT#PRICE":         "Restaurant Price",
    "DRINKS#PRICE":             "Drinks Price",
    # Mobile
    "BATTERY":                  "Battery Life",
    "CAMERA":                   "Camera",
    "DESIGN":                   "Design / Build",
    "FEATURES":                 "Features",
    "GENERAL":                  "Overall",
    "PERFORMANCE":              "Performance",
    "PRICE":                    "Price / Value",
    "SCREEN":                   "Display",
    "SER&ACC":                  "Sales & Service",
    "STORAGE":                  "Storage",
}

SENTIMENT_DISPLAY = {
    "POSITIVE": {"label": "Positive", "emoji": "😊", "color": "green"},
    "NEGATIVE": {"label": "Negative", "emoji": "😞", "color": "red"},
    "NEUTRAL":  {"label": "Neutral",  "emoji": "😐", "color": "gray"},
}
```

---

## 9. Roadmap Implications for Demo Build

### Phase ordering recommendation

1. **Inference API first** (FastAPI wrapping `TargetedABSAModel`) — everything else depends on this  
   - Critical: implement `preprocess_for_inference` with target substring validation  
   - Serve PhoBERT-large directly, no segmentation pipeline  
   - Return full probability vectors, not just argmax  

2. **Frontend: examples-first** — ship with the 14 curated examples clickable  
   - Users who see good examples trust the model; blank textarea is intimidating  
   - Pre-fill domain selector → show matching examples  

3. **Confidence visualization** — sentiment bars are the hero; aspect bars are supporting  
   - Use `sentiment_confidence` as the primary trust signal  
   - Show aspect top-3, not just top-1  

4. **Limitations panel** — one collapsible info box, not inline per prediction  
   - F1 scores should be visible but not alarming  

### Likely needs deeper research (phase-specific)

- **Inference latency on CPU**: PhoBERT-large is ~1-3s/request on CPU; may need `torch.compile` or quantization for snappy demo UX
- **Model loading strategy**: `best_model.pt` is a state-dict only — needs `TargetedABSAModel` class code co-deployed with API
- **Cross-domain vocabulary**: The `Vocabulary` class scans training data to build label maps at startup — API server needs access to training JSONL files (or a pre-serialized vocab pickle)

### Not needed for demo

- VnCoreNLP, py_vncorenlp, or underthesea — training distribution is raw Vietnamese
- Multi-model switching beyond PhoBERT-large (no other checkpoints available)
- Custom Vietnamese keyboard or IME — users paste their own text
