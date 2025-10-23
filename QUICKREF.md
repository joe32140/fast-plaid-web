# ⚡ FastPlaid Quick Reference

## 🎯 Which Implementation to Use?

| Need | Use | Why |
|------|-----|-----|
| Python API | **Native** | `pip install fast-plaid` |
| Server search | **Native** | CUDA support, full PLAID |
| Browser demo | **WASM** | No server, instant load |
| GitHub Pages | **WASM** | <10 MB, static files |
| Large indexes (>10k) | **Native** | Better for offline building |
| Small indexes (<10k) | **WASM** | Fast, self-contained |

## 📦 File Formats

| Format | Size | Load Time | Created By | Used By |
|--------|------|-----------|------------|---------|
| `.plaid` | Large | N/A | Native Rust | Python/CLI |
| `embeddings.bin` | 49 MB | 15s | Python script | WASM (on-the-fly) |
| `.fastplaid` | 6 MB | <1s | WASM (offline) | WASM (instant) |

## 🔧 Build Commands

### Option 1: On-the-fly (Simple)
```bash
# Compute embeddings only
python scripts/build_offline_wasm_index.py \
    --papers data/papers_1000.json \
    --output demo/data

# Browser quantizes when page loads (10-15s)
```

### Option 2: Precomputed (Fast)
```bash
# Step 1: Compute embeddings
python scripts/build_offline_wasm_index.py \
    --papers data/papers_1000.json \
    --output demo/data

# Step 2: Build .fastplaid offline
node scripts/build_fastplaid_index.js \
    demo/data \
    demo/data/index.fastplaid

# Browser loads instantly (<1s)
```

## 🚀 Browser Usage

### Load Precomputed Index
```javascript
import init, { FastPlaidQuantized } from './pkg/fast_plaid_rust.js';
await init();

const fastplaid = new FastPlaidQuantized();
const indexBytes = await fetch('./data/index.fastplaid')
    .then(r => r.arrayBuffer());
await fastplaid.load_index(new Uint8Array(indexBytes));

// Search
const queryShape = new Uint32Array([1, numTokens, embeddingDim]);
const resultJson = fastplaid.search(queryEmb, queryShape, topK);
const results = JSON.parse(resultJson);
```

### Build On-the-fly
```javascript
const fastplaid = new FastPlaidQuantized();

// Load embeddings + quantize
await fastplaid.load_documents_quantized(
    flatEmbeddings,    // Float32Array
    docInfo,           // BigInt64Array [id, numTokens, id, numTokens, ...]
    256                // num centroids
);

// Search immediately
const resultJson = fastplaid.search(queryEmb, queryShape, topK);
```

## 📊 Performance (1000 papers)

| Metric | Value |
|--------|-------|
| Index size (float32) | 49.5 MB |
| Index size (4-bit) | 6.2 MB |
| Compression ratio | 8x |
| IVF clusters | 32 |
| Papers per cluster | ~31 |
| Clusters probed | 6 (20%) |
| Candidates searched | ~190 (19%) |
| Search time (Direct) | 86 ms |
| Search time (IVF) | 23 ms |
| Speedup | 3.7x |

## 🔍 Console Output

### Building Index
```
🚀 Initializing FastPlaid WASM with 4-bit quantization...
📥 Loading and quantizing documents...
   Total embedding data: 12938496 floats (49.53 MB uncompressed)
✅ Quantized 1000 documents:
   Original size: 49.53 MB
   Compressed size: 6.19 MB
   Compression ratio: 8.0x
🎯 Building IVF index for fast search...
   Creating 32 IVF clusters...
   ✅ IVF index built:
      32 clusters, avg 31.2 docs/cluster
```

### Searching
```
🔍 Query: reinforcement learning
   Tokens: 6 Dim: 48
🔍 IVF Search: Probing 6 clusters, 190 candidates out of 1000 total docs
⚡ IVF: Probed 6 clusters, searched 190/1000 papers
✅ Direct MaxSim: 10 results in 86.5 ms (all 1000 papers)
✅ FastPlaid: 10 results in 23.0 ms
   🎯 IVF: 6 clusters → 190 candidates (19.0% of papers)
   ⚡ Speedup: 3.8x faster than Direct MaxSim
```

## 🐛 Common Issues

### "Invalid index file: bad magic number"
**Fix:** Rebuild index with `node scripts/build_fastplaid_index.js`

### "WASM table growth error"
**Fix:** Refresh page (Ctrl+Shift+R), close tabs, or use Chrome

### Slow loading (>5s)
**Fix:** Use precomputed `.fastplaid` instead of `embeddings.bin`

### Search results differ slightly
**Reason:** IVF is approximate search (searches ~30% of papers)
**Solution:** Normal! Top results should be the same

## 📚 Documentation

- [README.md](README.md) - Main documentation
- [OFFLINE_INDEX_GUIDE.md](OFFLINE_INDEX_GUIDE.md) - Complete workflows
- [SUMMARY.md](SUMMARY.md) - Implementation details

## 🎓 Architecture

```
┌─────────────────────────────────────────────┐
│ FastPlaid Project                           │
├─────────────────────────────────────────────┤
│                                             │
│ ┌─────────────────┐  ┌──────────────────┐ │
│ │ Native Rust     │  │ WASM             │ │
│ │ (search/index/) │  │ (lib_wasm_       │ │
│ │                 │  │  quantized.rs)   │ │
│ ├─────────────────┤  ├──────────────────┤ │
│ │ • Full PLAID    │  │ • 4-bit quant    │ │
│ │ • Multi-stage   │  │ • IVF clustering │ │
│ │ • Candle        │  │ • Pure Rust      │ │
│ │ • CUDA support  │  │ • Browser native │ │
│ │ • Python API    │  │ • .fastplaid fmt │ │
│ └─────────────────┘  └──────────────────┘ │
│         │                      │           │
│         │                      │           │
│   Python/CLI            GitHub Pages      │
│   Server-side           Browser demo      │
└─────────────────────────────────────────────┘
```

## 🎯 Decision Tree

```
Need to search ColBERT embeddings?
│
├─ In browser?
│  ├─ Yes → Use WASM
│  │  ├─ GitHub Pages? → Build .fastplaid offline
│  │  └─ Local dev? → On-the-fly quantization OK
│  │
│  └─ No → Use Native
│     ├─ Python? → pip install fast-plaid
│     └─ CLI? → Build with cargo
│
└─ Dataset size?
   ├─ <10k papers → WASM is perfect
   └─ >10k papers → Native recommended
```

---

**Questions?** See [OFFLINE_INDEX_GUIDE.md](OFFLINE_INDEX_GUIDE.md) or open an issue!
