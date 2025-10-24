# 🚀 Offline Index Building Guide

This guide explains how to build FastPlaid indexes offline for instant browser loading.

## 📊 Architecture Overview

### Two Implementations:

1. **Native Rust (`search/` modules)** - Full PLAID with Product Quantization
   - Used for: Python bindings, CLI tools, server-side search
   - Features: Multi-stage PQ, Candle tensors, CUDA support
   - Output: `.plaid` index files

2. **WASM (`lib_wasm_quantized.rs`)** - Lightweight 4-bit + IVF
   - Used for: Browser demos, GitHub Pages
   - Features: 4-bit quantization, IVF clustering, pure Rust
   - Output: `.fastplaid` binary files OR on-the-fly quantization

---

## 🎯 Workflow Options

### Option 1: Load Embeddings + Quantize in Browser (Current)

**Use when:** Prototyping, small datasets

```
Python: Compute embeddings → Save embeddings.bin (49 MB)
  ↓
Browser: Load embeddings.bin → WASM quantizes (10s) → Ready to search
```

**Pros:** Simple, no offline processing
**Cons:** 10-15 second load time in browser

### Option 2: Pre-build .fastplaid Index (Recommended)

**Use when:** Production, GitHub Pages, fast load times

```
Python: Compute embeddings → Save embeddings.bin (49 MB)
  ↓
Node.js: WASM quantizes offline → Save .fastplaid (6 MB)
  ↓
Browser: Load .fastplaid → Instant search (0.5s load)
```

**Pros:** Instant loading, 8x smaller
**Cons:** Requires Node.js build step

---

## 🔧 Step-by-Step Instructions

### Step 1: Compute Embeddings (Python)

```bash
python scripts/build_offline_wasm_index.py \
    --papers data/papers_1000.json \
    --output demo/data/precomputed \
    --model mixedbread-ai/mxbai-edge-colbert-v0-17m
```

**Output:**
- `demo/data/precomputed/embeddings.bin` (49 MB) - Float32 embeddings
- `demo/data/precomputed/embeddings_meta.json` - Metadata
- `demo/data/precomputed/papers_metadata.json` - Paper titles/abstracts

**What it does:**
1. Loads papers from JSON
2. Encodes with ColBERT model (256 → 512 → 48 dims with 2_Dense)
3. L2 normalizes token embeddings
4. Saves in binary format for WASM

### Step 2A: Use in Browser Directly (Simple)

```javascript
// In browser
const indexLoader = new PrecomputedIndexLoader('./data/precomputed');
const papers = await indexLoader.loadPapersMetadata();
const directData = await indexLoader.loadDirectMaxSimEmbeddings();

// WASM quantizes on-the-fly
const fastplaid = new FastPlaidQuantized();
const flatEmb = new Float32Array(/* flatten embeddings */);
const docInfo = new BigInt64Array(/* doc ids and token counts */);
await fastplaid.load_documents_quantized(flatEmb, docInfo, 256);

// Ready to search!
```

**Load time:** ~15 seconds (49 MB download + 10s quantization)

### Step 2B: Pre-build .fastplaid Index (Fast)

```bash
node scripts/build_fastplaid_index.js \
    demo/data/precomputed \
    demo/data/index.fastplaid
```

**Output:**
- `demo/data/index.fastplaid` (6.2 MB) - Pre-quantized index with IVF

**What it does:**
1. Loads embeddings.bin
2. Uses WASM to quantize to 4-bit (256 centroids)
3. Builds IVF index (32 clusters for 1000 papers)
4. Saves to binary `.fastplaid` format

**Then in browser:**

```javascript
// Load pre-built index
const fastplaid = new FastPlaidQuantized();
const indexBytes = await fetch('./data/index.fastplaid')
    .then(r => r.arrayBuffer());
await fastplaid.load_index(new Uint8Array(indexBytes));

// Ready to search instantly!
```

**Load time:** ~0.5 seconds (6 MB download, no processing)

### Step 2C: Build in Browser (No Node.js Required)

If you don't have Node.js available, you can build the index directly in the browser:

1. **Open demo in browser:**
   ```bash
   cd demo && python serve.py
   # Visit http://localhost:8000/
   ```

2. **Wait for index to build** (10-15 seconds)

3. **Open browser console** (F12) and run:
   ```javascript
   window.saveFastPlaidIndex();
   ```

4. **Move the downloaded file:**
   ```bash
   mv ~/Downloads/index.fastplaid demo/data/
   ```

5. **Refresh the page** - it will now load the precomputed index instantly!

**Note:** The demo exposes `window.fastPlaidWasm` and `window.saveFastPlaidIndex()` for this purpose.

---

## 📦 Binary Format Specification

### embeddings.bin Format (Python → WASM)

```
Header:
  u32: num_papers
  u32: embedding_dim

For each paper:
  u32: num_tokens
  f32[num_tokens * embedding_dim]: embeddings (flat array)
```

### .fastplaid Format (WASM save/load)

```
Header:
  [4 bytes] Magic: "FPQZ"
  u32: Version (1)
  u32: embedding_dim
  u32: num_documents
  u32: num_clusters

IVF Centroids:
  f32[num_clusters * embedding_dim]: cluster centroids

IVF Mapping:
  For each cluster:
    u32: cluster_size
    u32[cluster_size]: document indices

Quantization Codec:
  u32: num_centroids (256)
  f32[num_centroids * embedding_dim]: k-means centroids

Quantized Documents:
  For each document:
    i64: document_id
    u32: num_tokens
    u32: centroid_codes_len
    u8[centroid_codes_len]: centroid codes
    u32: packed_residuals_len
    u8[packed_residuals_len]: 4-bit packed residuals
```

---

## 🎯 Comparison: Native vs WASM

| Feature | Native (`search/`) | WASM (`lib_wasm_quantized.rs`) |
|---------|-------------------|--------------------------------|
| **Purpose** | Production PLAID | Browser demo |
| **Dependencies** | Candle, PyTorch-like | None (pure Rust) |
| **Quantization** | Multi-stage PQ | Simple 4-bit |
| **IVF** | Full PLAID algorithm | Basic k-means |
| **Output** | `.plaid` files | `.fastplaid` OR on-the-fly |
| **Load Speed** | N/A (server-side) | 0.5s (precomputed) |
| **Size** | Larger, more accurate | 8x smaller |
| **CUDA** | ✅ Supported | ❌ Browser only |
| **Use Case** | Python/CLI/Server | GitHub Pages |

---

## 💡 When to Use What?

### Use Native Implementation When:
- Building Python bindings for server use
- Need CUDA acceleration
- Want full PLAID algorithm from paper
- Building large indexes offline (>10k papers)

### Use WASM Implementation When:
- Deploying to GitHub Pages
- Browser-based demo
- Need instant loading (<1s)
- <10k papers
- Want simple 4-bit + IVF

---

## 🚀 Deployment Checklist

### For GitHub Pages:

1. **Build embeddings:**
   ```bash
   python scripts/build_offline_wasm_index.py --papers data/papers_1000.json --output demo/data
   ```

2. **Pre-build .fastplaid index:**
   ```bash
   node scripts/build_fastplaid_index.js demo/data demo/data/index.fastplaid
   ```

3. **Deploy files:**
   ```
   demo/
   ├── index.html
   ├── pkg/                      # WASM files
   ├── data/
   │   ├── index.fastplaid      # 6.2 MB - Pre-built index
   │   └── papers_metadata.json # 1.6 MB - Paper info
   └── mxbai-integration.js
   ```

4. **Total size:** ~8 MB (under GitHub Pages limits!)

### Performance:
- **Load time:** <1 second
- **Search time:** 20-30ms per query
- **Memory:** ~10 MB in browser

---

## 📚 Example: Complete Workflow

```bash
# 1. Extract 1000 papers
python scripts/precompute_index.py  # Creates data/papers_1000.json

# 2. Compute embeddings
python scripts/build_offline_wasm_index.py \
    --papers data/papers_1000.json \
    --output demo/data \
    --model mixedbread-ai/mxbai-edge-colbert-v0-17m

# 3. Build .fastplaid index
node scripts/build_fastplaid_index.js \
    demo/data \
    demo/data/index.fastplaid

# 4. Test locally
cd demo && python serve.py

# 5. Deploy to GitHub Pages
git add demo/data/index.fastplaid demo/data/papers_metadata.json
git commit -m "Add precomputed index"
git push
```

---

## 🔍 Troubleshooting

### "Invalid index file: bad magic number"
- Index was not built correctly
- Rebuild with: `node scripts/build_fastplaid_index.js`

### "WASM table growth error"
- Browser ran out of memory
- Try: Refresh page, close tabs, use Chrome

### "Index too large for GitHub Pages"
- GitHub Pages has 100 MB per file limit
- Reduce number of papers or use CDN

### Slow loading in browser
- Use pre-built `.fastplaid` instead of `embeddings.bin`
- Enable gzip compression on server

---

## 🎓 Further Reading

- [FastPlaid Paper](https://arxiv.org/abs/your-paper) - Original PLAID algorithm
- [ColBERT Docs](https://github.com/stanford-futuredata/ColBERT) - ColBERT embeddings
- [WASM bindgen](https://rustwasm.github.io/wasm-bindgen/) - Rust WASM bindings

---

**Questions?** Open an issue or check the main [README.md](README.md)
