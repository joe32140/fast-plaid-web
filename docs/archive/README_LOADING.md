# Loading Behavior: Direct MaxSim vs FastPlaid

## Quick Summary

**Q: Why does Direct MaxSim load in 3 seconds but FastPlaid takes 15 seconds?**

**A**:
- **Direct MaxSim** (3s): Loads precomputed float32 embeddings from `embeddings.bin` (49.5 MB)
- **FastPlaid** (15s): **Builds the index from scratch** in WASM by quantizing those embeddings

**FastPlaid is NOT loading from disk - it's computing!**

## Visual Explanation

```
┌─────────────────────────────────────────┐
│  Direct MaxSim (3 seconds)              │
├─────────────────────────────────────────┤
│  1. Fetch embeddings.bin (49.5 MB)     │
│  2. Parse binary format                 │
│  3. ✅ Ready to search                   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  FastPlaid (15 seconds)                 │
├─────────────────────────────────────────┤
│  1. Wait for Direct MaxSim (3s)         │
│  2. Train 256 quantization centroids    │
│  3. Quantize all tokens to 4-bit        │
│  4. Build 32 IVF clusters (k-means)     │
│  5. ✅ Ready to search                   │
└─────────────────────────────────────────┘
```

## What Files Exist

```
demo/data/
├── embeddings.bin (49.5 MB)              ✅ Used by Direct MaxSim
├── fastplaid_4bit/
│   ├── embeddings.bin (49.5 MB)          ✅ Used by Direct MaxSim
│   ├── fastplaid_4bit.bin (6.3 MB)       ❌ NOT used by WASM
│   ├── papers_metadata.json (1.6 MB)     ✅ Used for display
│   └── ...
```

**The problem**: `fastplaid_4bit.bin` exists but WASM can't load it!

## Why Can't WASM Load It?

The `fastplaid_4bit.bin` file was created by a Python script for JavaScript-based search. WASM expects a different format (`.fastplaid` with magic number "FPQZ").

**Two incompatible formats**:
1. Python format: `fastplaid_4bit.bin` (for JS demo)
2. WASM format: `.fastplaid` with FPQZ header (for WASM load_index())

## Current UI Messages

The UI now clearly shows what's happening:

### Loading Phase
```
Status: ⏳ Waiting for embeddings to load...
Direct MaxSim box: ⏳ Loading...
FastPlaid box: ⏳ Loading...
```

### Building Phase
```
Status: 🔧 Quantizing to 4-bit + building IVF clusters in WASM (10-15s)...
Direct MaxSim box: 49.5 MB
FastPlaid box: 🔧 Building...
```

### Ready
```
Status: ✅ All systems ready! Try: "transformer attention mechanisms"
Direct MaxSim box: 49.5 MB
FastPlaid box: 6.2 MB
Compression: 8.0x
```

## User Experience

1. **Page loads** (0s)
   - Shows loading indicators

2. **Direct MaxSim ready** (3s)
   - ✅ Search button enables
   - ✅ User can search with Direct MaxSim only
   - FastPlaid shows "🔧 Building..."

3. **FastPlaid ready** (15s)
   - ✅ Both methods available
   - ✅ Shows side-by-side comparison
   - ✅ Shows speedup statistics

## How to Make FastPlaid Load Faster

If you want FastPlaid to load from disk in <1 second (like Direct MaxSim), you need to:

### Option 1: Create WASM-Native Index (Recommended)

1. **Build script to create `.fastplaid` file**:
```javascript
// scripts/build_wasm_index.js
import init, { FastPlaidQuantized } from '../demo/pkg/fast_plaid_rust.js';

// Load embeddings
const embeddings = /* load from embeddings.bin */;

// Build in WASM
await init();
const wasm = new FastPlaidQuantized();
await wasm.load_documents_quantized(embeddings, docInfo, 256);

// Save to disk
const indexBytes = wasm.save_index();
fs.writeFileSync('demo/data/index.fastplaid', indexBytes);
```

2. **Update HTML to use it**:
```javascript
// Load from disk instead of building
const response = await fetch('./data/index.fastplaid');
const indexBytes = await response.arrayBuffer();
fastPlaidWasm.load_index(new Uint8Array(indexBytes));
// ✅ Instant loading!
```

3. **Result**: FastPlaid loads in <1s from disk

### Option 2: Keep Current (Simplest)

- Works now without changes
- User can search after 3s (with Direct MaxSim)
- Both methods ready after 15s
- Honest about what's happening

## Console Output You'll See

### Current Behavior
```
📂 Loading papers metadata...
✅ Loaded 1000 papers metadata

📥 Loading Direct MaxSim embeddings...
📥 Loading FastPlaid index...
🤖 Loading ColBERT model...

✅ WASM initialized
✅ Direct MaxSim embeddings loaded!

⏳ Waiting for Direct MaxSim embeddings to load...
✅ Embeddings ready, now quantizing to 4-bit + building IVF index...

📥 Loading and quantizing documents...
🔧 Auto-detected embedding_dim: 48
✅ Trained 256 centroids with 4 iterations
✅ Quantized 1000 documents (7.7x compression)
🎯 Building IVF index... (32 clusters)

✅ FastPlaid index built in 10.5s (4-bit quantization + IVF clustering)
✅ ColBERT model loaded!
```

### With Disk Loading (Future)
```
📂 Loading papers metadata...
✅ Loaded 1000 papers metadata

📥 Loading Direct MaxSim embeddings...
📥 Loading FastPlaid index from disk...
🤖 Loading ColBERT model...

✅ Direct MaxSim embeddings loaded! (3.2s)
✅ FastPlaid index loaded from disk! (0.8s)
✅ ColBERT model loaded! (2.1s)

✅ All systems ready!
```

## FAQ

**Q: Can we just use the existing `fastplaid_4bit.bin`?**

A: Not directly. WASM would need code to parse that format. Currently it only knows how to:
- Build from float32 embeddings (current approach)
- Load from WASM-native `.fastplaid` format (not yet created)

**Q: How much work to implement disk loading?**

A: About 2-3 hours:
- Write Node.js script to build `.fastplaid` file (~1 hour)
- Update HTML to load it instead of building (~30 minutes)
- Test and debug (~1 hour)

**Q: Is the current behavior acceptable?**

A: For a demo, yes! Users can:
- Start searching after 3 seconds (Direct MaxSim)
- See full comparison after 15 seconds
- Understand what's happening (clear UI messages)

**Q: Why does Direct MaxSim load faster?**

A: It's pure file loading with no processing:
- Read binary file from disk
- Parse header and tokens
- Done!

FastPlaid does heavy computation:
- K-means clustering for quantization (256 centroids)
- Quantize every token to 4-bit
- K-means clustering for IVF (32 clusters)
- Build search structures

## Recommendation

**For GitHub Pages demo**: Keep current implementation
- Works reliably
- Clear about what's happening
- User can start searching quickly

**For production app**: Implement disk loading
- 5x faster total load time
- Better user experience
- Worth the development time

## Related Documentation

- [FASTPLAID_LOADING_EXPLANATION.md](FASTPLAID_LOADING_EXPLANATION.md) - Detailed technical explanation
- [LOADING_FIX_SUMMARY.md](LOADING_FIX_SUMMARY.md) - Summary of changes made
- [ASYNC_LOADING_IMPLEMENTATION.md](ASYNC_LOADING_IMPLEMENTATION.md) - Async loading architecture
