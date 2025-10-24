# FastPlaid Loading Explanation

## Current Situation

### What Happens Now

1. **Direct MaxSim** (3 seconds):
   - ✅ Loads `embeddings.bin` (50 MB) from disk
   - ✅ Pure loading, no processing
   - ✅ Ready to search immediately

2. **FastPlaid** (10-15 seconds):
   - ⏳ Waits for Direct MaxSim to finish loading
   - 🔧 Quantizes embeddings from float32 → 4-bit (256 centroids, k-means)
   - 🔧 Builds IVF clusters (k-means clustering)
   - ✅ Ready to search

**Total time**: ~15 seconds until FastPlaid is ready

### Why FastPlaid is Slower

FastPlaid currently **rebuilds the index on-the-fly** in WASM:
- Trains 256 quantization centroids via k-means
- Quantizes all tokens to 4-bit
- Builds 32 IVF clusters via k-means
- All happens in browser during page load

This is **NOT loading from disk** - it's **computing from scratch**!

## What You Were Expecting

You have a precomputed file `fastplaid_4bit.bin` (6.3 MB) that contains:
- Pre-trained quantization centroids
- Pre-quantized 4-bit embeddings
- Pre-built IVF clusters

**Expected behavior**: Load this file → instant FastPlaid ready (<1 second)

## Why It's Not Working

The WASM `FastPlaidQuantized` class has two methods:

1. ✅ `load_documents_quantized()` - Takes float32, quantizes, builds IVF
2. ❌ `load_index()` - **NOT IMPLEMENTED PROPERLY**

The `load_index()` method in [rust/lib_wasm_quantized.rs](rust/lib_wasm_quantized.rs:1) was implemented but doesn't work correctly with the `fastplaid_4bit.bin` format.

## The Problem

Two incompatible formats:

| Format | Created By | Used By | Purpose |
|--------|-----------|---------|---------|
| `fastplaid_4bit.bin` | Python script | JavaScript (PrecomputedIndexLoader) | Demo with JS-based search |
| `.fastplaid` (FPQZ) | WASM `save_index()` | WASM `load_index()` | WASM-based instant loading |

The Python-created `fastplaid_4bit.bin` **cannot** be loaded by WASM's `load_index()` - they use different formats!

## Solution Options

### Option 1: Keep Current Approach (Simplest)
✅ **Already implemented**

```
Direct MaxSim: Load from disk (3s)
FastPlaid: Build from Direct MaxSim embeddings (10-15s)
Total: ~15s
```

**Pros:**
- Simple, no format conversion needed
- Already working
- Only one source of truth (embeddings.bin)

**Cons:**
- FastPlaid takes 10-15s to build
- Re-computes quantization every time

### Option 2: Use WASM save_index() / load_index()
Create `.fastplaid` file using WASM's native format

**Steps:**
1. Run once to build and save:
```bash
node scripts/build_fastplaid_wasm_index.js \
    demo/data/fastplaid_4bit/embeddings.bin \
    demo/data/index.fastplaid
```

2. Update index.html to load:
```javascript
const indexBytes = await fetch('./data/index.fastplaid').arrayBuffer();
fastPlaidWasm.load_index(new Uint8Array(indexBytes));
// Instant loading!
```

**Pros:**
- FastPlaid loads in <1 second from disk
- Both indexes load from disk, ~3s total

**Cons:**
- Need to create new build script for WASM format
- Two formats to maintain (Python vs WASM)

### Option 3: Make WASM Support Python Format
Modify WASM to load `fastplaid_4bit.bin` format

**Changes needed:**
1. Add new method to [rust/lib_wasm_quantized.rs](rust/lib_wasm_quantized.rs:1):
```rust
#[wasm_bindgen]
pub fn load_from_python_format(&mut self, index_bytes: &[u8]) -> Result<(), JsValue>
```

2. Parse the Python format:
   - Read header (total_tokens, embedding_dim, num_papers, num_clusters)
   - Read min_vals, max_vals
   - Read doc_boundaries
   - Read centroids
   - Read cluster_labels
   - Read cluster mappings
   - Read packed 4-bit data

**Pros:**
- Use existing `fastplaid_4bit.bin` file
- Fast loading (~1s)

**Cons:**
- Significant Rust development work
- Need to maintain compatibility with Python format

## Recommendation

### For Demo (Current Use Case)
**Keep Option 1** (current implementation)

Why:
- Works reliably now
- 15 seconds is acceptable for demo
- Simpler to maintain
- User can search after 3s (with Direct MaxSim only)

### For Production (Future)
**Implement Option 2** (WASM native format)

Why:
- Both indexes load from disk (~3s total)
- Better user experience
- Cleaner separation (Python for preprocessing, WASM for runtime)
- Already have save_index() implemented

## What to Show in UI

Current implementation now shows:

```
Status messages:
⏳ Loading Direct MaxSim embeddings... (0-3s)
✅ Direct MaxSim embeddings loaded! (3s)
⏳ Waiting for embeddings to load... (3s)
🔧 Quantizing to 4-bit + building IVF clusters in WASM (10-15s)...
✅ FastPlaid index built in 10.5s (4-bit quantization + IVF clustering)

Console logs:
📥 Loading Direct MaxSim embeddings...
✅ Loaded embeddings in 2847ms (49.54 MB)
📥 Loading FastPlaid index...
⏳ Waiting for Direct MaxSim embeddings to load...
✅ Embeddings ready, now quantizing to 4-bit + building IVF index...
🔧 Auto-detected embedding_dim: 48
✅ Trained 256 centroids with 4 iterations
✅ Quantized 1000 documents (7.7x compression)
🎯 Building IVF index... (32 clusters)
✅ FastPlaid index built in 10.5s
```

This makes it **crystal clear** that:
1. Direct MaxSim loads from disk (fast)
2. FastPlaid builds from those embeddings (slow but honest)
3. User sees exactly what's happening

## Implementation Status

✅ **Completed**:
- Clear status messages in UI
- Honest about building vs loading
- Shows timing information
- Async loading with partial search support

❌ **Not Implemented**:
- True disk loading for FastPlaid (would need Option 2 or 3)
- `.fastplaid` file creation script
- WASM load_index() for instant loading

## Next Steps (If You Want Fast Loading)

If you want FastPlaid to load from disk in <1 second:

1. **Create WASM index builder script**:
```javascript
// scripts/build_wasm_fastplaid_index.js
import init, { FastPlaidQuantized } from './demo/pkg/fast_plaid_rust.js';

// Load embeddings
// Build WASM index
// Save using save_index()
// Write to demo/data/index.fastplaid
```

2. **Update index.html**:
```javascript
// Try WASM native format first
const indexResponse = await fetch('./data/index.fastplaid');
if (indexResponse.ok) {
    const indexBytes = await indexResponse.arrayBuffer();
    fastPlaidWasm.load_index(new Uint8Array(indexBytes));
    console.log('✅ FastPlaid loaded from disk in <1s!');
    return;
}

// Fallback to current approach
// ...build from embeddings...
```

3. **Build the index once**:
```bash
node scripts/build_wasm_fastplaid_index.js
# Creates demo/data/index.fastplaid (~6 MB)
```

4. **Deploy both files**:
- `embeddings.bin` (50 MB) - for Direct MaxSim
- `index.fastplaid` (6 MB) - for FastPlaid
- Total: 56 MB, both load from disk in ~3s

## Summary

**Current behavior**: Direct MaxSim loads from disk (3s), FastPlaid builds in WASM (10-15s)

**Why**: WASM doesn't have code to load the Python-created `fastplaid_4bit.bin` format

**To fix**: Need to either:
- Create WASM-native `.fastplaid` files and use `load_index()` (Option 2)
- Teach WASM to read Python format (Option 3)

**For now**: UI is honest about what's happening, and users can search after 3s with Direct MaxSim
