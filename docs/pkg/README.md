<div align="center">
  <h1>FastPlaid</h1>
</div>

<p align="center"><img width=500 src="https://github.com/lightonai/fast-plaid/blob/6184631dd9b9609efac8ce43e3e15be2efbb5355/docs/logo.png"/></p>

<div align="center">
    <a href="https://github.com/rust-lang/rust"><img src="https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="rust"></a>
    <a href="https://github.com/pyo3"><img src="https://img.shields.io/badge/PyO₃-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="PyO₃"></a>
</div>

&nbsp;

<div align="center">
    <b>FastPlaid</b> - High-Performance Multi-Vector Search with WASM Support
</div>

&nbsp;

## ⭐️ Overview

**FastPlaid** implements efficient multi-vector search for ColBERT-style models. Unlike traditional single-vector search, multi-vector approaches maintain token-level embeddings for fine-grained similarity matching.

**Key Features:**
- 🚀 **WASM Support** - Browser-native search with `mxbai-edge-colbert-v0-17m` (48-dim embeddings)
- ⚡ **4-bit Quantization + IVF** - 8x compression, 3-5x faster search
- 🔄 **Incremental Updates** - Add documents without full rebuild (NEW!)
- 🎯 **MaxSim Search** - Token-level late interaction for accurate retrieval
- 📦 **Pure Rust** - Fast, safe, and portable
- 🗂️ **Offline Index Building** - Pre-compute indexes for instant browser loading

## 🏗️ Architecture

FastPlaid has **two implementations** for different use cases:

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **Native Rust** (`search/`, `index/`) | Full PLAID with Product Quantization | Python bindings, CLI, server-side |
| **WASM** (`lib_wasm_quantized.rs`) | Lightweight 4-bit + IVF | Browser demos, GitHub Pages |

**Why two implementations?**
- Native uses Candle (PyTorch-like) tensors for full PLAID algorithm
- WASM uses pure Rust for browser compatibility (no Candle in WASM)
- Both share the same 4-bit quantization codec

📖 **See [OFFLINE_INDEX_GUIDE.md](OFFLINE_INDEX_GUIDE.md) for detailed architecture and workflows**

## 💻 Installation

### Python Package
```bash
pip install fast-plaid
```

**PyTorch Compatibility:**
| FastPlaid | PyTorch | Command |
|-----------|---------|---------|
| 1.2.4.280 | 2.8.0   | `pip install fast-plaid==1.2.4.280` |
| 1.2.4.271 | 2.7.1   | `pip install fast-plaid==1.2.4.271` |

### WASM Demo
```bash
cd docs
python3 serve.py
# Visit http://localhost:8000/
```

### Offline Index Building
```bash
# 1. Compute embeddings (Python)
python scripts/build_offline_wasm_index.py \
    --papers data/papers_1000.json \
    --output docs/data

# 2. Build .fastplaid index (Node.js + WASM)
node scripts/build_fastplaid_index.js \
    docs/data \
    docs/data/index.fastplaid

# 3. Deploy to browser
# index.fastplaid: 6.2 MB, loads in <1s
```

📖 **See [OFFLINE_INDEX_GUIDE.md](OFFLINE_INDEX_GUIDE.md) for complete workflows**

## 🎯 Quick Start

### Python API
```python
from fast_plaid import FastPlaid

# Initialize with ColBERT embeddings (48-dim token vectors)
index = FastPlaid(dim=48, nbits=4)  # 4-bit quantization

# Add documents (shape: [num_docs, max_tokens, 48])
index.add(doc_embeddings)

# Search (shape: [num_queries, query_tokens, 48])
scores = index.search(query_embeddings, k=10)
```

### WASM Browser Demo
```javascript
// Load model
const colbert = new ColBERT(
    modelWeights, dense1Weights, dense2Weights,
    tokenizer, config, stConfig,
    dense1Config, dense2Config, tokensConfig, 32
);

// Encode and search
const queryEmb = await colbert.encode({sentences: [query], is_query: true});
const results = await fastPlaid.search(queryEmb, 10);

// Incremental updates (NEW!)
const newDocEmb = await colbert.encode({sentences: [newDoc], is_query: false});
fastPlaid.update_index_incremental(newDocEmb, newDocInfo);
```

### Incremental Index Updates 🔄

FastPlaid now supports adding documents without rebuilding the entire index:

```javascript
// Create initial index
fastPlaid.load_documents_quantized(embeddings, docInfo, 256);

// Add new documents incrementally (8x faster than rebuild!)
fastPlaid.update_index_incremental(newEmbeddings, newDocInfo);

// Check statistics
const info = JSON.parse(fastPlaid.get_index_info());
console.log(`${info.num_documents} docs, ${info.pending_deltas} deltas`);

// Manual compaction (optional - auto-compacts at 10%)
fastPlaid.compact_index();
```

**Performance:**
- 8.3x faster for small batches (<100 docs)
- 2.7x faster for large batches (1000 docs)
- Auto-compaction when deltas exceed 10%
- <5% search overhead with deltas

📖 **See [INCREMENTAL_UPDATES.md](INCREMENTAL_UPDATES.md) for full API documentation**

## 🏗️ Architecture

### Multi-Vector Pipeline
```
Text → Tokenizer → ModernBERT (256d) → 1_Dense (512d) → 2_Dense (48d) → MaxSim Search
```

**Key Components:**
- **ModernBERT**: 17M parameter encoder
- **2_Dense Projection**: 256→512→48 dimensions (10.6x compression)
- **4-bit Quantization**: Additional 8x storage savings
- **MaxSim Scoring**: `score = Σ max(q_token · d_token)` per query token

### WASM Implementation
- **Model**: `mixedbread-ai/mxbai-edge-colbert-v0-17m`
- **Runtime**: Pure browser (no server)
- **Index Size**: ~2.7MB for 200 documents (48-dim, 4-bit)
- **Search Speed**: <50ms for 1000 documents

## 📊 Performance

### Index Size Comparison (200 documents)
| Method | Dimensions | Size | Compression |
|--------|-----------|------|-------------|
| Without 2_Dense | 512 | ~28.6 MB | 1x |
| With 2_Dense | 48 | ~2.7 MB | 10.6x |
| With 2_Dense + 4-bit | 48 | ~0.7 MB | 40x |

### Speed Benchmarks
- **Encoding**: ~50ms per document (WASM)
- **Search**: ~10ms for 100 docs, ~50ms for 1000 docs
- **Index Build**: ~500ms for 200 documents

## 🔧 WASM Build

The WASM package includes both FastPlaid indexing and ColBERT model inference:

```bash
# Build pylate-rs with 2_Dense support
cd pylate-rs
cargo build --lib --release --target wasm32-unknown-unknown \
    --no-default-features --features wasm

# Generate bindings
cargo install wasm-bindgen-cli --version 0.2.104
wasm-bindgen target/wasm32-unknown-unknown/release/pylate_rs.wasm \
    --out-dir pkg --target web

# Build FastPlaid WASM
cd rust
wasm-pack build --target web
```

**Output:**
- `pylate_rs_bg.wasm` (4.9MB) - ColBERT model + 2_Dense
- `fast_plaid_rust_bg.wasm` (114KB) - Indexing + search

## 🎨 Demo Features

### 1. Real-Time Search (`index.html`)
- Load `mxbai-edge-colbert-v0-17m` model
- Index 100 documents
- Interactive search with result highlighting
- Performance metrics display

### 2. Paper Search (`papers-demo.html`)
- Adjustable dataset size (10-1000 papers)
- Compare FastPlaid vs Direct MaxSim
- Index size visualization
- Search method toggle

### 3. Method Comparison
- **FastPlaid (Indexed)**: 4-bit quantized, ~7KB for 10 docs
- **Direct MaxSim**: Full precision, ~57KB for 10 docs
- **Speedup**: 2-5x faster with FastPlaid for 100+ documents

## 📁 Project Structure

```
fast-plaid/
├── rust/                  # Core Rust implementation
│   ├── lib.rs            # FastPlaid index
│   └── lib_wasm.rs       # WASM bindings
├── docs/                 # Browser demos (GitHub Pages)
│   ├── index.html        # Main demo
│   ├── build-index.html  # Index builder
│   ├── mxbai-integration.js  # ColBERT integration
│   └── node_modules/     # WASM modules
├── python/               # Python bindings
└── README.md            # This file
```

## 🔬 Technical Details

### 2_Dense Support
FastPlaid uses `pylate-rs` with full 2_Dense layer support for `mxbai-edge-colbert-v0-17m`:

**Architecture:**
1. **1_Dense**: 256 → 512 (expansion for representation)
2. **2_Dense**: 512 → 48 (compression for efficiency)

**Benefits:**
- Correct 48-dim output (not 512)
- 10.6x smaller indexes
- Matches official model specifications

### Quantization
4-bit quantization with centroids:
```rust
// Quantize to 4-bit (16 levels)
let quantized = embeddings.map(|x| ((x - min) / (max - min) * 15.0) as u8);

// Dequantize for search
let reconstructed = quantized.map(|q| min + (q as f32 / 15.0) * (max - min));
```

**Trade-offs:**
- Storage: 8x smaller
- Speed: ~10% faster (less memory bandwidth)
- Quality: <2% accuracy loss

## 🚀 Deployment

### GitHub Pages
The WASM demo can be deployed to GitHub Pages:

```bash
# Build for production
cd demo
./build-prod.sh

# Deploy
git add .
git commit -m "Update demo"
git push origin main
```

**Limitations:**
- Max file size: 100MB (GitHub Pages limit)
- Total site size: <1GB recommended
- Use 4-bit quantization for large datasets

### Local Development
```bash
cd demo
python3 serve.py  # http://localhost:8000/
```

## 🔗 Resources

- **Model**: [mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m)
- **pylate-rs**: [GitHub](https://github.com/lightonai/pylate-rs)
- **ColBERT**: [Paper](https://arxiv.org/abs/2004.12832)
- **Mixedbread Blog**: [ColBERT Models](https://www.mixedbread.com/blog/colbertus-maximus-mxbai-colbert-large-v1)

## 📝 Recent Updates

**v5.0 (2025-01-22):**
- ✅ Full 2_Dense support (48-dim embeddings)
- ✅ 4-bit quantization (8x compression)
- ✅ WASM demo with real ColBERT model
- ✅ Query expansion support
- ✅ Index size comparison UI
- ✅ Adjustable dataset size

**Previous:**
- SIMD optimizations
- Offline index caching
- PLAID implementation
- Python/Rust bindings

## 🤝 Contributing

Contributions welcome! Key areas:
- Performance optimizations
- Additional quantization methods
- More demo examples
- Documentation improvements

## 📄 License

MIT License - see LICENSE file for details

---

**Status**: Production Ready | **WASM**: 4.9MB | **Embedding Dim**: 48 | **Model**: mxbai-edge-colbert-v0-17m
