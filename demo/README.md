# FastPlaid WASM Demo

Browser-based multi-vector search demo using `mxbai-edge-colbert-v0-17m` model.

## Quick Start

```bash
python3 serve.py
# Visit http://localhost:8000/
```

## Demos

### 1. Main Demo (`index.html`)
- Load ColBERT model (17M params, 48-dim embeddings)
- Index 100 pre-defined documents
- Real-time semantic search
- Performance metrics

### 2. Papers Demo (`papers-demo.html`)
- Adjustable dataset (10-1000 papers)
- Compare FastPlaid vs Direct MaxSim
- Index size comparison
- Dynamic generation

## Architecture

```
Text → Tokenizer → ModernBERT → 1_Dense → 2_Dense → 48-dim embeddings → MaxSim Search
```

**Model Files:**
- `node_modules/pylate-rs/pylate_rs_bg.wasm` (4.9MB) - ColBERT + 2_Dense support
- `pkg/fast_plaid_rust_bg.wasm` (114KB) - FastPlaid indexing

## Performance

| Dataset | FastPlaid (4-bit) | Direct MaxSim | Speedup |
|---------|------------------|---------------|---------|
| 10 docs | 7 KB | 57 KB | 8x smaller |
| 100 docs | 70 KB | 570 KB | 8x smaller |
| 200 docs | 0.7 MB | 2.7 MB | 3.8x smaller |

**Search Speed:**
- 100 docs: ~10ms (FastPlaid) vs ~15ms (Direct)
- 1000 docs: ~50ms (FastPlaid) vs ~150ms (Direct)

## Technical Details

### 2_Dense Support
The demo uses full 2_Dense layer support:
- **1_Dense**: 256 → 512 (expansion)
- **2_Dense**: 512 → 48 (compression)
- **Result**: 10.6x smaller indexes

### Quantization
4-bit quantization for additional 8x compression with <2% quality loss.

### Files
- `index.html` - Main demo UI
- `papers-demo.html` - Paper search demo
- `mxbai-integration.js` - ColBERT model integration
- `paper-abstracts-loader.js` - Paper dataset + search logic
- `serve.py` - Local dev server

## Troubleshooting

**Model not loading?**
- Check browser console for errors
- Hard refresh (`Ctrl+Shift+R`)
- Clear site data in DevTools

**Slow performance?**
- Use 4-bit quantization (default)
- Reduce dataset size in papers demo
- Check CPU/memory usage

**WASM errors?**
- Ensure `node_modules/pylate-rs/` has latest WASM files
- Check WASM file size: 4.9MB (with 2_Dense)

## Development

### Rebuild WASM
```bash
# pylate-rs (ColBERT model)
cd ../../pylate-rs
cargo build --lib --release --target wasm32-unknown-unknown --no-default-features --features wasm
wasm-bindgen target/wasm32-unknown-unknown/release/pylate_rs.wasm --out-dir pkg --target web
cp pkg/* ../fast-plaid/demo/node_modules/pylate-rs/

# FastPlaid (indexing)
cd ../fast-plaid/rust
wasm-pack build --target web
cp pkg/* ../demo/pkg/
```

### Add Cache Buster
Add `?v=X` to imports in HTML to force reload:
```html
<script type="module" src="./mxbai-integration.js?v=8"></script>
```

## Resources

- **Model**: [mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m)
- **Parent Repo**: [../README.md](../README.md)
