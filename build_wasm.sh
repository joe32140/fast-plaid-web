#!/bin/bash
# Build WASM with automatic table size fix

set -e

echo "🔨 Building WASM package..."
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir docs/pkg --release

echo "🔧 Fixing WASM table limits..."
python3 fix_wasm_table.py

echo "✅ Build complete!"
echo ""
echo "📦 Package info:"
ls -lh docs/pkg/*.wasm | awk '{print "  " $9 ": " $5}'
echo ""
echo "🧪 Test locally:"
echo "  python3 -m http.server 8000"
echo "  open http://localhost:8000/test_incremental_update.html"
