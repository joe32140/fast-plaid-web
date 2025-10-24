#!/bin/bash
# Build WASM with automatic table size fix

set -e

echo "🔨 Building WASM package..."
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir docs/pkg --release

echo "🔧 Fixing WASM table limits using WABT tools..."
echo "   fast_plaid_rust: 64 → 256"
wasm2wat docs/pkg/fast_plaid_rust_bg.wasm 2>&1 | \
  sed 's/(table (;0;) 64 64 funcref)/(table (;0;) 64 256 funcref)/g' | \
  wat2wasm -o docs/pkg/fast_plaid_rust_bg.wasm -

echo "✅ Build complete!"
echo ""
echo "📦 Package info:"
ls -lh docs/pkg/*.wasm | awk '{print "  " $9 ": " $5}'
echo ""
echo "🧪 Test locally:"
echo "  python3 -m http.server 8000"
echo "  open http://localhost:8000/test_incremental_update.html"
