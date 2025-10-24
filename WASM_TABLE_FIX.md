# WASM Table Size Fix

## Problem

After adding incremental update methods in v1.3.0, the GitHub Pages demo started failing with:

```
RangeError: WebAssembly.Table.get(): invalid address 132 in funcref table of size 64
```

## Root Cause

The WASM module's **funcref table** was limited to a maximum size of 64 entries. When we added new exported functions (`update_index_incremental`, `compact_index`, etc.), the total number of functions exceeded this limit.

The table configuration was:
```
table[0] type=funcref initial=64 max=64
```

The `max=64` limit prevented the table from growing beyond 64 entries, causing the error when trying to access function #132.

## Solution

We use WABT tools (WebAssembly Binary Toolkit) to convert the WASM to text format, modify the table limits, and convert back:

```
table[0] type=funcref initial=64 max=256
```

This allows the table to grow as needed when more functions are added.

## How It Works

The fix uses `wasm2wat` + `sed` + `wat2wasm`:

1. Convert WASM binary to WAT (WebAssembly Text format)
2. Use `sed` to replace table limit: `64 64 funcref` → `64 256 funcref`
3. Convert WAT back to WASM binary

```bash
wasm2wat docs/pkg/fast_plaid_rust_bg.wasm 2>&1 | \
  sed 's/(table (;0;) 64 64 funcref)/(table (;0;) 64 256 funcref)/g' | \
  wat2wasm -o docs/pkg/fast_plaid_rust_bg.wasm -
```

### Why WABT Tools?

**Python binary modification approach FAILED** because:
- Changing LEB128 from 1 byte (64) to 2 bytes (256) shifts all subsequent data
- This corrupts the WASM structure and breaks validation
- Error: "reached end while decoding" or "multiple Type sections"

WABT tools work because:
- They properly parse and reconstruct the entire WASM structure
- All offsets and sizes are recalculated correctly
- No corruption or validation errors

## Build Process

### Automatic (Recommended)

Use the provided build script:

```bash
./build_wasm.sh
```

This automatically:
1. Builds WASM with `wasm-pack`
2. Fixes the table limits
3. Shows package info

### Manual

```bash
# Step 1: Build WASM
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir docs/pkg --release

# Step 2: Fix table limits using WABT tools
wasm2wat docs/pkg/fast_plaid_rust_bg.wasm 2>&1 | \
  sed 's/(table (;0;) 64 64 funcref)/(table (;0;) 64 256 funcref)/g' | \
  wat2wasm -o docs/pkg/fast_plaid_rust_bg.wasm -

# Step 3: Verify
wasm-objdump -x docs/pkg/fast_plaid_rust_bg.wasm -j Table
# Should show: table[0] type=funcref initial=64 max=256
```

## Why Not Fix at Compile Time?

We tried several approaches to fix this during compilation:

1. **RUSTFLAGS with link-arg** - Caused linker errors
   ```bash
   RUSTFLAGS="-C link-arg=--initial-table=128"
   # Error: rust-lld: error: unknown argument '--initial-table=128'
   ```

2. **build.rs with linker flags** - Same linker errors
   ```rust
   println!("cargo:rustc-link-arg=--initial-table=256");
   # Error: rust-lld: error: unknown argument '--initial-table=256'
   ```

3. **.cargo/config.toml** - Same linker errors
   ```toml
   [target.wasm32-unknown-unknown]
   rustflags = ["-C", "link-arg=--initial-table=256"]
   ```

4. **Cargo.toml metadata** - No effect on table limits

5. **wasm-bindgen settings** - No table size control exposed

The post-build modification is the most reliable solution.

## Verification

After fixing, verify the table was updated:

```bash
wasm-objdump -x docs/pkg/fast_plaid_rust_bg.wasm | grep -A 2 "table\[0\]"
```

Expected output:
```
 - table[0] type=funcref initial=64 max=256
 - table[1] type=externref initial=128
```

## Testing

Test the fixed WASM:

```bash
# Start local server
python3 -m http.server 8000

# Open test page
open http://localhost:8000/test_incremental_update.html

# Or test main demo
open http://localhost:8000/docs/index.html
```

If the fix worked, you should see:
- ✅ No "invalid address" errors
- ✅ Search works correctly
- ✅ All methods accessible

## Alternative Solutions (Not Used)

### 1. Reduce Function Count
- Remove unused methods
- Combine similar functions
- **Downside**: Loses functionality

### 2. Use Dynamic Import
- Load methods on-demand via JS
- **Downside**: Complex, slower

### 3. Split WASM Modules
- Separate incremental updates into second module
- **Downside**: More files, coordination overhead

### 4. Use Larger Initial Table
- Set initial=256 instead of 64
- **Downside**: Same compile-time limitations

## Future Considerations

### wasm-tools Support

If `wasm-tools` becomes available, use it instead:

```bash
# Install wasm-tools
cargo install wasm-tools

# Modify table limits
wasm-tools parse docs/pkg/fast_plaid_rust_bg.wasm \
  | wasm-tools print \
  | sed 's/max 0x40/max 0x100/' \
  | wasm-tools parse -o docs/pkg/fast_plaid_rust_bg.wasm
```

### LLVM/Rust Improvements

Future Rust/LLVM versions may support:
- `#[link_section]` attributes for table control
- Better `wasm-ld` flags
- Cargo.toml table configuration

Until then, the post-build fix is necessary.

## Troubleshooting

### Error: "invalid address X in funcref table of size 64"

**Cause**: Table fix wasn't applied or didn't work.

**Solution**:
```bash
wasm2wat docs/pkg/fast_plaid_rust_bg.wasm 2>&1 | \
  sed 's/(table (;0;) 64 64 funcref)/(table (;0;) 64 256 funcref)/g' | \
  wat2wasm -o docs/pkg/fast_plaid_rust_bg.wasm -

git add docs/pkg/fast_plaid_rust_bg.wasm
git commit -m "fix: Apply WASM table size fix"
git push
```

### Error: "Table section not found"

**Cause**: WASM file format changed or corrupted.

**Solution**:
1. Rebuild WASM: `./build_wasm.sh`
2. Verify magic number: `hexdump -C docs/pkg/fast_plaid_rust_bg.wasm | head -1`
   Should start with: `00 61 73 6d` (`\0asm`)

### Error: "Multiple Type sections" or "reached end while decoding"

**Cause**: WASM file was corrupted by Python binary modification that changed LEB128 byte sizes.

**Solution**: Rebuild cleanly from source:
```bash
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir docs/pkg --release
wasm2wat docs/pkg/fast_plaid_rust_bg.wasm 2>&1 | \
  sed 's/(table (;0;) 64 64 funcref)/(table (;0;) 64 256 funcref)/g' | \
  wat2wasm -o docs/pkg/fast_plaid_rust_bg.wasm -
```

### Important: pylate_rs is a Pre-built Dependency

**DO NOT modify pylate_rs_bg.wasm** - it's a pre-built WASM module for the ColBERT model, not part of our build process. Only modify fast_plaid_rust_bg.wasm.

If pylate_rs gets corrupted, restore from a working commit:
```bash
git show 9a0aa9e:docs/pkg/pylate_rs_bg.wasm > docs/pkg/pylate_rs_bg.wasm
git show 9a0aa9e:docs/pkg/pylate_rs.js > docs/pkg/pylate_rs.js
```

## Related Issues

- wasm-bindgen issue: https://github.com/rustwasm/wasm-bindgen/issues/2367
- Rust WASM book: https://rustwasm.github.io/docs/book/reference/code-size.html

## Key Learnings

1. **Python binary modification fails** - Changing LEB128 sizes corrupts WASM
2. **WABT tools work reliably** - Proper parsing and reconstruction
3. **Only modify our WASM** - Leave pre-built dependencies (pylate_rs) alone
4. **Test before committing** - Load the page locally to verify it works

## See Also

- [INCREMENTAL_UPDATES.md](INCREMENTAL_UPDATES.md) - Feature documentation
- [build_wasm.sh](build_wasm.sh) - Automated build script with WABT tools
