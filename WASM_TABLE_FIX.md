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

We use a post-build script ([fix_wasm_table.py](fix_wasm_table.py)) to modify the WASM binary and increase the funcref table's maximum size from 64 to 256:

```
table[0] type=funcref initial=64 max=256
```

This allows the table to grow as needed when more functions are added.

## How It Works

The `fix_wasm_table.py` script:

1. Reads the WASM binary
2. Finds the Table section (section ID 4)
3. Locates the funcref table (type 0x70)
4. Replaces the max size LEB128 value: 64 → 256
5. Writes the modified WASM back

### Technical Details

WASM binary format for tables:
```
Section 4: Table
├── Section size (LEB128)
├── Number of tables (LEB128)
└── For each table:
    ├── Table type (0x70 = funcref, 0x6f = externref)
    ├── Limits type (0x00 = no max, 0x01 = has max)
    ├── Initial size (LEB128)
    └── Max size (LEB128, if limits type == 0x01)
```

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

# Step 2: Fix table limits
python3 fix_wasm_table.py

# Step 3: Verify
wasm-objdump -x docs/pkg/fast_plaid_rust_bg.wasm | grep "table\[0\]"
# Should show: table[0] type=funcref initial=64 max=256
```

## Why Not Fix at Compile Time?

We tried several approaches to fix this during compilation:

1. **RUSTFLAGS with link-arg** - Caused linker errors
   ```bash
   RUSTFLAGS="-C link-arg=--initial-table=128"
   # Error: rust-lld: error: unknown argument '--initial-table=128'
   ```

2. **Cargo.toml metadata** - No effect on table limits
   ```toml
   [package.metadata.wasm-pack.profile.release.wasm-bindgen]
   # These options don't control table size
   ```

3. **wasm-bindgen settings** - No table size control exposed

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
python3 fix_wasm_table.py
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

### Error: "Multiple Type sections"

**Cause**: WASM has unconventional structure (seen in wasm-objdump output).

**Impact**: None - the fix still works. This is a parsing warning from wasm-objdump, not a runtime error.

## Related Issues

- wasm-bindgen issue: https://github.com/rustwasm/wasm-bindgen/issues/2367
- Rust WASM book: https://rustwasm.github.io/docs/book/reference/code-size.html

## See Also

- [INCREMENTAL_UPDATES.md](INCREMENTAL_UPDATES.md) - Feature documentation
- [build_wasm.sh](build_wasm.sh) - Automated build script
- [fix_wasm_table.py](fix_wasm_table.py) - Table fix script
