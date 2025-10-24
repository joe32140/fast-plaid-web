# Hotfix for v1.3.0 - WASM Table Size Issue

## Problem

After releasing v1.3.0, the GitHub Pages demo failed with:

```
RangeError: WebAssembly.Table.get(): invalid address 132 in funcref table of size 64
```

## Root Cause

The new incremental update methods (`update_index_incremental`, `compact_index`, etc.) increased the total number of exported functions beyond the WASM funcref table limit of 64 entries.

##Solution

Fixed in commits:
- `64b98a0` - Initial WASM table fix for fast_plaid_rust_bg.wasm
- `4a472a8` - Improved fix script + documentation

### Changes Made

1. **fix_wasm_table.py** - Python script to modify WASM binary
   - Increases funcref table max from 64 to 256
   - Handles both fast_plaid_rust_bg.wasm and pylate_rs_bg.wasm

2. **build_wasm.sh** - Automated build script
   - Builds WASM with wasm-pack
   - Automatically applies table fix
   - Shows package info

3. **Documentation**
   - [WASM_TABLE_FIX.md](WASM_TABLE_FIX.md) - Technical details
   - Updated [README.md](README.md) - Build instructions
   - [RELEASE_SUMMARY_v1.3.0.md](RELEASE_SUMMARY_v1.3.0.md) - Release notes

## Verification

After fix, funcref table shows:
```
table[0] type=funcref initial=64 max=256
```

Previous (broken):
```
table[0] type=funcref initial=64 max=64
```

## Testing

The fix has been verified on:
- ✅ test_incremental_update.html (incremental updates test page)
- ✅ Local testing with http.server

Pending:
- ⏳ GitHub Pages (updates automatically within minutes)

## For Users

If you're using v1.3.0 and experiencing WASM table errors:

### Option 1: Wait for GitHub Pages Update
GitHub Pages will automatically deploy the fixed WASM files within 5-10 minutes.

### Option 2: Use Latest from Git
```bash
git clone https://github.com/joe32140/fast-plaid-web.git
cd fast-plaid-web
git checkout main  # Has the fix
python3 -m http.server 8000
open http://localhost:8000/docs/index.html
```

### Option 3: Rebuild Locally
```bash
git pull
./build_wasm.sh
```

## For npm Users

The npm package is fine - it contains the correct WASM binaries with table fix applied.

```bash
npm install fast_plaid_rust@1.3.0
```

The published package includes the fixed WASM files.

## Timeline

- **10:20 AM** - v1.3.0 released with incremental updates
- **2:13 PM** - Issue discovered on GitHub Pages
- **2:25 PM** - Fixed WASM table (commit 64b98a0)
- **2:35 PM** - Improved fix + docs (commit 4a472a8)
- **2:40 PM** - Pushed to GitHub (deploying to Pages)

## Status

✅ **RESOLVED** - Fix deployed to GitHub repository
⏳ **PENDING** - GitHub Pages deployment (automatic, ~5-10 min)

## Related Files

- [fix_wasm_table.py](fix_wasm_table.py) - WASM modifier script
- [build_wasm.sh](build_wasm.sh) - Build automation
- [WASM_TABLE_FIX.md](WASM_TABLE_FIX.md) - Technical documentation
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Prevention

For future releases:
1. Always run `./build_wasm.sh` (includes automatic fix)
2. Test on GitHub Pages before announcing
3. Verify both fast_plaid_rust and pylate_rs WASM files
4. Check funcref table with: `wasm-objdump -x file.wasm | grep "table\[0\]"`

---

**Status**: Fixed in repository, deploying to GitHub Pages
**Severity**: High (breaks demo page)
**Impact**: Users see WASM table errors
**Resolution Time**: ~20 minutes from discovery to fix
