# Quick: Build FastPlaid Index Now

## The Problem

You're seeing this console output:
```
🔍 Checking for precomputed index.fastplaid...
❌ GET http://localhost:8000/data/index.fastplaid 404 (File not found)
🔨 Building FastPlaid index from embeddings...
✅ FastPlaid index built in 14.1s
```

**Why**: `index.fastplaid` doesn't exist yet, so it builds from scratch (slow).

## Solution: Build It Once (2 Options)

### Option 1: Use Browser (Easiest - Recommended)

1. **Your server is already running at http://localhost:8000**

2. **Open the builder page**:
   ```
   http://localhost:8000/build-index.html
   ```

3. **Click "Build Index" button**
   - Wait 10-15 seconds
   - Console will show progress
   - File will download automatically as `index.fastplaid`

4. **Move the downloaded file**:
   ```bash
   mv ~/Downloads/index.fastplaid /home/joe/fast-plaid/demo/data/index.fastplaid
   ```

5. **Reload the main page**:
   ```
   http://localhost:8000/index.html
   ```

6. **Check console - should see**:
   ```
   🔍 Checking for precomputed index.fastplaid...
   📦 Loading precomputed .fastplaid index from disk...
   ✅ FastPlaid loaded from disk in 0.8s!
   ```

### Option 2: Use existing save_index() in browser console

If build-index.html doesn't work, you can save the index after it builds:

1. **Wait for current page to finish building** (you already did this)

2. **Open browser console** and run:
   ```javascript
   // Save the built index
   const indexBytes = fastPlaidWasm.save_index();
   const blob = new Blob([indexBytes], { type: 'application/octet-stream' });
   const url = URL.createObjectURL(blob);
   const a = document.createElement('a');
   a.href = url;
   a.download = 'index.fastplaid';
   document.body.appendChild(a);
   a.click();
   document.body.removeChild(a);
   URL.revokeObjectURL(url);
   ```

3. **Move the downloaded file**:
   ```bash
   mv ~/Downloads/index.fastplaid /home/joe/fast-plaid/demo/data/index.fastplaid
   ```

4. **Reload the page** - should load instantly!

## Expected Result

### Before (Current - Slow):
```
Direct MaxSim: 3s (load from disk)
FastPlaid: 14s (build from scratch)
Total: ~14s
```

### After (Fast!):
```
Direct MaxSim: 3s (load from disk)
FastPlaid: <1s (load from disk)
Total: ~3s
```

**5x faster overall! 🚀**

## Verify It Worked

Check the console output after reload:

✅ **Success** - Should see:
```
📦 Loading precomputed .fastplaid index from disk...
✅ FastPlaid loaded from disk in 0.85s!
```

❌ **Still building** - Still see:
```
🔨 Building FastPlaid index from embeddings...
```
→ File not in right location or named incorrectly

## File Locations

```
/home/joe/fast-plaid/demo/data/
├── embeddings.bin              (49.5 MB) - Direct MaxSim ✅
├── index.fastplaid             (6-7 MB) - FastPlaid ⬅️ ADD THIS!
├── papers_metadata.json        (1.6 MB) - Metadata ✅
└── fastplaid_4bit/             (directory - not used if index.fastplaid exists)
```

## Troubleshooting

### "Build Index" button does nothing
- Check browser console for errors
- Make sure server is running
- Try Option 2 instead

### Still shows 404 after moving file
- Verify file exists: `ls -lh /home/joe/fast-plaid/demo/data/index.fastplaid`
- Check file size: should be 6-7 MB
- Clear browser cache (Ctrl+Shift+R)

### File is too large (50+ MB)
- The Python script I created makes uncompressed format
- Use the browser builder instead (Option 1)
- Browser builder creates proper 6-7 MB compressed format

## Quick Commands

```bash
# Check if index exists
ls -lh /home/joe/fast-plaid/demo/data/index.fastplaid

# If file is in Downloads
mv ~/Downloads/index.fastplaid /home/joe/fast-plaid/demo/data/index.fastplaid

# Verify it's there
ls -lh /home/joe/fast-plaid/demo/data/index.fastplaid

# Should show: ~6-7 MB file
```

## Why This Matters

**Without index.fastplaid**:
- Every page load builds index from scratch
- Takes 14 seconds
- Uses CPU to quantize and cluster
- Wastes time every single time

**With index.fastplaid**:
- Loads precomputed index from disk
- Takes <1 second
- No CPU work needed
- Fast forever! ⚡

## Next Steps

1. ✅ Build index using browser builder
2. ✅ Move file to demo/data/index.fastplaid
3. ✅ Reload page - enjoy fast loading!
4. 🚀 Deploy to GitHub Pages with both files
