# Build FastPlaid Index Locally - Final Instructions

## The Issue

The Python format converter I created doesn't work correctly - it creates an incompatible format that causes "index out of bounds" errors during search.

**Why**: The Python quantization format and WASM quantization format are fundamentally different in how they store residuals and codes.

## The Solution: Use Browser Console (Simple & Works!)

Since your browser already builds the index correctly in 14 seconds, just save it from the console!

### Step-by-Step (Takes 2 minutes)

1. **Your server is running at http://localhost:8000**

2. **Open**: http://localhost:8000/index.html

3. **Wait for FastPlaid to build** (you'll see):
   ```
   🔨 Building FastPlaid index from embeddings...
   ✅ FastPlaid index built in 14.1s
   ```

4. **Open browser console** (F12 or Ctrl+Shift+I)

5. **Paste this code** and press Enter:
   ```javascript
   (async () => {
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
       console.log('✅ Downloaded index.fastplaid! Save to demo/data/');
   })();
   ```

6. **Save the downloaded file**:
   ```bash
   mv ~/Downloads/index.fastplaid /home/joe/fast-plaid/demo/data/index.fastplaid
   ```

7. **Reload the page** (Ctrl+R)

### Expected Result

Console will show:
```
📦 Loading precomputed .fastplaid index from disk...
✅ FastPlaid loaded from disk in 0.01s!
```

And searching will work instantly!

## Alternative: Use build-index.html

1. **Open**: http://localhost:8000/build-index.html

2. **Click "Build Index"** button

3. **Wait** 10-15 seconds

4. **File downloads automatically** as `index.fastplaid`

5. **Move it**:
   ```bash
   mv ~/Downloads/index.fastplaid /home/joe/fast-plaid/demo/data/index.fastplaid
   ```

## Why Python Converter Didn't Work

The Python `fastplaid_4bit.bin` format:
- Uses different quantization codebook
- Packs residuals differently
- Has different cluster structure

WASM expects:
- Specific centroid codes layout
- Specific residual packing
- Must match the codec it trains

**Bottom line**: Let WASM build and save its own format.

## Why This is OK

Building in browser once:
- Takes 14 seconds
- Creates correct format
- Save it, use forever

You only do this ONCE:
1. Build in browser (14s)
2. Save from console (<1s)
3. Copy to demo/data/ (<1s)
4. Done! All future loads are <1s

## Automation (Optional)

If you really want automation, you'd need:

1. **Use wasm-pack in lib mode** to run WASM in Node without table issues
2. **Or use Puppeteer** (headless browser) to automate the browser build
3. **Or fix the Python converter** to match WASM's exact format

But honestly, doing it manually once is faster than automating! 😄

## Summary

**Easiest way to build index locally**:

1. Open index.html, wait for build to complete
2. Run console command to save
3. Move file to demo/data/
4. Done!

**Time**: 2 minutes of your time, 14 seconds of computer time

**Result**: FastPlaid loads in 0.01s forever! 🚀
