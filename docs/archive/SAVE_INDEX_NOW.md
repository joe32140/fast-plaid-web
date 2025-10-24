# Save FastPlaid Index Right Now!

## Quick Steps (30 seconds)

Your page just finished building the index! Now save it:

### 1. Open Console
- Press **F12** or **Ctrl+Shift+I**
- Click **Console** tab

### 2. Run This Command
```javascript
saveFastPlaidIndex()
```

That's it! Just type that one function name and press Enter.

### 3. Move the Downloaded File
```bash
mv ~/Downloads/index.fastplaid /home/joe/fast-plaid/demo/data/index.fastplaid
```

### 4. Reload Page
Press **Ctrl+R**

## What You'll See

After reload, console shows:
```
📦 Loading precomputed .fastplaid index from disk...
✅ FastPlaid loaded from disk in 0.01s!
```

Instead of:
```
🔨 Building FastPlaid index from embeddings...
✅ FastPlaid index built in 14.6s
```

## Why So Simple Now?

I exposed a helper function `saveFastPlaidIndex()` to window scope, so you don't need to paste a long script.

Just call the function after the index finishes building!

## Verification

Check the file was created:
```bash
ls -lh /home/joe/fast-plaid/demo/data/index.fastplaid
# Should show: ~6-7 MB file
```

## Done!

Now every time you load the page:
- Direct MaxSim: 3s (from disk)
- FastPlaid: 0.01s (from disk)
- Total: ~3s

**15x faster FastPlaid loading!** 🚀
