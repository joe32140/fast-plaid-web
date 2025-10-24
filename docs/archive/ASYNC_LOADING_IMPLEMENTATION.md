# Async Loading Implementation

## Overview

Implemented independent async loading for Direct MaxSim and FastPlaid indexes with visual loading indicators and partial search capability.

## Key Features

### 1. Independent Loading State Management

```javascript
const loadingState = {
    directMaxSim: { loaded: false, loading: false, error: null },
    fastPlaid: { loaded: false, loading: false, error: null },
    colbert: { loaded: false, loading: false, error: null }
};
```

Each component tracks its own loading state independently.

### 2. Visual Loading Indicators

Size boxes show real-time loading status:
- **Before loading**: Shows "-"
- **During loading**: Shows "⏳ Loading..."
- **After loading**: Shows actual size (49.5 MB / 6.2 MB)
- **On error**: Shows "❌ Error"

```javascript
function updateLoadingUI() {
    // Updates Direct MaxSim size box
    if (loadingState.directMaxSim.loading) {
        directSize.innerHTML = '<span style="font-size:20px;">⏳</span> Loading...';
    } else if (loadingState.directMaxSim.loaded) {
        directSize.textContent = '49.5 MB';
    }
    // Similar for FastPlaid...
}
```

### 3. Parallel Loading

All three components load simultaneously:

```javascript
Promise.all([
    loadDirectMaxSimIndex(),
    loadFastPlaidIndex(),
    loadColBertModel()
]);
```

**Loading Sequence:**
1. Papers metadata (required for all)
2. Direct MaxSim embeddings (independent)
3. FastPlaid index (precomputed OR on-the-fly)
4. ColBERT model (independent)

### 4. Smart Search Button Enablement

Search button enables when:
- ColBERT model is loaded AND
- At least ONE index (Direct MaxSim OR FastPlaid) is ready

```javascript
if (colbert.loaded && (direct.loaded || fastplaid.loaded)) {
    document.getElementById('searchBtn').disabled = false;
}
```

### 5. Partial Search Support

Users can search anytime after at least one index + ColBERT are ready:

```javascript
async function performSearch() {
    // Check prerequisites
    if (!loadingState.colbert.loaded) {
        showStatus('⏳ ColBERT model is still loading, please wait...');
        return;
    }

    // Search with available indexes
    if (loadingState.directMaxSim.loaded) {
        directResults = searchDirectMaxSim(...);
    } else {
        console.log('⏳ Direct MaxSim: Not loaded yet');
    }

    if (loadingState.fastPlaid.loaded) {
        fastplaidResult = searchFastPlaid(...);
    } else {
        console.log('⏳ FastPlaid: Not loaded yet');
    }

    // Display whatever is available
    displayResults(directResults, fastplaidResult);
}
```

### 6. Graceful Degradation in UI

Result panels show loading messages for unready indexes:

```javascript
function displayResults(directResults, fastplaidResults) {
    if (directResults) {
        // Show actual results
    } else {
        directDiv.innerHTML =
            '<div class="status-message status-info">⏳ Direct MaxSim index is still loading. Results will appear when ready.</div>';
    }
    // Similar for FastPlaid...
}
```

## Loading Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ Page Load                                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Load Papers Metadata (1000 papers)                      │
│ ✅ Required for display                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
         ┌───────────────┴────────────────┐
         ↓                                  ↓
┌──────────────────────┐         ┌──────────────────────┐
│ loadDirectMaxSimIndex│         │ loadFastPlaidIndex() │
│ (Independent)        │         │ (Independent)        │
└──────────────────────┘         └──────────────────────┘
         ↓                                  ↓
   Loading: ⏳                         Loading: ⏳
   49.5 MB embeddings                 WASM + quantization
         ↓                                  ↓
   Loaded: 49.5 MB                   Loaded: 6.2 MB
         ↓                                  ↓
         └───────────────┬────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ loadColBertModel() (Independent)                        │
│ Loading: ⏳ Model initialization                         │
│ Loaded: ✅ Ready for query encoding                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Update Overall Status                                   │
│ • All loaded? Enable search button                      │
│ • Partial? Enable if ColBERT + 1 index ready            │
│ • Errors? Show warning, enable if possible              │
└─────────────────────────────────────────────────────────┘
```

## Status Messages

The UI shows different status messages during loading:

| Condition | Status Message |
|-----------|----------------|
| All loading | "⏳ Loading... Direct MaxSim... FastPlaid... ColBERT..." |
| Partial ready | "⚠️ Partial system ready. Direct MaxSim: ✅ FastPlaid: ✅" |
| All ready | "✅ All systems ready! Try: 'transformer attention mechanisms'" |
| Error | "❌ [Component] Error: [message]" with helpful hints |

## Search Scenarios

### Scenario 1: Both Indexes Ready
```
User searches → Both panels show results → Status shows speedup comparison
✅ FastPlaid: 8.0x smaller, 3.7x faster! (IVF: 6 clusters → 30% of papers searched)
```

### Scenario 2: Only Direct MaxSim Ready
```
User searches → Direct MaxSim shows results → FastPlaid shows "⏳ still loading"
✅ Direct MaxSim results shown (FastPlaid still loading)
```

### Scenario 3: Only FastPlaid Ready
```
User searches → FastPlaid shows results → Direct MaxSim shows "⏳ still loading"
✅ FastPlaid results shown (Direct MaxSim still loading)
```

### Scenario 4: Neither Index Ready (blocked)
```
User tries to search → Blocked with message
⏳ Both indexes are still loading, please wait...
```

## Error Handling

### WASM Memory Errors
```javascript
if (error.message.includes('Table.grow') || error.message.includes('Memory')) {
    helpText = '💡 This is a WASM memory issue. Try:<br>' +
               '• Refreshing the page (Ctrl+Shift+R)<br>' +
               '• Closing other tabs to free memory<br>' +
               '• Using a browser with more WASM support (Chrome/Edge)';
}
```

### Network Errors
```javascript
if (error.message.includes('fetch') || error.message.includes('network')) {
    helpText = '💡 Network issue. Check that the server is running.';
}
```

### Partial Failures
If one index fails but another succeeds, search is still enabled:
```javascript
// Enable search if at least one index and colbert are ready
if (colbert.loaded && (direct.loaded || fastplaid.loaded)) {
    document.getElementById('searchBtn').disabled = false;
}
```

## Implementation Files

### Modified Files
- **[demo/index.html](demo/index.html)**: Complete rewrite of loading logic

### Key Functions Added
1. `updateLoadingUI()` - Updates size boxes with loading state
2. `loadDirectMaxSimIndex()` - Async Direct MaxSim loader
3. `loadFastPlaidIndex()` - Async FastPlaid loader
4. `loadColBertModel()` - Async ColBERT loader
5. `updateOverallStatus()` - Determines search button state
6. Modified `performSearch()` - Handles partial loading
7. Modified `displayResults()` - Shows loading messages for unready indexes

## Testing

### Manual Test Steps
1. Open http://localhost:8000/index.html
2. Open DevTools Console (F12)
3. Observe loading sequence in console
4. Watch size boxes update independently
5. Try searching before all indexes load
6. Verify partial results appear correctly

### Expected Console Output
```
📂 Loading papers metadata...
✅ Loaded 1000 papers metadata
📥 Loading Direct MaxSim embeddings...
📥 Loading FastPlaid index...
🤖 Loading ColBERT model...
✅ WASM initialized
🔍 Checking for precomputed .fastplaid index...
⚠️  No precomputed index found, will build on-the-fly
🔨 Building FastPlaid index from embeddings (this may take 10-15s)...
✅ Direct MaxSim embeddings loaded!
✅ FastPlaid index loaded!
✅ ColBERT model loaded!
✅ All systems ready!
```

## Performance Characteristics

### With Precomputed .fastplaid Index
- **Direct MaxSim**: ~3 seconds (49 MB download)
- **FastPlaid**: <1 second (6 MB download, instant load)
- **Total**: ~4 seconds to fully ready
- **Search enabled**: As soon as first index + ColBERT ready (~3-4 seconds)

### Without Precomputed Index (On-the-fly)
- **Direct MaxSim**: ~3 seconds (49 MB download)
- **FastPlaid**: ~15 seconds (49 MB download + quantization + IVF building)
- **Total**: ~15 seconds to fully ready
- **Search enabled**: As soon as Direct MaxSim + ColBERT ready (~3-4 seconds)

## Benefits

1. **Faster Time-to-Interactive**: Users can search as soon as ONE index is ready
2. **Better UX**: Clear visual feedback on what's loading
3. **Resilience**: Partial failures don't block all functionality
4. **Flexibility**: Works with or without precomputed indexes
5. **Transparency**: Users see exactly what's happening in real-time

## Future Enhancements

1. **Progress Bars**: Show percentage for large downloads
2. **Retry Buttons**: Allow manual retry on failed components
3. **Prefetch**: Start loading next component before previous finishes
4. **Service Worker**: Cache indexes for instant subsequent loads
5. **Incremental Loading**: Load first N papers quickly, then rest in background
