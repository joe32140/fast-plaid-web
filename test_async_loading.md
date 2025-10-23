# Async Loading Test Plan

## Test URL
http://localhost:8000/index.html

## Expected Behavior

### Initial Load (Page Load)
1. ✅ Shows "Initializing..." status
2. ✅ Shows paper count (1000)
3. ✅ Size boxes show "⏳ Loading..." for both Direct MaxSim and FastPlaid
4. ✅ Compression shows "-" (not calculated yet)

### During Loading (Independent)
1. ✅ Direct MaxSim size box updates to "⏳ Loading..."
2. ✅ FastPlaid size box updates to "⏳ Loading..."
3. ✅ Status message shows which components are loading
4. ✅ Search button remains disabled

### When One Index Finishes First
1. ✅ Size box updates to actual size (49.5 MB or 6.2 MB)
2. ✅ Status message updates to show partial ready state
3. ✅ If ColBERT is also ready, search button enables

### When Both Indexes Loaded
1. ✅ Both size boxes show actual sizes
2. ✅ Compression ratio shows "8.0x"
3. ✅ Status shows "All systems ready!"
4. ✅ Search button is enabled

### Search with Partial Loading
**Test Case 1: Only Direct MaxSim loaded**
- User can submit query
- Only Direct MaxSim panel shows results
- FastPlaid panel shows "⏳ Index is still loading..."

**Test Case 2: Only FastPlaid loaded**
- User can submit query
- Only FastPlaid panel shows results
- Direct MaxSim panel shows "⏳ Index is still loading..."

**Test Case 3: Both loaded**
- User can submit query
- Both panels show results
- Status shows speedup comparison

### Error Handling
1. ✅ If Direct MaxSim fails: Size box shows "❌ Error"
2. ✅ If FastPlaid fails: Size box shows "❌ Error", helpful WASM message
3. ✅ If ColBERT fails: Shows error message
4. ✅ Search button enables if at least one index + ColBERT loaded

## Manual Test Steps

1. Open http://localhost:8000/index.html in browser
2. Open Developer Console (F12)
3. Watch console logs for loading sequence
4. Observe size boxes updating independently
5. Try searching before all indexes load (should show partial results)
6. Try searching after all indexes load (should show comparison)

## Console Output to Look For

```
📂 Loading papers metadata...
✅ Loaded 1000 papers metadata
📥 Loading Direct MaxSim embeddings...
📥 Loading FastPlaid index...
🤖 Loading ColBERT model...
✅ WASM initialized
🔍 Checking for precomputed .fastplaid index...
📦 Loading precomputed .fastplaid index...
✅ Loaded precomputed .fastplaid index instantly!
✅ FastPlaid index loaded!
✅ Direct MaxSim embeddings loaded!
✅ ColBERT model loaded!
```

## Key Features Implemented

1. **Loading State Management**: Separate state tracking for Direct MaxSim, FastPlaid, and ColBERT
2. **Visual Loading Indicators**: Hourglass (⏳) emoji in size boxes during loading
3. **Independent Loading**: All three components load in parallel via Promise.all()
4. **Partial Search**: Users can search as soon as ONE index + ColBERT are ready
5. **Graceful Degradation**: Shows loading message in result panel for unready index
6. **Status Updates**: Real-time status messages showing what's loading/ready
7. **Error Handling**: Clear error indicators (❌) with helpful messages

## Implementation Notes

- `loadingState` object tracks loaded/loading/error for each component
- `updateLoadingUI()` updates size boxes based on loading state
- `updateOverallStatus()` determines when to enable search button
- `performSearch()` checks which indexes are ready and searches accordingly
- `displayResults()` handles null results by showing loading messages
