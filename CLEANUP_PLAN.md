# Cleanup Plan for FastPlaid Repository

## 📚 Documentation to Keep

### Essential Guides
- **QUANTIZATION_GUIDE.md** - ✅ Educational deep-dive on quantization (500+ lines)
- **NATIVE_MATCHING_ROADMAP.md** - ✅ Implementation roadmap (useful reference)

### To Archive (move to `docs/archive/`)
- ASYNC_LOADING_IMPLEMENTATION.md - Historical implementation notes
- ASYNC_LOADING_SUMMARY.md - Historical summary
- BUILD_INDEX_FINAL.md - Superseded by current implementation
- BUILD_INDEX_GUIDE.md - Superseded
- BUILD_LOCAL_SUCCESS.md - Historical notes
- FASTPLAID_LOADING_EXPLANATION.md - Historical explanation
- LOADING_FIX_SUMMARY.md - Historical fixes
- OFFLINE_INDEX_GUIDE.md - Superseded
- QUICK_BUILD_INSTRUCTIONS.md - Superseded
- README_LOADING.md - Historical
- SOLUTION_SUMMARY.md - Historical

## 🗂️ Data Files to Clean Up

### Keep in `demo/data/`
- **embeddings.bin** - ✅ Source embeddings for all 1000 papers
- **embeddings_meta.json** - ✅ Metadata for embeddings
- **papers_metadata.json** - ✅ Paper titles/abstracts for display
- **papers_1000.json** - ✅ Source data

### Remove (old test data)
- `demo/data/fastplaid_4bit/` - Old Python-generated index (superseded)
- `demo/data/fastplaid_optimized/` - Old experiments
- `demo/data/fastplaid_optimized_fixed/` - Old experiments
- `demo/data/offline_index_1500/` - Old experiments
- `demo/data/offline_index_test/` - Old experiments
- `demo/data/fastplaid_index.bin.old_python` - Archived Python index
- `demo/data/papers_*.json` (except papers_1000.json and papers_metadata.json) - Test data

### To Create
- **demo/data/fastplaid_index.bin** - NEW precomputed index with learned quantization

## 🧹 Cleanup Commands

```bash
# 1. Create archive directory
mkdir -p docs/archive

# 2. Move historical docs
mv ASYNC_LOADING_*.md BUILD_*.md LOADING_*.md OFFLINE_*.md \
   QUICK_BUILD_INSTRUCTIONS.md README_LOADING.md \
   SOLUTION_SUMMARY.md FASTPLAID_LOADING_EXPLANATION.md docs/archive/

# 3. Remove old data directories
rm -rf demo/data/fastplaid_4bit \
       demo/data/fastplaid_optimized \
       demo/data/fastplaid_optimized_fixed \
       demo/data/offline_index_* \
       demo/data/fastplaid_index.bin.old_python

# 4. Remove old test papers
rm demo/data/papers_500.json \
   demo/data/papers_arxiv_*.json \
   demo/data/papers_synthetic_*.json \
   demo/data/papers_test_*.json

# 5. Build new precomputed index (see BUILD_PRECOMPUTED_INDEX.md)
```

## 📝 New Documentation to Create

- **BUILD_PRECOMPUTED_INDEX.md** - How to build the production index
- **DEPLOYMENT_GUIDE.md** - How to deploy FastPlaid demo
- **ARCHITECTURE.md** - System architecture overview
