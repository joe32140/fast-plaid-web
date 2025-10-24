# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-01-24

### Added
- **Incremental Index Updates** - Add documents without full IVF rebuild (8x faster for small batches!)
  - `update_index_incremental()` - Add documents using existing codec
  - `compact_index()` - Manual compaction control
  - Auto-compaction at 10% delta threshold
  - Delta-encoded IVF with transparent search merging
- **WASM Initialization Retry Logic** - Robust initialization with exponential backoff
  - Handles `WebAssembly.Table.grow()` failures automatically
  - 3 retries with exponential backoff (500ms, 1000ms, 2000ms)
- **Enhanced Index Statistics** - Extended `get_index_info()` with delta tracking
  - `base_documents` - Count in base IVF
  - `pending_deltas` - Count in delta log
  - `delta_ratio_percent` - Delta percentage vs base
  - `incremental_updates` - Feature status

### Changed
- Updated `search()` to merge base IVF with pending deltas transparently
- Modified `save_index()` to auto-compact deltas before saving
- Enhanced `load_index()` to initialize delta tracking

### Performance
- 8.3x faster updates for small batches (<100 docs)
- 2.7x faster updates for large batches (1000 docs)
- <5% search overhead with deltas
- ~16 bytes memory per delta entry

### Documentation
- Added [INCREMENTAL_UPDATES.md](INCREMENTAL_UPDATES.md) - Complete API reference with examples
- Added [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- Added [TEST_RESULTS.md](TEST_RESULTS.md) - Verified test results
- Added [test_incremental_update.html](test_incremental_update.html) - Interactive demo
- Updated [README.md](README.md) - Featured incremental updates

### Fixed
- WASM initialization failures now handled gracefully with retry logic
- Console.log recursion in test page fixed

## [1.2.4] - 2025-01-23

### Added
- Initial release with 4-bit quantization and IVF support
- WASM support for browser-native search
- MaxSim search with token-level late interaction
- Offline index building for GitHub Pages deployment

### Features
- 8x compression with 4-bit quantization
- 3-5x faster search with IVF indexing
- Pure Rust implementation
- SIMD optimizations for WASM

---

## Migration Guide

### Upgrading from 1.2.x to 1.3.0

The new incremental update feature is **backward compatible**. Existing code continues to work:

```javascript
// Existing code (still works)
plaid.load_documents_quantized(embeddings, docInfo, 256);
const results = plaid.search(query, queryShape, 10);

// New feature (optional)
plaid.update_index_incremental(newEmbeddings, newDocInfo);
```

**New APIs:**
- `update_index_incremental(embeddings, doc_info)` - Add documents incrementally
- `compact_index()` - Manually trigger compaction
- Enhanced `get_index_info()` - Returns delta statistics

**Behavioral Changes:**
- `save_index()` now auto-compacts deltas (transparent, improves load time)
- `load_index()` initializes delta tracking (no visible impact)

**No Breaking Changes** - All existing APIs remain unchanged.

## Future Roadmap

### Planned Features
- [ ] Incremental deletions with tombstone markers
- [ ] Codec versioning for distribution drift detection
- [ ] Smart compaction based on search performance metrics
- [ ] Background compaction using Web Workers
- [ ] Per-cluster IVF files for true append-only updates

### Under Consideration
- [ ] Native Rust incremental updates (Python bindings)
- [ ] Distributed index for large-scale deployments
- [ ] Vector database backend adapters (Milvus, Qdrant)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
