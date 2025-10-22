# PLAID WASM Implementation Roadmap

## Discovery

**Key Finding**: The FastPlaid repository **already has a complete PLAID implementation** in the Rust library (`rust/search/`, `rust/index/`, `rust/utils/`), but it's **NOT yet ported to the WASM demo**.

### What Exists (Rust Library - Python Bindings)

| Feature | Status | Location |
|---------|--------|----------|
| **IVF Clustering** | ✅ Implemented | `rust/search/search.rs` (lines 252-421) |
| **Residual Quantization** | ✅ Implemented | `rust/utils/residual_codec.rs` |
| **Multi-stage Search** | ✅ Implemented | `rust/search/search.rs::search()` |
| **F16 Quantization** | ✅ Implemented | `residual_codec.rs:117` |
| **Decompression** | ✅ Implemented | `search.rs::decompress_residuals()` |
| **ColBERT MaxSim** | ✅ Implemented | `search.rs::colbert_score_reduce()` |

### What's Missing (WASM Demo)

| Feature | Status | Location |
|---------|--------|----------|
| **SIMD MaxSim** | ✅ Implemented | `rust/lib_wasm.rs` |
| **Exhaustive Search** | ✅ Implemented | `rust/lib_wasm.rs` |
| **IVF Clustering** | ❌ Not ported | Needs implementation |
| **Quantization** | ❌ Not ported | Needs implementation |
| **Two-stage Search** | ❌ Not ported | Needs implementation |

---

## Why WASM Doesn't Have PLAID Yet

### Technical Challenges

1. **Dependency on Candle**: The main Rust library uses `candle_core` (ML tensor library), which:
   - Is NOT compiled for `wasm32` target
   - Requires GPU/CPU compute backends
   - Has heavy dependencies

2. **Complexity**: The existing PLAID implementation has:
   - ~500 lines of tensor operations
   - Complex data structures (`StridedTensor`, `ResidualCodec`)
   - Multi-stage pipeline with batching

3. **WASM Constraints**:
   - No access to `candle_core` in WASM
   - Need pure Rust + `wasm-bindgen` implementation
   - Memory and performance considerations

### Current WASM Approach

The WASM demo was built for **simplicity and demonstration**:
- Focus: Show SIMD performance benefits
- Strategy: Exhaustive search with SIMD-optimized MaxSim
- Result: 3.3x faster than JavaScript

---

## PLAID Implementation Plan

### Phase 1: Simplified IVF (No Quantization)

**Goal**: Implement IVF clustering for 10-100x speedup without residual quantization

**Estimated Effort**: 4-6 hours

**Components**:

1. **K-Means Clustering** (~2 hours)
   ```rust
   // Cluster document centroids into k clusters
   fn build_ivf_index(documents: &[DocumentEmbedding], num_clusters: usize) -> Vec<IVFCluster>
   ```
   - Input: Document embeddings
   - Output: K centroids + document-to-cluster mapping
   - Algorithm: Simple k-means (10-20 iterations)

2. **IVF Index Structure** (~1 hour)
   ```rust
   struct IVFCluster {
       centroid: Vec<f32>,           // Cluster centroid embedding
       document_ids: Vec<usize>,     // Documents in this cluster
   }
   ```

3. **Two-Stage Search** (~2 hours)
   ```rust
   fn search_with_ivf(query, n_probe, top_k) -> Results {
       // Stage 1: Find n_probe nearest clusters
       let clusters = find_nearest_clusters(query, n_probe);

       // Stage 2: Exhaustive search within selected clusters only
       let candidates = get_documents_from_clusters(clusters);
       let results = maxsim_search(query, candidates, top_k);

       results
   }
   ```

4. **Wiring & Testing** (~1 hour)
   - Expose `build_ivf_index()` to WASM
   - Update `search()` to use IVF when enabled
   - Add UI toggle for IVF vs exhaustive

**Expected Performance**:
- 100 docs: ~2x speedup (not worth it)
- 1,000 docs: ~10x speedup
- 10,000 docs: ~50-100x speedup
- 100,000 docs: ~100-500x speedup

**Tradeoffs**:
- ✅ Massive speedup for large document sets
- ✅ Relatively simple to implement
- ❌ Slight accuracy loss (misses ~1-5% of results)
- ❌ Index building time (~1-2s for 10K docs)

---

### Phase 2: Residual Quantization (Advanced)

**Goal**: Add 50% memory savings through f16 quantization

**Estimated Effort**: 8-12 hours

**Components**:

1. **Centroids + Residuals** (~3 hours)
   - Compute coarse centroids (IVF clusters)
   - Compute residuals: `residual = embedding - centroid`
   - Store as f16 instead of f32

2. **Bucket Quantization** (~4 hours)
   - Divide residuals into buckets
   - Store bucket indices instead of full residuals
   - Requires codebook lookup tables

3. **Decompression** (~2 hours)
   - Reconstruct: `embedding = centroid + decompress(residual)`
   - Bit manipulation for packed storage

4. **Integration** (~2 hours)
   - Update search to decompress on-the-fly
   - Benchmark memory and performance

**Expected Benefits**:
- **Memory**: 50% reduction (f32 → f16)
- **Speed**: Minimal impact (decompression overhead)
- **Accuracy**: ~1% score difference

**Complexity**: HIGH - requires careful bit-packing and lookup tables

---

### Phase 3: Full PLAID (Research-Grade)

**Goal**: Port the complete Rust library implementation to WASM

**Estimated Effort**: 20-30 hours

**Components**:
- All of Phase 1 + Phase 2
- Advanced: Product quantization (PQ)
- Advanced: Residual compressed storage
- Advanced: Batch processing pipeline
- Extensive testing and optimization

**When to do this**: Only if targeting 100K+ documents in browser

---

## Recommendation

### For Current Demo (100 docs):
**Status Quo**: Keep exhaustive SIMD search
- **Why**: 100 docs is already fast (4.5ms)
- **IVF overhead**: Would actually slow it down
- **Verdict**: ✅ Current implementation is optimal

### For 1K-10K Documents:
**Phase 1**: Implement simplified IVF (no quantization)
- **Why**: 10-50x speedup justifies 4-6 hour investment
- **Memory**: Still uses f32 (no savings)
- **Verdict**: ✅ **High ROI** if you need to scale

### For 100K+ Documents:
**Phase 2 + 3**: Full PLAID with quantization
- **Why**: Memory becomes critical bottleneck
- **Effort**: Significant (15-20+ hours)
- **Verdict**: ⚠️ Only if this is production use case

---

## Implementation Checklist (Phase 1)

If you decide to proceed with Phase 1 (IVF only):

### 1. K-Means Clustering
- [ ] Implement `kmeans_cluster()` in `lib_wasm.rs`
  - [ ] Random centroid initialization
  - [ ] Assignment step (find nearest centroid)
  - [ ] Update step (recompute centroids)
  - [ ] Convergence check (10-20 iterations)
- [ ] Test with sample data

### 2. IVF Index Building
- [ ] Add `build_ivf_index()` method to `FastPlaidWasm`
- [ ] Expose to JavaScript via `wasm-bindgen`
- [ ] Call from `mxbai-integration.js` after loading documents

### 3. IVF Search
- [ ] Implement `find_nearest_clusters()`
- [ ] Modify `search()` to support IVF mode
- [ ] Add `n_ivf_probe` parameter (default: 10)

### 4. UI & Testing
- [ ] Add "Use IVF" checkbox in demo
- [ ] Display IVF stats (# clusters, avg docs/cluster)
- [ ] Benchmark: IVF vs Exhaustive
- [ ] Update PERFORMANCE_ANALYSIS.md

### 5. Documentation
- [ ] Document IVF parameters (n_clusters, n_probe)
- [ ] Add performance charts
- [ ] Update README with PLAID status

---

## Code Skeleton (Phase 1)

### `lib_wasm.rs` additions

```rust
impl FastPlaidWasm {
    /// Build IVF index using k-means clustering
    #[wasm_bindgen]
    pub fn build_ivf_index(&mut self, num_clusters: usize) -> Result<(), JsValue> {
        console_log!("🔨 Building IVF index with {} clusters...", num_clusters);

        if self.documents.is_empty() {
            return Err(JsValue::from_str("No documents loaded"));
        }

        // Get document centroids (average embedding per document)
        let doc_centroids = self.compute_document_centroids();

        // Run k-means clustering
        let clusters = self.kmeans_cluster(&doc_centroids, num_clusters)?;

        // Assign documents to clusters
        for (doc_idx, doc_centroid) in doc_centroids.iter().enumerate() {
            let nearest_cluster = self.find_nearest_cluster(doc_centroid, &clusters);
            clusters[nearest_cluster].document_ids.push(doc_idx);
        }

        self.ivf_clusters = clusters;
        self.ivf_enabled = true;

        console_log!("✅ IVF index built: {} clusters", self.ivf_clusters.len());
        Ok(())
    }

    /// Compute centroid embedding for each document
    fn compute_document_centroids(&self) -> Vec<Vec<f32>> {
        self.documents.iter().map(|doc| {
            // Average all token embeddings
            let mut centroid = vec![0.0; self.embedding_dim];
            for token_idx in 0..doc.num_tokens {
                for dim_idx in 0..self.embedding_dim {
                    centroid[dim_idx] += doc.embeddings[token_idx * self.embedding_dim + dim_idx];
                }
            }
            // Normalize
            for val in &mut centroid {
                *val /= doc.num_tokens as f32;
            }
            centroid
        }).collect()
    }

    /// K-means clustering algorithm
    fn kmeans_cluster(&self, points: &[Vec<f32>], k: usize) -> Result<Vec<IVFCluster>, JsValue> {
        // Initialize centroids randomly
        let mut clusters: Vec<IVFCluster> = (0..k).map(|i| {
            IVFCluster {
                centroid: points[i % points.len()].clone(),
                document_ids: Vec::new(),
            }
        }).collect();

        // Iterate until convergence (max 20 iterations)
        for iter in 0..20 {
            // Assignment step
            let mut assignments = vec![Vec::new(); k];
            for (point_idx, point) in points.iter().enumerate() {
                let nearest = self.find_nearest_cluster(point, &clusters);
                assignments[nearest].push(point_idx);
            }

            // Update step
            for (cluster_idx, assigned_points) in assignments.iter().enumerate() {
                if assigned_points.is_empty() {
                    continue;
                }

                // Recompute centroid
                let mut new_centroid = vec![0.0; self.embedding_dim];
                for &point_idx in assigned_points {
                    for (dim_idx, val) in points[point_idx].iter().enumerate() {
                        new_centroid[dim_idx] += val;
                    }
                }
                for val in &mut new_centroid {
                    *val /= assigned_points.len() as f32;
                }

                clusters[cluster_idx].centroid = new_centroid;
            }

            console_log!("  K-means iteration {}/20", iter + 1);
        }

        Ok(clusters)
    }

    /// Find nearest cluster to a point
    fn find_nearest_cluster(&self, point: &[f32], clusters: &[IVFCluster]) -> usize {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (idx, cluster) in clusters.iter().enumerate() {
            let score = Self::dot_product_scalar(point, &cluster.centroid, self.embedding_dim);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Search with IVF acceleration
    fn search_with_ivf(
        &self,
        query_emb: &[f32],
        query_tokens: usize,
        n_probe: usize,
        top_k: usize,
    ) -> Vec<(i64, f32)> {
        // Compute query centroid
        let query_centroid = self.compute_query_centroid(query_emb, query_tokens);

        // Find n_probe nearest clusters
        let mut cluster_scores: Vec<(usize, f32)> = self.ivf_clusters.iter().enumerate()
            .map(|(idx, cluster)| {
                let score = Self::dot_product_scalar(&query_centroid, &cluster.centroid, self.embedding_dim);
                (idx, score)
            })
            .collect();

        cluster_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        cluster_scores.truncate(n_probe);

        // Collect candidate documents from selected clusters
        let mut candidate_doc_ids = Vec::new();
        for (cluster_idx, _) in cluster_scores {
            candidate_doc_ids.extend(&self.ivf_clusters[cluster_idx].document_ids);
        }

        // Exhaustive search within candidates only
        let mut scores: Vec<(i64, f32)> = candidate_doc_ids.iter()
            .map(|&doc_idx| {
                let doc = &self.documents[doc_idx];
                let score = Self::calculate_maxsim(
                    query_emb,
                    &doc.embeddings,
                    query_tokens,
                    doc.num_tokens,
                    self.embedding_dim,
                );
                (doc.id, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);

        scores
    }
}
```

---

## Next Steps

1. **Decision Point**: Do you want to implement Phase 1 now?
   - **Yes**: Follow the checklist above (~4-6 hours)
   - **No**: Keep this roadmap for future reference

2. **If proceeding**:
   - Start with k-means implementation
   - Test with current 100-doc demo
   - Expand to 1K-10K docs for meaningful benchmarks

3. **Alternative**: Keep current implementation
   - Current demo is already fast enough for 100 docs
   - Memory tracking added (50% savings already documented)
   - SIMD providing 3.3x speedup

---

## References

- Existing PLAID implementation: `rust/search/search.rs`
- Residual codec: `rust/utils/residual_codec.rs`
- PLAID paper: https://arxiv.org/abs/2205.09707
- ColBERT paper: https://arxiv.org/abs/2004.12832

---

**Status**: Roadmap complete. Ready for implementation decision.

**Recommendation**: For a 100-document demo, the current SIMD-optimized exhaustive search is optimal. Consider IVF only when scaling to 1,000+ documents.
