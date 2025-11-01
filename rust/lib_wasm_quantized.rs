// WASM implementation with 4-bit quantization for maximum density
// This allows ~8x more paper abstracts to fit in GitHub Pages limits
//
// Based on FastPLAID's actual quantization strategy (nbits=4)
// See: ACTUAL_PLAID_QUANTIZATION.md

use wasm_bindgen::prelude::*;
use web_sys::console;
use serde::{Serialize, Deserialize};

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

// Utility function for logging to browser console
macro_rules! console_log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()))
}

/// 4-bit quantization codec (FastPLAID-style)
#[derive(Clone, Debug)]
struct ResidualCodec4bit {
    embedding_dim: usize,
    num_centroids: usize,
    centroids: Vec<f32>,  // [num_centroids, embedding_dim]
    bucket_weights: Vec<f32>,  // [16, embedding_dim] - 2^4 = 16 buckets
    bucket_cutoffs: Vec<f32>,  // [17] - boundaries between buckets (including -∞ and ∞)
    avg_residual: Vec<f32>,  // [embedding_dim] - average residual across all tokens
}

impl ResidualCodec4bit {
    /// Create a new codec with k-means centroids
    fn new(embedding_dim: usize, num_centroids: usize) -> Self {
        // Initialize with random centroids (will be trained)
        let centroids = vec![0.0; num_centroids * embedding_dim];

        // Initialize 16 bucket weights (4-bit = 16 levels)
        // Uniformly distributed in [-1, 1] range
        let mut bucket_weights = Vec::with_capacity(16 * embedding_dim);
        for i in 0..16 {
            for _ in 0..embedding_dim {
                let weight = -1.0 + (i as f32) * (2.0 / 15.0);
                bucket_weights.push(weight);
            }
        }

        Self {
            embedding_dim,
            num_centroids,
            centroids,
            bucket_weights,
            bucket_cutoffs: vec![0.0; 17],  // Will be computed during training (17 boundaries for 16 buckets)
            avg_residual: vec![0.0; embedding_dim],  // Will be computed during training
        }
    }

    /// Train centroids using k-means on sample embeddings
    fn train(&mut self, embeddings: &[f32], _num_samples: usize, num_iters: usize) {
        let num_tokens = embeddings.len() / self.embedding_dim;
        if num_tokens == 0 {
            return;
        }

        // Simple k-means initialization: pick random samples as initial centroids
        let step = num_tokens.max(self.num_centroids) / self.num_centroids;
        for c in 0..self.num_centroids {
            let token_idx = (c * step).min(num_tokens - 1);
            let src_start = token_idx * self.embedding_dim;
            let dst_start = c * self.embedding_dim;
            self.centroids[dst_start..dst_start + self.embedding_dim]
                .copy_from_slice(&embeddings[src_start..src_start + self.embedding_dim]);
        }

        // K-means iterations
        for _iter in 0..num_iters {
            let mut new_centroids = vec![0.0; self.num_centroids * self.embedding_dim];
            let mut counts = vec![0; self.num_centroids];

            // Assignment step
            for token_idx in 0..num_tokens {
                let token_start = token_idx * self.embedding_dim;
                let token_emb = &embeddings[token_start..token_start + self.embedding_dim];

                // Find nearest centroid
                let centroid_idx = self.find_nearest_centroid(token_emb);

                // Accumulate
                let dst_start = centroid_idx * self.embedding_dim;
                for d in 0..self.embedding_dim {
                    new_centroids[dst_start + d] += token_emb[d];
                }
                counts[centroid_idx] += 1;
            }

            // Update step
            for c in 0..self.num_centroids {
                if counts[c] > 0 {
                    let start = c * self.embedding_dim;
                    for d in 0..self.embedding_dim {
                        new_centroids[start + d] /= counts[c] as f32;
                    }
                }
            }

            self.centroids = new_centroids;
        }

        // After training centroids, compute average residual and learn bucket weights
        self.compute_avg_residual(embeddings, num_tokens);
        self.train_bucket_weights(embeddings, num_tokens);

        console_log!("✅ Trained {} centroids with {} iterations", self.num_centroids, num_iters);
    }

    /// Compute average residual across all tokens (Phase 1: Native parity)
    fn compute_avg_residual(&mut self, embeddings: &[f32], num_tokens: usize) {
        let mut sum_residuals = vec![0.0; self.embedding_dim];
        let mut count = 0;

        for token_idx in 0..num_tokens {
            let token_start = token_idx * self.embedding_dim;
            let token_emb = &embeddings[token_start..token_start + self.embedding_dim];

            let centroid_idx = self.find_nearest_centroid(token_emb);
            let centroid_start = centroid_idx * self.embedding_dim;
            let centroid = &self.centroids[centroid_start..centroid_start + self.embedding_dim];

            for d in 0..self.embedding_dim {
                sum_residuals[d] += token_emb[d] - centroid[d];
            }
            count += 1;
        }

        // Average across all tokens
        for d in 0..self.embedding_dim {
            self.avg_residual[d] = sum_residuals[d] / count as f32;
        }

        console_log!("   Average residual computed (Phase 1: Native parity)");
    }

    /// Train bucket weights based on actual residual distribution
    fn train_bucket_weights(&mut self, embeddings: &[f32], num_tokens: usize) {
        // Collect PER-DIMENSION residuals to compute statistics
        // We need to learn separate bucket weights for each dimension
        let mut residuals_per_dim: Vec<Vec<f32>> = vec![Vec::with_capacity(num_tokens); self.embedding_dim];

        for token_idx in 0..num_tokens {
            let token_start = token_idx * self.embedding_dim;
            let token_emb = &embeddings[token_start..token_start + self.embedding_dim];

            let centroid_idx = self.find_nearest_centroid(token_emb);
            let centroid_start = centroid_idx * self.embedding_dim;
            let centroid = &self.centroids[centroid_start..centroid_start + self.embedding_dim];

            for d in 0..self.embedding_dim {
                // Centered residual: subtract both centroid AND average residual
                let residual = token_emb[d] - centroid[d] - self.avg_residual[d];
                residuals_per_dim[d].push(residual);
            }
        }

        // For each dimension, learn bucket weights from percentiles
        for d in 0..self.embedding_dim {
            // Sort residuals for this dimension
            residuals_per_dim[d].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // For each bucket, compute the average residual value in that percentile range
            for bucket in 0..16 {
                let start_pct = bucket as f32 / 16.0;
                let end_pct = (bucket + 1) as f32 / 16.0;

                let start_idx = (start_pct * residuals_per_dim[d].len() as f32) as usize;
                let end_idx = (end_pct * residuals_per_dim[d].len() as f32) as usize;

                let bucket_residuals = &residuals_per_dim[d][start_idx..end_idx];
                let avg_residual: f32 = if bucket_residuals.is_empty() {
                    0.0
                } else {
                    bucket_residuals.iter().sum::<f32>() / bucket_residuals.len() as f32
                };

                // Update bucket weight for this dimension
                let weight_idx = bucket * self.embedding_dim + d;
                self.bucket_weights[weight_idx] = avg_residual;
            }
        }

        // Phase 2: Compute explicit bucket cutoffs as MIDPOINTS between bucket weights
        // This enables binary search quantization without changing the learned weights
        //
        // Approach: For each dimension, compute average bucket weight across all dimensions,
        // then set cutoffs as midpoints between consecutive bucket weights

        // Compute global average bucket weight for each bucket (average across dimensions)
        let mut avg_bucket_weights = vec![0.0f32; 16];
        for bucket in 0..16 {
            let mut sum = 0.0;
            for d in 0..self.embedding_dim {
                sum += self.bucket_weights[bucket * self.embedding_dim + d];
            }
            avg_bucket_weights[bucket] = sum / self.embedding_dim as f32;
        }

        // Set cutoffs as midpoints between consecutive average bucket weights
        self.bucket_cutoffs = vec![f32::NEG_INFINITY];
        for i in 0..15 {
            let midpoint = (avg_bucket_weights[i] + avg_bucket_weights[i + 1]) / 2.0;
            self.bucket_cutoffs.push(midpoint);
        }
        self.bucket_cutoffs.push(f32::INFINITY);

        // Log statistics
        console_log!("   Bucket weights learned from per-dimension percentiles");
        console_log!("   Bucket cutoffs computed as midpoints (Phase 2: Native parity)");
    }

    /// Find nearest centroid index for a token embedding
    fn find_nearest_centroid(&self, token_emb: &[f32]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for c in 0..self.num_centroids {
            let centroid_start = c * self.embedding_dim;
            let centroid = &self.centroids[centroid_start..centroid_start + self.embedding_dim];

            let dist: f32 = token_emb.iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Quantize a residual value to 4-bit bucket index (0-15)
    /// Phase 2: Use binary search with explicit cutoffs (faster and more precise)
    fn quantize_to_4bit(&self, value: f32) -> u8 {
        // Binary search to find the correct bucket
        // cutoffs = [-∞, c1, c2, ..., c15, ∞]
        // If value < c1, bucket 0; if c1 <= value < c2, bucket 1; etc.

        let mut left = 0;
        let mut right = 16;

        while left < right {
            let mid = (left + right) / 2;
            if value < self.bucket_cutoffs[mid + 1] {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        left.min(15) as u8
    }

    /// Pack two 4-bit values into one byte
    fn pack_4bit(high: u8, low: u8) -> u8 {
        ((high & 0x0F) << 4) | (low & 0x0F)
    }

    /// Unpack one byte into two 4-bit values
    fn unpack_4bit(packed: u8) -> (u8, u8) {
        let high = (packed >> 4) & 0x0F;
        let low = packed & 0x0F;
        (high, low)
    }

    /// Encode a document's token embeddings
    /// Returns: (centroid_codes, packed_residuals)
    fn encode_document(&self, embeddings: &[f32], num_tokens: usize) -> (Vec<u8>, Vec<u8>) {
        let mut centroid_codes = Vec::with_capacity(num_tokens);
        let mut residual_codes = Vec::with_capacity(num_tokens * self.embedding_dim);

        for token_idx in 0..num_tokens {
            let token_start = token_idx * self.embedding_dim;
            let token_emb = &embeddings[token_start..token_start + self.embedding_dim];

            // Find nearest centroid
            let centroid_idx = self.find_nearest_centroid(token_emb);
            centroid_codes.push(centroid_idx as u8);

            // Compute residual
            let centroid_start = centroid_idx * self.embedding_dim;
            let centroid = &self.centroids[centroid_start..centroid_start + self.embedding_dim];

            // Quantize each dimension of residual to 4 bits
            for d in 0..self.embedding_dim {
                // Phase 1: Subtract avg_residual to center the distribution
                let residual = token_emb[d] - centroid[d] - self.avg_residual[d];
                let bucket = self.quantize_to_4bit(residual);
                residual_codes.push(bucket);
            }
        }

        // Pack residual codes (2 per byte)
        let mut packed_residuals = Vec::with_capacity(residual_codes.len() / 2 + 1);
        for chunk in residual_codes.chunks(2) {
            let high = chunk[0];
            let low = chunk.get(1).copied().unwrap_or(0);
            packed_residuals.push(Self::pack_4bit(high, low));
        }

        (centroid_codes, packed_residuals)
    }

    /// Decode document embeddings from quantized representation
    fn decode_document(&self, centroid_codes: &[u8], packed_residuals: &[u8], num_tokens: usize) -> Vec<f32> {
        let mut embeddings = Vec::with_capacity(num_tokens * self.embedding_dim);

        // Unpack residuals
        let mut residual_codes = Vec::with_capacity(num_tokens * self.embedding_dim);
        for &packed in packed_residuals {
            let (high, low) = Self::unpack_4bit(packed);
            residual_codes.push(high);
            residual_codes.push(low);
        }

        for token_idx in 0..num_tokens {
            let centroid_idx = centroid_codes[token_idx] as usize;
            let centroid_start = centroid_idx * self.embedding_dim;
            let centroid = &self.centroids[centroid_start..centroid_start + self.embedding_dim];

            // Reconstruct from centroid + quantized residual
            for d in 0..self.embedding_dim {
                let residual_idx = token_idx * self.embedding_dim + d;
                let bucket = residual_codes[residual_idx] as usize;

                // Lookup bucket weight
                let weight_idx = bucket * self.embedding_dim + d;
                let residual = self.bucket_weights.get(weight_idx).copied().unwrap_or(0.0);

                // Phase 1: Add centroid + avg_residual + centered_residual
                embeddings.push(centroid[d] + self.avg_residual[d] + residual);
            }
        }

        // L2 normalize each token embedding (CRITICAL for ColBERT!)
        for token_idx in 0..num_tokens {
            let token_start = token_idx * self.embedding_dim;
            let token_end = token_start + self.embedding_dim;
            let token = &mut embeddings[token_start..token_end];

            // Compute L2 norm
            let norm: f32 = token.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Normalize (avoid division by zero)
            if norm > 1e-12 {
                for val in token.iter_mut() {
                    *val /= norm;
                }
            }
        }

        embeddings
    }
}

/// Quantized document embedding (4-bit compressed)
#[derive(Clone, Debug)]
struct QuantizedDocument {
    id: i64,
    centroid_codes: Vec<u8>,      // [num_tokens] - 1 byte each
    packed_residuals: Vec<u8>,    // [num_tokens * embedding_dim / 2] - 4-bit packed
    num_tokens: usize,
}

/// Search result for a single query
#[derive(Serialize, Deserialize, Debug)]
pub struct QueryResult {
    pub query_id: usize,
    pub passage_ids: Vec<i64>,
    pub scores: Vec<f32>,
    pub candidates_searched: usize,
    pub num_clusters_probed: usize,
}

/// Delta entry for incremental IVF updates
#[derive(Clone, Debug)]
struct IVFDelta {
    cluster_id: usize,
    doc_id: usize,
}

/// WASM wrapper for quantized FastPlaid
#[wasm_bindgen]
pub struct FastPlaidQuantized {
    index_loaded: bool,
    embedding_dim: usize,
    documents: Vec<QuantizedDocument>,
    codec: Option<ResidualCodec4bit>,
    // IVF (Inverted File Index) for fast approximate search
    ivf_clusters: Vec<Vec<usize>>,  // cluster_id -> [doc_indices] (base index)
    ivf_centroids: Vec<f32>,         // [num_clusters, embedding_dim]
    num_clusters: usize,
    nprobe: usize,                   // Number of clusters to probe PER QUERY TOKEN (PLAID-style)
    // Delta-encoded IVF for incremental updates
    ivf_deltas: Vec<IVFDelta>,       // Pending additions to IVF
    delta_threshold: usize,          // Compact when deltas exceed this percentage
    base_doc_count: usize,           // Number of documents in base index
}

#[wasm_bindgen]
impl FastPlaidQuantized {
    /// Creates a new quantized FastPlaid instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<FastPlaidQuantized, JsValue> {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        console_log!("🚀 Initializing FastPlaid WASM with 4-bit quantization...");
        console_log!("   🗜️ 8x compression vs f32 - fit more papers in GitHub Pages!");
        console_log!("   🔄 With incremental update support!");

        Ok(FastPlaidQuantized {
            index_loaded: false,
            embedding_dim: 384,
            documents: Vec::new(),
            codec: None,
            ivf_clusters: Vec::new(),
            ivf_centroids: Vec::new(),
            num_clusters: 0,
            nprobe: 4, // PLAID default: 4 clusters per query token
            ivf_deltas: Vec::new(),
            delta_threshold: 10, // Compact when deltas exceed 10% of base
            base_doc_count: 0,
        })
    }

    /// Load and quantize document embeddings
    /// Training happens automatically on the provided embeddings
    #[wasm_bindgen]
    pub fn load_documents_quantized(
        &mut self,
        embeddings_data: &[f32],
        doc_info: &[i64],
        num_centroids: Option<usize>,
    ) -> Result<(), JsValue> {
        console_log!("📥 Loading and quantizing documents...");
        console_log!("   Total embedding data: {} floats ({:.2} MB uncompressed)",
            embeddings_data.len(),
            (embeddings_data.len() * 4) as f32 / 1_000_000.0
        );

        if doc_info.len() % 2 != 0 {
            return Err(JsValue::from_str("doc_info must contain pairs of [id, num_tokens]"));
        }

        let num_docs = doc_info.len() / 2;

        // Auto-detect embedding dimension
        let total_tokens: usize = (0..num_docs)
            .map(|i| doc_info[i * 2 + 1] as usize)
            .sum();

        if total_tokens > 0 {
            let detected_dim = embeddings_data.len() / total_tokens;
            if detected_dim != self.embedding_dim {
                console_log!("🔧 Auto-detected embedding_dim: {} (was {})", detected_dim, self.embedding_dim);
                self.embedding_dim = detected_dim;
            }
        }

        // Initialize and train codec
        let num_centroids = num_centroids.unwrap_or(256.min(total_tokens / 10));
        let mut codec = ResidualCodec4bit::new(self.embedding_dim, num_centroids);
        codec.train(embeddings_data, total_tokens, 4);

        // Encode all documents
        let mut documents = Vec::with_capacity(num_docs);
        let mut offset = 0;
        let mut total_compressed_size = 0;

        for i in 0..num_docs {
            let doc_id = doc_info[i * 2];
            let num_tokens = doc_info[i * 2 + 1] as usize;
            let embedding_size = num_tokens * self.embedding_dim;

            if offset + embedding_size > embeddings_data.len() {
                return Err(JsValue::from_str(&format!(
                    "Not enough embedding data for document {}",
                    doc_id
                )));
            }

            let doc_embeddings = &embeddings_data[offset..offset + embedding_size];

            // Encode with 4-bit quantization
            let (centroid_codes, packed_residuals) = codec.encode_document(doc_embeddings, num_tokens);

            total_compressed_size += centroid_codes.len() + packed_residuals.len();

            documents.push(QuantizedDocument {
                id: doc_id,
                centroid_codes,
                packed_residuals,
                num_tokens,
            });

            offset += embedding_size;
        }

        let original_size = embeddings_data.len() * 4; // f32 = 4 bytes
        let compression_ratio = original_size as f32 / total_compressed_size as f32;

        console_log!("✅ Quantized {} documents:", documents.len());
        console_log!("   Original size: {:.2} MB", original_size as f32 / 1_000_000.0);
        console_log!("   Compressed size: {:.2} MB", total_compressed_size as f32 / 1_000_000.0);
        console_log!("   Compression ratio: {:.1}x", compression_ratio);
        console_log!("   🎯 Can fit {:.0}x more papers in GitHub Pages!", compression_ratio);

        self.documents = documents;
        self.codec = Some(codec);

        // Build IVF index for fast approximate search
        console_log!("🎯 Building IVF index for fast search...");
        self.build_ivf_index(embeddings_data, doc_info, num_docs)?;

        self.index_loaded = true;
        self.base_doc_count = num_docs;

        Ok(())
    }

    /// Incrementally add new documents to the index without full rebuild
    /// Uses existing codec to compress new documents and stores IVF updates as deltas
    #[wasm_bindgen]
    pub fn update_index_incremental(
        &mut self,
        embeddings_data: &[f32],
        doc_info: &[i64],
    ) -> Result<(), JsValue> {
        if !self.index_loaded {
            return Err(JsValue::from_str("No index loaded. Call load_documents_quantized() first."));
        }

        let codec = self.codec.as_ref()
            .ok_or_else(|| JsValue::from_str("Codec not initialized"))?;

        console_log!("🔄 Incrementally updating index...");

        if doc_info.len() % 2 != 0 {
            return Err(JsValue::from_str("doc_info must contain pairs of [id, num_tokens]"));
        }

        let num_new_docs = doc_info.len() / 2;
        let total_new_tokens: usize = (0..num_new_docs)
            .map(|i| doc_info[i * 2 + 1] as usize)
            .sum();

        console_log!("   Adding {} new documents ({} tokens)", num_new_docs, total_new_tokens);

        // Encode new documents using EXISTING codec (no retraining)
        let mut offset = 0;
        let start_doc_idx = self.documents.len();

        for i in 0..num_new_docs {
            let doc_id = doc_info[i * 2];
            let num_tokens = doc_info[i * 2 + 1] as usize;
            let embedding_size = num_tokens * self.embedding_dim;

            if offset + embedding_size > embeddings_data.len() {
                return Err(JsValue::from_str(&format!(
                    "Not enough embedding data for document {}",
                    doc_id
                )));
            }

            let doc_embeddings = &embeddings_data[offset..offset + embedding_size];

            // Encode with existing codec
            let (centroid_codes, packed_residuals) = codec.encode_document(doc_embeddings, num_tokens);

            // Add document
            let doc_idx = self.documents.len();
            self.documents.push(QuantizedDocument {
                id: doc_id,
                centroid_codes,
                packed_residuals,
                num_tokens,
            });

            // Determine which IVF cluster this document belongs to
            // Use first token as representative (same as build_ivf_index)
            let first_token = &doc_embeddings[0..self.embedding_dim];
            let cluster_id = self.find_nearest_ivf_cluster(first_token);

            // Add to deltas instead of modifying base IVF
            self.ivf_deltas.push(IVFDelta {
                cluster_id,
                doc_id: doc_idx,
            });

            offset += embedding_size;
        }

        console_log!("   ✅ Added {} documents (total: {} docs, {} deltas)",
            num_new_docs, self.documents.len(), self.ivf_deltas.len());

        // Check if compaction is needed
        let delta_ratio = (self.ivf_deltas.len() as f32 / self.base_doc_count.max(1) as f32) * 100.0;
        if delta_ratio >= self.delta_threshold as f32 {
            console_log!("   📊 Delta ratio {:.1}% exceeds threshold {}%, triggering compaction...",
                delta_ratio, self.delta_threshold);
            self.compact_deltas()?;
        } else {
            console_log!("   📊 Delta ratio: {:.1}% (threshold: {}%)", delta_ratio, self.delta_threshold);
        }

        Ok(())
    }

    /// Set nprobe (clusters to probe per query token)
    /// PLAID default: 4 clusters per token
    /// Higher values = better recall, slower search
    /// Lower values = faster search, lower recall
    #[wasm_bindgen]
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe.max(1).min(self.num_clusters);
        console_log!("🔧 Set nprobe = {} clusters per query token", self.nprobe);
    }

    /// Get current nprobe setting
    #[wasm_bindgen]
    pub fn get_nprobe(&self) -> usize {
        self.nprobe
    }

    /// Search with quantized embeddings
    #[wasm_bindgen]
    pub fn search(
        &self,
        query_embeddings: &[f32],
        query_shape: &[usize],
        top_k: usize,
    ) -> Result<String, JsValue> {
        if !self.index_loaded {
            return Err(JsValue::from_str("No index loaded. Call load_documents_quantized() first."));
        }

        let codec = self.codec.as_ref()
            .ok_or_else(|| JsValue::from_str("Codec not initialized"))?;

        // Validate input
        if query_shape.len() != 3 {
            return Err(JsValue::from_str("Query shape must be 3D: [batch_size, seq_len, embedding_dim]"));
        }

        let query_num_tokens = query_shape[1];
        let query_dim = query_shape[2];

        if query_dim != self.embedding_dim {
            return Err(JsValue::from_str(&format!(
                "Query dimension mismatch: {} vs {}",
                query_dim, self.embedding_dim
            )));
        }

        let query_emb = &query_embeddings[0..query_num_tokens * query_dim];

        // V2: Token-level IVF Search (PLAID-style)
        // PLAID paper: nprobe specifies clusters to probe PER QUERY TOKEN, not total
        // Default: nprobe=4 means each query token probes its top-4 nearest clusters
        // For 8 query tokens × 4 clusters/token = ~30 unique clusters after HashSet dedup
        let nprobe_per_token = if self.num_clusters > 100 {
            // Token-level IVF: use configured nprobe per token
            self.nprobe.min(self.num_clusters)
        } else {
            // Document-level IVF: probe sqrt(num_clusters) for backward compatibility
            (self.num_clusters as f32).sqrt().ceil() as usize
        };

        console_log!("🔍 IVF Config: {} clusters, nprobe={} per token ({} query tokens)",
                     self.num_clusters, nprobe_per_token, query_num_tokens);

        // V2: Token-level IVF requires finding clusters for ALL query tokens, not just first
        // For each query token, find its nearest clusters. Then probe the union of all these clusters.
        let mut cluster_set = std::collections::HashSet::new();

        for token_idx in 0..query_num_tokens {
            let token_start = token_idx * self.embedding_dim;
            let query_token = &query_emb[token_start..token_start + self.embedding_dim];

            // Score each cluster by similarity to this query token
            let mut cluster_scores: Vec<(usize, f32)> = (0..self.num_clusters)
                .map(|c| {
                    let centroid_start = c * self.embedding_dim;
                    let centroid = &self.ivf_centroids[centroid_start..centroid_start + self.embedding_dim];

                    let mut score = 0.0;
                    for d in 0..self.embedding_dim {
                        score += query_token[d] * centroid[d];
                    }

                    (c, score)
                })
                .collect();

            cluster_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Add top nprobe_per_token clusters for this query token (PLAID-style)
            for (cluster_id, _score) in cluster_scores.iter().take(nprobe_per_token) {
                cluster_set.insert(*cluster_id);
            }
        }

        // Convert to sorted vec for deterministic behavior
        let top_clusters: Vec<usize> = {
            let mut clusters: Vec<usize> = cluster_set.into_iter().collect();
            clusters.sort();
            clusters
        };

        // Collect candidate documents from top clusters (BASE + DELTAS)
        let mut candidate_docs: Vec<usize> = Vec::new();
        let mut cluster_sizes: Vec<usize> = Vec::new();
        for cluster_id in &top_clusters {
            // Add documents from base IVF
            let base_size = self.ivf_clusters[*cluster_id].len();
            candidate_docs.extend(&self.ivf_clusters[*cluster_id]);

            // Add documents from deltas
            let delta_docs: Vec<usize> = self.ivf_deltas
                .iter()
                .filter(|delta| delta.cluster_id == *cluster_id)
                .map(|delta| delta.doc_id)
                .collect();
            let delta_size = delta_docs.len();
            candidate_docs.extend(delta_docs);

            cluster_sizes.push(base_size + delta_size);
        }

        // Debug: check for duplicate candidates
        let mut sorted_candidates = candidate_docs.clone();
        sorted_candidates.sort();
        let original_count = sorted_candidates.len();
        sorted_candidates.dedup();
        let unique_count = sorted_candidates.len();

        if unique_count != original_count {
            console_log!("⚠️  Warning: {} duplicate candidates found!", original_count - unique_count);
        }

        console_log!("🔍 IVF Search: {} query tokens × {} nprobe/token → {} unique clusters probed",
            query_num_tokens, nprobe_per_token, top_clusters.len());
        console_log!("   Cluster sizes: {:?}", cluster_sizes.iter().take(10).collect::<Vec<_>>());
        console_log!("   Result: {} unique candidates out of {} total docs ({:.1}% recall)",
            unique_count, self.documents.len(), (unique_count as f32 / self.documents.len() as f32) * 100.0);

        // Use unique candidates only to avoid redundant work
        let candidate_docs = sorted_candidates;

        // Calculate MaxSim score for candidate documents only
        let mut scores: Vec<(i64, f32)> = candidate_docs
            .iter()
            .map(|&doc_idx| {
                let doc: &QuantizedDocument = &self.documents[doc_idx];

                // Decode document embeddings
                let doc_embeddings = codec.decode_document(
                    &doc.centroid_codes,
                    &doc.packed_residuals,
                    doc.num_tokens,
                );

                // Calculate MaxSim
                let score = Self::calculate_maxsim(
                    query_emb,
                    &doc_embeddings,
                    query_num_tokens,
                    doc.num_tokens,
                    self.embedding_dim,
                );

                (doc.id, score)
            })
            .collect();

        // Sort and return top-k
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        let passage_ids: Vec<i64> = scores.iter().map(|(id, _)| *id).collect();
        let score_values: Vec<f32> = scores.iter().map(|(_, score)| *score).collect();

        let results = vec![QueryResult {
            query_id: 0,
            passage_ids,
            scores: score_values,
            candidates_searched: candidate_docs.len(),
            num_clusters_probed: top_clusters.len(),
        }];

        serde_json::to_string(&results)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {}", e)))
    }

    /// SIMD-optimized MaxSim calculation (same as unquantized version)
    fn calculate_maxsim(
        query_embeddings: &[f32],
        doc_embeddings: &[f32],
        query_tokens: usize,
        doc_tokens: usize,
        embedding_dim: usize,
    ) -> f32 {
        let mut total_score = 0.0;

        for q in 0..query_tokens {
            let mut max_sim = f32::NEG_INFINITY;

            for d in 0..doc_tokens {
                let q_start = q * embedding_dim;
                let d_start = d * embedding_dim;

                #[cfg(target_arch = "wasm32")]
                let dot_product = unsafe {
                    Self::dot_product_simd(
                        &query_embeddings[q_start..q_start + embedding_dim],
                        &doc_embeddings[d_start..d_start + embedding_dim],
                        embedding_dim
                    )
                };

                #[cfg(not(target_arch = "wasm32"))]
                let dot_product = Self::dot_product_scalar(
                    &query_embeddings[q_start..q_start + embedding_dim],
                    &doc_embeddings[d_start..d_start + embedding_dim],
                    embedding_dim
                );

                max_sim = max_sim.max(dot_product);
            }

            total_score += max_sim;
        }

        total_score
    }

    #[inline]
    #[cfg(target_arch = "wasm32")]
    #[target_feature(enable = "simd128")]
    unsafe fn dot_product_simd(a: &[f32], b: &[f32], len: usize) -> f32 {
        let mut sum = f32x4_splat(0.0);
        let simd_len = (len / 4) * 4;
        let mut i = 0;

        while i < simd_len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            sum = f32x4_add(sum, f32x4_mul(va, vb));
            i += 4;
        }

        let mut result = f32x4_extract_lane::<0>(sum)
                       + f32x4_extract_lane::<1>(sum)
                       + f32x4_extract_lane::<2>(sum)
                       + f32x4_extract_lane::<3>(sum);

        while i < len {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    #[inline]
    fn dot_product_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
        a.iter().zip(b.iter()).take(len).map(|(x, y)| x * y).sum()
    }

    #[wasm_bindgen]
    pub fn get_index_info(&self) -> Result<String, JsValue> {
        let compression_info = if let Some(codec) = &self.codec {
            format!("4-bit quantized with {} centroids", codec.num_centroids)
        } else {
            "Not initialized".to_string()
        };

        let delta_ratio = if self.base_doc_count > 0 {
            (self.ivf_deltas.len() as f32 / self.base_doc_count as f32) * 100.0
        } else {
            0.0
        };

        let info = serde_json::json!({
            "loaded": self.index_loaded,
            "num_documents": self.documents.len(),
            "base_documents": self.base_doc_count,
            "pending_deltas": self.ivf_deltas.len(),
            "delta_ratio_percent": format!("{:.1}", delta_ratio),
            "embedding_dim": self.embedding_dim,
            "quantization": "4-bit residual coding",
            "compression_ratio": "~8x",
            "implementation": compression_info,
            "incremental_updates": "enabled",
        });

        serde_json::to_string(&info)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize info: {}", e)))
    }

    #[wasm_bindgen]
    pub fn get_num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Save the quantized index to binary format
    /// Returns binary data that can be saved to disk
    /// Note: Automatically compacts deltas before saving for optimal performance
    #[wasm_bindgen]
    pub fn save_index(&mut self) -> Result<Vec<u8>, JsValue> {
        if !self.index_loaded {
            return Err(JsValue::from_str("No index loaded"));
        }

        // Compact deltas before saving for optimal load performance
        if !self.ivf_deltas.is_empty() {
            console_log!("💾 Compacting {} deltas before save...", self.ivf_deltas.len());
            self.compact_deltas()?;
        }

        let codec = self.codec.as_ref()
            .ok_or_else(|| JsValue::from_str("Codec not initialized"))?;

        let mut buffer = Vec::new();

        // Write header: magic number, version, metadata
        buffer.extend_from_slice(b"FPQZ"); // FastPlaid Quantized
        buffer.extend_from_slice(&1u32.to_le_bytes()); // Version 1
        buffer.extend_from_slice(&(self.embedding_dim as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.documents.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.num_clusters as u32).to_le_bytes());

        // Write IVF centroids
        for &val in &self.ivf_centroids {
            buffer.extend_from_slice(&val.to_le_bytes());
        }

        // Write IVF cluster mappings
        for cluster in &self.ivf_clusters {
            buffer.extend_from_slice(&(cluster.len() as u32).to_le_bytes());
            for &doc_idx in cluster {
                buffer.extend_from_slice(&(doc_idx as u32).to_le_bytes());
            }
        }

        // Write quantization codec
        buffer.extend_from_slice(&(codec.num_centroids as u32).to_le_bytes());
        for &val in &codec.centroids {
            buffer.extend_from_slice(&val.to_le_bytes());
        }

        // Write avg_residual (Phase 1)
        for &val in &codec.avg_residual {
            buffer.extend_from_slice(&val.to_le_bytes());
        }

        // Write bucket_cutoffs (Phase 2)
        for &val in &codec.bucket_cutoffs {
            buffer.extend_from_slice(&val.to_le_bytes());
        }

        // Write bucket_weights
        for &val in &codec.bucket_weights {
            buffer.extend_from_slice(&val.to_le_bytes());
        }

        // Write quantized documents
        for doc in &self.documents {
            buffer.extend_from_slice(&doc.id.to_le_bytes());
            buffer.extend_from_slice(&(doc.num_tokens as u32).to_le_bytes());

            // Centroid codes
            buffer.extend_from_slice(&(doc.centroid_codes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(&doc.centroid_codes);

            // Packed residuals
            buffer.extend_from_slice(&(doc.packed_residuals.len() as u32).to_le_bytes());
            buffer.extend_from_slice(&doc.packed_residuals);
        }

        console_log!("💾 Index saved: {} bytes ({:.2} MB)", buffer.len(), buffer.len() as f32 / 1_000_000.0);

        Ok(buffer)
    }

    /// Load a precomputed quantized index from binary format
    #[wasm_bindgen]
    pub fn load_index(&mut self, index_bytes: &[u8]) -> Result<(), JsValue> {
        console_log!("📥 Loading precomputed index...");

        let mut offset = 0;

        // Read and verify magic number
        if &index_bytes[offset..offset + 4] != b"FPQZ" {
            return Err(JsValue::from_str("Invalid index file: bad magic number"));
        }
        offset += 4;

        // Read version
        let version = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if version != 1 {
            return Err(JsValue::from_str(&format!("Unsupported index version: {}", version)));
        }

        // Read metadata
        self.embedding_dim = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let num_docs = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        self.num_clusters = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        console_log!("   Embedding dim: {}", self.embedding_dim);
        console_log!("   Documents: {}", num_docs);
        console_log!("   IVF clusters: {}", self.num_clusters);

        // Read IVF centroids
        let centroid_size = self.num_clusters * self.embedding_dim;
        self.ivf_centroids = Vec::with_capacity(centroid_size);
        for _ in 0..centroid_size {
            let val = f32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap());
            self.ivf_centroids.push(val);
            offset += 4;
        }

        // Read IVF cluster mappings
        self.ivf_clusters = Vec::with_capacity(self.num_clusters);
        for _ in 0..self.num_clusters {
            let cluster_size = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let mut cluster = Vec::with_capacity(cluster_size);
            for _ in 0..cluster_size {
                let doc_idx = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
                cluster.push(doc_idx);
                offset += 4;
            }
            self.ivf_clusters.push(cluster);
        }

        // Read quantization codec
        let num_centroids = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let codec_centroid_size = num_centroids * self.embedding_dim;
        let mut codec_centroids = Vec::with_capacity(codec_centroid_size);
        for _ in 0..codec_centroid_size {
            let val = f32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap());
            codec_centroids.push(val);
            offset += 4;
        }

        // Read avg_residual (Phase 1)
        let mut avg_residual = Vec::with_capacity(self.embedding_dim);
        for _ in 0..self.embedding_dim {
            let val = f32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap());
            avg_residual.push(val);
            offset += 4;
        }

        // Read bucket_cutoffs (Phase 2)
        let mut bucket_cutoffs = Vec::with_capacity(17);
        for _ in 0..17 {
            let val = f32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap());
            bucket_cutoffs.push(val);
            offset += 4;
        }

        // Read bucket_weights
        let bucket_weights_size = 16 * self.embedding_dim;
        let mut bucket_weights = Vec::with_capacity(bucket_weights_size);
        for _ in 0..bucket_weights_size {
            let val = f32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap());
            bucket_weights.push(val);
            offset += 4;
        }

        let mut codec = ResidualCodec4bit::new(self.embedding_dim, num_centroids);
        codec.centroids = codec_centroids;
        codec.avg_residual = avg_residual;
        codec.bucket_cutoffs = bucket_cutoffs;
        codec.bucket_weights = bucket_weights;
        self.codec = Some(codec);

        // Read quantized documents
        self.documents = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            let id = i64::from_le_bytes(index_bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let num_tokens = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            // Read centroid codes
            let codes_len = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let centroid_codes = index_bytes[offset..offset + codes_len].to_vec();
            offset += codes_len;

            // Read packed residuals
            let residuals_len = u32::from_le_bytes(index_bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let packed_residuals = index_bytes[offset..offset + residuals_len].to_vec();
            offset += residuals_len;

            self.documents.push(QuantizedDocument {
                id,
                centroid_codes,
                packed_residuals,
                num_tokens,
            });
        }

        self.index_loaded = true;

        // Initialize delta tracking for loaded index
        self.base_doc_count = num_docs;
        self.ivf_deltas.clear();

        console_log!("✅ Index loaded successfully!");
        console_log!("   Total size: {:.2} MB", index_bytes.len() as f32 / 1_000_000.0);
        console_log!("   Ready for incremental updates!");

        Ok(())
    }

    /// Manually trigger compaction of deltas into base IVF
    /// Useful for forcing compaction before save or for performance tuning
    #[wasm_bindgen]
    pub fn compact_index(&mut self) -> Result<(), JsValue> {
        if !self.index_loaded {
            return Err(JsValue::from_str("No index loaded"));
        }
        self.compact_deltas()
    }

    /// Compact deltas into base IVF index
    /// Merges pending delta additions into the base ivf_clusters
    fn compact_deltas(&mut self) -> Result<(), JsValue> {
        if self.ivf_deltas.is_empty() {
            console_log!("   No deltas to compact");
            return Ok(());
        }

        console_log!("   🔨 Compacting {} deltas into base IVF...", self.ivf_deltas.len());

        // Merge deltas into base IVF clusters
        for delta in &self.ivf_deltas {
            self.ivf_clusters[delta.cluster_id].push(delta.doc_id);
        }

        // Clear deltas
        let num_compacted = self.ivf_deltas.len();
        self.ivf_deltas.clear();

        // Update base document count
        self.base_doc_count = self.documents.len();

        console_log!("   ✅ Compacted {} deltas, base now has {} documents", num_compacted, self.base_doc_count);

        Ok(())
    }

    /// Build IVF (Inverted File Index) for fast approximate search
    /// Uses first token of each document as representative
    fn build_ivf_index(
        &mut self,
        embeddings_data: &[f32],
        doc_info: &[i64],
        num_docs: usize,
    ) -> Result<(), JsValue> {
        // V2: Extract ALL TOKENS first to calculate cluster count
        let total_tokens: usize = (0..num_docs)
            .map(|i| doc_info[i * 2 + 1] as usize)
            .sum();

        // V2: Determine number of clusters based on TOTAL TOKENS (not documents)
        // sqrt(N) heuristic: sqrt(270k) ≈ 520 clusters for proper token-level selectivity
        self.num_clusters = (total_tokens as f32).sqrt().ceil() as usize;
        self.num_clusters = self.num_clusters.max(100).min(1000); // Reasonable bounds: 100-1000 clusters

        console_log!("   V2: Creating {} IVF clusters for {} tokens from {} documents...",
                     self.num_clusters, total_tokens, num_docs);

        let mut all_tokens = Vec::with_capacity(total_tokens);
        let mut token_to_doc: Vec<usize> = Vec::with_capacity(total_tokens);
        let mut offset = 0;

        for doc_idx in 0..num_docs {
            let num_tokens = doc_info[doc_idx * 2 + 1] as usize;
            let doc_emb = &embeddings_data[offset..offset + num_tokens * self.embedding_dim];

            // Collect ALL tokens from this document
            for token_idx in 0..num_tokens {
                let token_start = token_idx * self.embedding_dim;
                let token = &doc_emb[token_start..token_start + self.embedding_dim];
                all_tokens.extend_from_slice(token);
                token_to_doc.push(doc_idx);
            }

            offset += num_tokens * self.embedding_dim;
        }

        // K-means clustering on ALL tokens (but same number of clusters as before)
        self.ivf_centroids = vec![0.0; self.num_clusters * self.embedding_dim];
        self.ivf_clusters = vec![Vec::new(); self.num_clusters];

        // Initialize centroids: pick evenly spaced tokens
        for c in 0..self.num_clusters {
            let token_idx = (c * total_tokens) / self.num_clusters;
            let token_start = token_idx * self.embedding_dim;
            let centroid_start = c * self.embedding_dim;
            self.ivf_centroids[centroid_start..centroid_start + self.embedding_dim]
                .copy_from_slice(&all_tokens[token_start..token_start + self.embedding_dim]);
        }

        // Run k-means iterations on ALL tokens
        for _iteration in 0..10 {
            // Clear clusters
            for cluster in &mut self.ivf_clusters {
                cluster.clear();
            }

            // Assign each TOKEN to nearest cluster
            // Then map token → doc and add doc to cluster
            let mut cluster_doc_sets: Vec<std::collections::HashSet<usize>> =
                vec![std::collections::HashSet::new(); self.num_clusters];

            for token_idx in 0..total_tokens {
                let token_start = token_idx * self.embedding_dim;
                let token = &all_tokens[token_start..token_start + self.embedding_dim];
                let nearest_cluster = self.find_nearest_ivf_cluster(token);
                let doc_idx = token_to_doc[token_idx];

                // Add document to this cluster (HashSet ensures no duplicates)
                cluster_doc_sets[nearest_cluster].insert(doc_idx);
            }

            // Convert HashSets to Vecs
            for (cluster_id, doc_set) in cluster_doc_sets.iter().enumerate() {
                self.ivf_clusters[cluster_id] = doc_set.iter().copied().collect();
            }

            // Update centroids by averaging assigned TOKENS
            let mut new_centroids = vec![0.0; self.num_clusters * self.embedding_dim];
            let mut centroid_counts = vec![0usize; self.num_clusters];

            for token_idx in 0..total_tokens {
                let token_start = token_idx * self.embedding_dim;
                let token = &all_tokens[token_start..token_start + self.embedding_dim];
                let nearest_cluster = self.find_nearest_ivf_cluster(token);

                let centroid_start = nearest_cluster * self.embedding_dim;
                for d in 0..self.embedding_dim {
                    new_centroids[centroid_start + d] += token[d];
                }
                centroid_counts[nearest_cluster] += 1;
            }

            // Average
            for c in 0..self.num_clusters {
                if centroid_counts[c] > 0 {
                    let centroid_start = c * self.embedding_dim;
                    let count = centroid_counts[c] as f32;
                    for d in 0..self.embedding_dim {
                        new_centroids[centroid_start + d] /= count;
                    }
                }
            }

            self.ivf_centroids = new_centroids;
        }

        // Log cluster statistics with distribution
        let avg_docs_per_cluster = num_docs as f32 / self.num_clusters as f32;
        let mut cluster_sizes: Vec<usize> = self.ivf_clusters.iter().map(|c| c.len()).collect();
        cluster_sizes.sort();
        let min_size = cluster_sizes.first().copied().unwrap_or(0);
        let max_size = cluster_sizes.last().copied().unwrap_or(0);
        let median_size = cluster_sizes.get(cluster_sizes.len() / 2).copied().unwrap_or(0);

        console_log!("   ✅ IVF index built:");
        console_log!("      {} clusters, avg {:.1} docs/cluster", self.num_clusters, avg_docs_per_cluster);
        console_log!("      Distribution: min={}, median={}, max={}", min_size, median_size, max_size);

        // Count empty clusters
        let empty_clusters = self.ivf_clusters.iter().filter(|c| c.is_empty()).count();
        if empty_clusters > 0 {
            console_log!("      ⚠️  {} empty clusters!", empty_clusters);
        }

        Ok(())
    }

    /// Find nearest IVF cluster for a query/document representative
    fn find_nearest_ivf_cluster(&self, query_rep: &[f32]) -> usize {
        let mut best_cluster = 0;
        let mut best_score = f32::NEG_INFINITY;

        for c in 0..self.num_clusters {
            let centroid_start = c * self.embedding_dim;
            let centroid = &self.ivf_centroids[centroid_start..centroid_start + self.embedding_dim];

            let mut score = 0.0;
            for d in 0..self.embedding_dim {
                score += query_rep[d] * centroid[d];
            }

            if score > best_score {
                best_score = score;
                best_cluster = c;
            }
        }

        best_cluster
    }
}

// Note: The WASM start function is defined in lib_wasm.rs to avoid duplicate symbols
// Both modules share the same initialization
