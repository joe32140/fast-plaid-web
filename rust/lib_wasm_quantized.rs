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

        // After training centroids, learn optimal bucket weights from residual distribution
        self.train_bucket_weights(embeddings, num_tokens);

        console_log!("✅ Trained {} centroids with {} iterations", self.num_centroids, num_iters);
    }

    /// Train bucket weights based on actual residual distribution
    fn train_bucket_weights(&mut self, embeddings: &[f32], num_tokens: usize) {
        // Collect all residuals to compute statistics
        let mut all_residuals = Vec::with_capacity(num_tokens * self.embedding_dim);

        for token_idx in 0..num_tokens {
            let token_start = token_idx * self.embedding_dim;
            let token_emb = &embeddings[token_start..token_start + self.embedding_dim];

            let centroid_idx = self.find_nearest_centroid(token_emb);
            let centroid_start = centroid_idx * self.embedding_dim;
            let centroid = &self.centroids[centroid_start..centroid_start + self.embedding_dim];

            for d in 0..self.embedding_dim {
                let residual = token_emb[d] - centroid[d];
                all_residuals.push(residual);
            }
        }

        // Compute percentiles for bucket boundaries (16 buckets = 15 boundaries)
        all_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // For each bucket, compute the average residual value in that quantile
        for bucket in 0..16 {
            let start_pct = bucket as f32 / 16.0;
            let end_pct = (bucket + 1) as f32 / 16.0;

            let start_idx = (start_pct * all_residuals.len() as f32) as usize;
            let end_idx = (end_pct * all_residuals.len() as f32) as usize;

            let bucket_residuals = &all_residuals[start_idx..end_idx];
            let avg_residual: f32 = if bucket_residuals.is_empty() {
                0.0
            } else {
                bucket_residuals.iter().sum::<f32>() / bucket_residuals.len() as f32
            };

            // Update bucket weights for all dimensions with this average
            for d in 0..self.embedding_dim {
                let weight_idx = bucket * self.embedding_dim + d;
                self.bucket_weights[weight_idx] = avg_residual;
            }
        }

        // Log residual statistics
        let min_res = all_residuals.first().copied().unwrap_or(0.0);
        let max_res = all_residuals.last().copied().unwrap_or(0.0);
        console_log!("   Residual range: [{:.4}, {:.4}]", min_res, max_res);
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
    /// Finds the bucket with the closest learned weight
    fn quantize_to_4bit(&self, value: f32) -> u8 {
        let mut best_bucket = 0u8;
        let mut best_diff = f32::INFINITY;

        // Find the bucket whose learned weight is closest to this value
        // Note: all dimensions share the same bucket weights, so we just check bucket 0's embedding_dim
        for bucket in 0..16 {
            let weight_idx = bucket * self.embedding_dim; // Just check first dimension
            let weight = self.bucket_weights[weight_idx];
            let diff = (value - weight).abs();

            if diff < best_diff {
                best_diff = diff;
                best_bucket = bucket as u8;
            }
        }

        best_bucket
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
                let residual = token_emb[d] - centroid[d];
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

                // Add centroid + residual (no scaling!)
                embeddings.push(centroid[d] + residual);
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

/// WASM wrapper for quantized FastPlaid
#[wasm_bindgen]
pub struct FastPlaidQuantized {
    index_loaded: bool,
    embedding_dim: usize,
    documents: Vec<QuantizedDocument>,
    codec: Option<ResidualCodec4bit>,
    // IVF (Inverted File Index) for fast approximate search
    ivf_clusters: Vec<Vec<usize>>,  // cluster_id -> [doc_indices]
    ivf_centroids: Vec<f32>,         // [num_clusters, embedding_dim]
    num_clusters: usize,
}

#[wasm_bindgen]
impl FastPlaidQuantized {
    /// Creates a new quantized FastPlaid instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<FastPlaidQuantized, JsValue> {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        console_log!("🚀 Initializing FastPlaid WASM with 4-bit quantization...");
        console_log!("   🗜️ 8x compression vs f32 - fit more papers in GitHub Pages!");

        Ok(FastPlaidQuantized {
            index_loaded: false,
            embedding_dim: 384,
            documents: Vec::new(),
            codec: None,
            ivf_clusters: Vec::new(),
            ivf_centroids: Vec::new(),
            num_clusters: 0,
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

        Ok(())
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

        // IVF Search: Find candidate documents from top clusters
        // NOTE: Native FastPlaid uses token-level IVF (more accurate but complex)
        // WASM uses document-level IVF which is simpler but less accurate
        // For now, disable IVF by probing all clusters to verify quantization quality
        let num_probe_clusters = self.num_clusters; // Disable IVF filtering

        // Compute query representative (average of all tokens)
        // Since centroids are averages, we compare with query average
        let mut query_rep = vec![0.0; self.embedding_dim];
        for t in 0..query_num_tokens {
            for d in 0..self.embedding_dim {
                query_rep[d] += query_emb[t * self.embedding_dim + d];
            }
        }
        for d in 0..self.embedding_dim {
            query_rep[d] /= query_num_tokens as f32;
        }

        // Score each cluster by similarity to query representative
        let mut cluster_scores: Vec<(usize, f32)> = (0..self.num_clusters)
            .map(|c| {
                let centroid_start = c * self.embedding_dim;
                let centroid = &self.ivf_centroids[centroid_start..centroid_start + self.embedding_dim];

                let mut score = 0.0;
                for d in 0..self.embedding_dim {
                    score += query_rep[d] * centroid[d];
                }

                (c, score)
            })
            .collect();

        cluster_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_clusters: Vec<usize> = cluster_scores
            .iter()
            .take(num_probe_clusters)
            .map(|(c, _)| *c)
            .collect();

        // Collect candidate documents from top clusters
        let mut candidate_docs: Vec<usize> = Vec::new();
        for cluster_id in &top_clusters {
            candidate_docs.extend(&self.ivf_clusters[*cluster_id]);
        }

        // Debug: check for duplicate candidates
        let mut sorted_candidates = candidate_docs.clone();
        sorted_candidates.sort();
        sorted_candidates.dedup();
        if sorted_candidates.len() != candidate_docs.len() {
            console_log!("⚠️  Warning: {} duplicate candidates found!", candidate_docs.len() - sorted_candidates.len());
        }

        console_log!("🔍 IVF Search: Probing {} clusters, {} candidates out of {} total docs",
            num_probe_clusters, candidate_docs.len(), self.documents.len());

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
            num_clusters_probed: num_probe_clusters,
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

        let info = serde_json::json!({
            "loaded": self.index_loaded,
            "num_documents": self.documents.len(),
            "embedding_dim": self.embedding_dim,
            "quantization": "4-bit residual coding",
            "compression_ratio": "~8x",
            "implementation": compression_info,
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
    #[wasm_bindgen]
    pub fn save_index(&self) -> Result<Vec<u8>, JsValue> {
        if !self.index_loaded {
            return Err(JsValue::from_str("No index loaded"));
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

        let mut codec = ResidualCodec4bit::new(self.embedding_dim, num_centroids);
        codec.centroids = codec_centroids;
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

        console_log!("✅ Index loaded successfully!");
        console_log!("   Total size: {:.2} MB", index_bytes.len() as f32 / 1_000_000.0);

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
        // Determine number of clusters (sqrt(N) is a common heuristic)
        self.num_clusters = (num_docs as f32).sqrt().ceil() as usize;
        self.num_clusters = self.num_clusters.max(10).min(100); // Between 10-100 clusters

        console_log!("   Creating {} IVF clusters...", self.num_clusters);

        // Extract document representatives (average of all tokens)
        // Using average instead of first token gives better clustering quality
        let mut doc_reps = Vec::with_capacity(num_docs);
        let mut offset = 0;

        for i in 0..num_docs {
            let num_tokens = doc_info[i * 2 + 1] as usize;
            let doc_emb = &embeddings_data[offset..offset + num_tokens * self.embedding_dim];

            // Compute average of all tokens
            let mut avg = vec![0.0; self.embedding_dim];
            for t in 0..num_tokens {
                for d in 0..self.embedding_dim {
                    avg[d] += doc_emb[t * self.embedding_dim + d];
                }
            }
            for d in 0..self.embedding_dim {
                avg[d] /= num_tokens as f32;
            }

            doc_reps.push(avg);
            offset += num_tokens * self.embedding_dim;
        }

        // K-means clustering
        self.ivf_centroids = vec![0.0; self.num_clusters * self.embedding_dim];
        self.ivf_clusters = vec![Vec::new(); self.num_clusters];

        // Initialize centroids: pick evenly spaced documents
        for c in 0..self.num_clusters {
            let doc_idx = (c * num_docs) / self.num_clusters;
            let src_start = c * self.embedding_dim;
            self.ivf_centroids[src_start..src_start + self.embedding_dim]
                .copy_from_slice(&doc_reps[doc_idx]);
        }

        // Run k-means iterations
        for _iteration in 0..10 {
            // Clear clusters
            for cluster in &mut self.ivf_clusters {
                cluster.clear();
            }

            // Assign documents to nearest cluster
            for doc_idx in 0..num_docs {
                let doc_rep = &doc_reps[doc_idx];
                let nearest_cluster = self.find_nearest_ivf_cluster(doc_rep);
                self.ivf_clusters[nearest_cluster].push(doc_idx);
            }

            // Update centroids
            let mut new_centroids = vec![0.0; self.num_clusters * self.embedding_dim];
            for (cluster_id, doc_indices) in self.ivf_clusters.iter().enumerate() {
                if doc_indices.is_empty() {
                    continue;
                }

                let centroid_start = cluster_id * self.embedding_dim;
                for &doc_idx in doc_indices {
                    for d in 0..self.embedding_dim {
                        new_centroids[centroid_start + d] += doc_reps[doc_idx][d];
                    }
                }

                // Average
                let count = doc_indices.len() as f32;
                for d in 0..self.embedding_dim {
                    new_centroids[centroid_start + d] /= count;
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
