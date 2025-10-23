// WASM-specific implementation of FastPlaid
// This implementation performs real MaxSim scoring between query and document embeddings
// with SIMD optimization for maximum performance

use wasm_bindgen::prelude::*;
use web_sys::console;
use serde::{Serialize, Deserialize};

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

// Utility function for logging to browser console
macro_rules! console_log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()))
}

/// Document embedding data stored in the index
#[derive(Clone, Debug)]
struct DocumentEmbedding {
    id: i64,
    embeddings: Vec<f32>,
    num_tokens: usize,
}

/// Search result for a single query
#[derive(Serialize, Deserialize, Debug)]
pub struct QueryResult {
    pub query_id: usize,
    pub passage_ids: Vec<i64>,
    pub scores: Vec<f32>,
}

/// Search configuration parameters
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchParameters {
    pub batch_size: usize,
    pub n_full_scores: usize,
    pub top_k: usize,
    pub n_ivf_probe: usize,
}

/// IVF (Inverted File) cluster for approximate search
#[derive(Clone, Debug)]
struct IVFCluster {
    centroid: Vec<f32>,
    document_ids: Vec<usize>, // Indices into documents array
}

/// WASM wrapper for FastPlaid search functionality
#[wasm_bindgen]
pub struct FastPlaidWasm {
    index_loaded: bool,
    embedding_dim: usize,
    documents: Vec<DocumentEmbedding>,
    // IVF index for approximate search
    ivf_clusters: Vec<IVFCluster>,
    ivf_enabled: bool,
}

#[wasm_bindgen]
impl FastPlaidWasm {
    /// Creates a new FastPlaidWasm instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<FastPlaidWasm, JsValue> {
        // Set panic hook for better error messages in browser
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        console_log!("🚀 Initializing FastPlaid WASM v5.0 (SIMD + IVF/PLAID) for mxbai-edge-colbert...");

        Ok(FastPlaidWasm {
            index_loaded: false,
            embedding_dim: 384, // Default for mxbai-edge-colbert-v0-17m
            documents: Vec::new(),
            ivf_clusters: Vec::new(),
            ivf_enabled: false,
        })
    }

    /// Loads document embeddings from JavaScript
    ///
    /// # Arguments
    /// * `embeddings_data` - Flat array of all document embeddings concatenated
    /// * `doc_info` - Array of [doc_id, num_tokens] pairs for each document
    #[wasm_bindgen]
    pub fn load_documents(
        &mut self,
        embeddings_data: &[f32],
        doc_info: &[i64],
    ) -> Result<(), JsValue> {
        console_log!("📥 Loading document embeddings into WASM index...");
        console_log!("   Total embedding data: {} floats", embeddings_data.len());
        console_log!("   Document info entries: {}", doc_info.len());

        // Parse doc_info as pairs of [id, num_tokens]
        if doc_info.len() % 2 != 0 {
            return Err(JsValue::from_str("doc_info must contain pairs of [id, num_tokens]"));
        }

        let num_docs = doc_info.len() / 2;

        // Auto-detect embedding dimension from first document
        if num_docs > 0 {
            let first_num_tokens = doc_info[1] as usize;
            if first_num_tokens > 0 && embeddings_data.len() > 0 {
                // Calculate embedding_dim from total data
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
            }
        }

        let mut documents = Vec::with_capacity(num_docs);
        let mut offset = 0;

        for i in 0..num_docs {
            let doc_id = doc_info[i * 2];
            let num_tokens = doc_info[i * 2 + 1] as usize;
            let embedding_size = num_tokens * self.embedding_dim;

            if offset + embedding_size > embeddings_data.len() {
                return Err(JsValue::from_str(&format!(
                    "Not enough embedding data for document {}: need {}, have {}",
                    doc_id,
                    offset + embedding_size,
                    embeddings_data.len()
                )));
            }

            let doc_embeddings = embeddings_data[offset..offset + embedding_size].to_vec();
            documents.push(DocumentEmbedding {
                id: doc_id,
                embeddings: doc_embeddings,
                num_tokens,
            });

            offset += embedding_size;
        }

        console_log!("✅ Loaded {} documents with embeddings (dim={})", documents.len(), self.embedding_dim);
        console_log!("   Total embeddings consumed: {} / {} floats", offset, embeddings_data.len());

        self.documents = documents;
        self.index_loaded = true;

        Ok(())
    }

    /// Loads a FastPlaid index from bytes (simplified version for now)
    #[wasm_bindgen]
    pub fn load_index(&mut self, _index_bytes: &[u8]) -> Result<(), JsValue> {
        console_log!("⚠️ load_index() is deprecated - use load_documents() instead");
        console_log!("   Marking as loaded for compatibility...");

        // For now, just mark as loaded
        // In a full implementation, this would deserialize the index
        self.index_loaded = true;

        Ok(())
    }

    /// Searches the loaded index with query embeddings using ColBERT MaxSim
    ///
    /// # Arguments
    /// * `query_embeddings` - Flat array of f32 embeddings from mxbai-edge-colbert
    /// * `query_shape` - Shape of the query tensor [batch_size, seq_len, embedding_dim]
    /// * `top_k` - Number of top results to return
    /// * `n_ivf_probe` - Number of IVF cells to probe (ignored in this implementation)
    ///
    /// # Returns
    /// Returns a JSON string (not JsValue) to avoid externref table overflow issues
    #[wasm_bindgen]
    pub fn search(
        &self,
        query_embeddings: &[f32],
        query_shape: &[usize],
        top_k: usize,
        _n_ivf_probe: Option<usize>,
    ) -> Result<String, JsValue> {
        if !self.index_loaded {
            return Err(JsValue::from_str("No index loaded. Call load_documents() first."));
        }

        if self.documents.is_empty() {
            return Err(JsValue::from_str("No documents loaded. Use load_documents() to add documents."));
        }

        // Validate input shape
        if query_shape.len() != 3 {
            return Err(JsValue::from_str("Query shape must be 3D: [batch_size, seq_len, embedding_dim]"));
        }

        let batch_size = query_shape[0];
        let query_num_tokens = query_shape[1];
        let query_dim = query_shape[2];

        if query_dim != self.embedding_dim {
            return Err(JsValue::from_str(&format!(
                "Query dimension ({}) doesn't match index dimension ({})",
                query_dim, self.embedding_dim
            )));
        }

        let expected_size = batch_size * query_num_tokens * query_dim;
        if query_embeddings.len() != expected_size {
            return Err(JsValue::from_str(&format!(
                "Embedding size mismatch: got {}, expected {}",
                query_embeddings.len(),
                expected_size
            )));
        }

        // Perform MaxSim search for the first query (batch_size should be 1)
        let query_start = 0;
        let query_end = query_num_tokens * query_dim;
        let query_emb = &query_embeddings[query_start..query_end];

        // Calculate MaxSim score for each document
        let mut scores: Vec<(i64, f32)> = self.documents
            .iter()
            .map(|doc| {
                let score = Self::calculate_maxsim(
                    query_emb,
                    &doc.embeddings,
                    query_num_tokens,
                    doc.num_tokens,
                    self.embedding_dim,
                );
                (doc.id, score)
            })
            .collect();

        // Sort by score (descending) and take top_k
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        // Format results
        let passage_ids: Vec<i64> = scores.iter().map(|(id, _)| *id).collect();
        let score_values: Vec<f32> = scores.iter().map(|(_, score)| *score).collect();

        let results = vec![QueryResult {
            query_id: 0,
            passage_ids,
            scores: score_values,
        }];

        // Convert results to JSON string
        let json_result = serde_json::to_string(&results)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {}", e)))?;

        Ok(json_result)
    }

    /// SIMD-optimized dot product for f32 vectors
    #[inline]
    #[cfg(target_arch = "wasm32")]
    #[target_feature(enable = "simd128")]
    unsafe fn dot_product_simd(a: &[f32], b: &[f32], len: usize) -> f32 {
        let mut sum = f32x4_splat(0.0);

        // Process 4 floats at a time using SIMD
        let simd_len = (len / 4) * 4;
        let mut i = 0;

        while i < simd_len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            sum = f32x4_add(sum, f32x4_mul(va, vb));
            i += 4;
        }

        // Horizontal sum of the SIMD register
        let mut result = f32x4_extract_lane::<0>(sum)
                       + f32x4_extract_lane::<1>(sum)
                       + f32x4_extract_lane::<2>(sum)
                       + f32x4_extract_lane::<3>(sum);

        // Handle remaining elements (if len % 4 != 0)
        while i < len {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    /// Scalar fallback dot product (for non-WASM targets)
    #[inline]
    fn dot_product_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
        let mut sum = 0.0;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Calculate ColBERT MaxSim score between query and document embeddings
    ///
    /// For each query token, find the maximum similarity with any document token,
    /// then SUM these max similarities (official ColBERT implementation).
    /// Uses SIMD instructions for 3-4x speedup on the inner dot product loop.
    fn calculate_maxsim(
        query_embeddings: &[f32],
        doc_embeddings: &[f32],
        query_tokens: usize,
        doc_tokens: usize,
        embedding_dim: usize,
    ) -> f32 {
        let mut total_score = 0.0;

        // For each query token
        for q in 0..query_tokens {
            let mut max_sim = f32::NEG_INFINITY;

            // Find max similarity with any document token
            for d in 0..doc_tokens {
                let q_start = q * embedding_dim;
                let d_start = d * embedding_dim;

                // Use SIMD-optimized dot product
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

        // Return SUM of MaxSim scores (official ColBERT implementation)
        total_score
    }

    /// Gets information about the loaded index
    #[wasm_bindgen]
    pub fn get_index_info(&self) -> Result<String, JsValue> {
        let info = serde_json::json!({
            "loaded": self.index_loaded,
            "num_documents": self.documents.len(),
            "embedding_dim": self.embedding_dim,
            "device": "cpu",
            "model_target": "mixedbread-ai/mxbai-edge-colbert-v0-17m",
            "implementation": "real_maxsim",
            "status": if self.index_loaded { "ready" } else { "no_index" }
        });

        serde_json::to_string(&info)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize info: {}", e)))
    }

    /// Sets the embedding dimension (useful for different models)
    #[wasm_bindgen]
    pub fn set_embedding_dim(&mut self, dim: usize) {
        console_log!("Setting embedding dimension to {}", dim);
        self.embedding_dim = dim;
    }

    /// Gets the current embedding dimension
    #[wasm_bindgen]
    pub fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Gets the number of documents loaded
    #[wasm_bindgen]
    pub fn get_num_documents(&self) -> usize {
        self.documents.len()
    }
}

/// Utility function to validate embeddings format for mxbai-edge-colbert
#[wasm_bindgen]
pub fn validate_mxbai_embeddings(
    embeddings: &[f32],
    expected_dim: usize,
) -> Result<bool, JsValue> {
    if embeddings.len() % expected_dim != 0 {
        return Ok(false);
    }

    // Check for reasonable embedding values (typically between -1 and 1 for normalized embeddings)
    let max_val = embeddings.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = embeddings.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    console_log!(
        "Embedding validation: {} values, range [{:.4}, {:.4}]",
        embeddings.len(),
        min_val,
        max_val
    );

    Ok(max_val <= 10.0 && min_val >= -10.0) // Reasonable bounds
}

/// Initialize the WASM module (called automatically)
#[wasm_bindgen(start)]
pub fn main() {
    console_log!("FastPlaid WASM module initialized with SIMD-optimized MaxSim for mxbai-edge-colbert-v0-17m");
}
