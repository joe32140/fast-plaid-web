// WASM-specific implementation of FastPlaid
// This is a simplified version focused on browser compatibility
// Note: This version doesn't use Candle to avoid dependency conflicts

use wasm_bindgen::prelude::*;
use js_sys::{Array, Uint8Array};
use web_sys::console;
use serde::{Serialize, Deserialize};
use serde_wasm_bindgen;

// Utility function for logging to browser console
macro_rules! console_log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()))
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

/// WASM wrapper for FastPlaid search functionality
#[wasm_bindgen]
pub struct FastPlaidWasm {
    index_loaded: bool,
    embedding_dim: usize,
}

#[wasm_bindgen]
impl FastPlaidWasm {
    /// Creates a new FastPlaidWasm instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<FastPlaidWasm, JsValue> {
        // Set panic hook for better error messages in browser
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        
        console_log!("Initializing FastPlaid WASM for mxbai-edge-colbert...");
        
        Ok(FastPlaidWasm {
            index_loaded: false,
            embedding_dim: 384, // Default for mxbai-edge-colbert-v0-17m
        })
    }

    /// Loads a FastPlaid index from bytes
    #[wasm_bindgen]
    pub fn load_index(&mut self, index_bytes: &[u8]) -> Result<(), JsValue> {
        console_log!("Loading index from {} bytes", index_bytes.len());
        
        // TODO: Implement actual index loading
        // For now, just mark as loaded for demo purposes
        self.index_loaded = true;
        
        console_log!("Index loaded successfully (demo mode)");
        Ok(())
    }

    /// Searches the loaded index with query embeddings
    /// 
    /// # Arguments
    /// * `query_embeddings` - Flat array of f32 embeddings from mxbai-edge-colbert
    /// * `query_shape` - Shape of the query tensor [batch_size, seq_len, embedding_dim]
    /// * `top_k` - Number of top results to return
    /// * `n_ivf_probe` - Number of IVF cells to probe (optional, defaults to 10)
    #[wasm_bindgen]
    pub fn search(
        &self,
        query_embeddings: &[f32],
        query_shape: &[usize],
        top_k: usize,
        n_ivf_probe: Option<usize>,
    ) -> Result<JsValue, JsValue> {
        if !self.index_loaded {
            return Err(JsValue::from_str("No index loaded. Call load_index() first."));
        }

        console_log!(
            "Searching with {} embeddings, shape: {:?}, top_k={}",
            query_embeddings.len(),
            query_shape,
            top_k
        );

        // Validate input shape
        if query_shape.len() != 3 {
            return Err(JsValue::from_str("Query shape must be 3D: [batch_size, seq_len, embedding_dim]"));
        }

        let expected_size = query_shape.iter().product::<usize>();
        if query_embeddings.len() != expected_size {
            return Err(JsValue::from_str(&format!(
                "Embedding size mismatch: got {}, expected {}",
                query_embeddings.len(),
                expected_size
            )));
        }

        // TODO: Implement actual search logic with Candle tensors
        // For now, return demo results
        let demo_results = vec![QueryResult {
            query_id: 0,
            passage_ids: vec![1, 2, 3, 4, 5],
            scores: vec![0.9234, 0.8876, 0.8654, 0.8432, 0.8123],
        }];

        // Limit to top_k results
        let mut results = demo_results;
        if let Some(result) = results.get_mut(0) {
            result.passage_ids.truncate(top_k);
            result.scores.truncate(top_k);
        }

        // Convert results to JavaScript
        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize results: {}", e)))
    }

    /// Gets information about the loaded index
    #[wasm_bindgen]
    pub fn get_index_info(&self) -> Result<JsValue, JsValue> {
        let info = serde_json::json!({
            "loaded": self.index_loaded,
            "embedding_dim": self.embedding_dim,
            "device": "cpu",
            "model_target": "mixedbread-ai/mxbai-edge-colbert-v0-17m",
            "status": if self.index_loaded { "ready" } else { "no_index" }
        });

        serde_wasm_bindgen::to_value(&info)
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
    console_log!("FastPlaid WASM module initialized for mxbai-edge-colbert-v0-17m");
}