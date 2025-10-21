// Conditional compilation for WASM vs native
#[cfg(target_arch = "wasm32")]
pub use crate::lib_wasm::*;

#[cfg(target_arch = "wasm32")]
mod lib_wasm;

#[cfg(not(target_arch = "wasm32"))]
pub mod index;
#[cfg(not(target_arch = "wasm32"))]
pub mod search;
#[cfg(not(target_arch = "wasm32"))]
pub mod utils;

// Native implementation (non-WASM)
#[cfg(not(target_arch = "wasm32"))]
mod native_impl {
    // External crate imports.
    use anyhow::anyhow;
    use candle_core::Device;

    // Internal module imports.
    use crate::index::create::create_index;
    use crate::index::update::update_index;
    use crate::index::delete::delete_from_index;
    use crate::search::load::load_index;
    use crate::search::search::{search_index, QueryResult, SearchParameters};

    /// Parses a string identifier into a `candle_core::Device`.
    fn get_device(device: &str) -> Result<Device, anyhow::Error> {
        match device.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Ok(Device::new_cuda(0)?),
            #[cfg(feature = "cuda")]
            s if s.starts_with("cuda:") => {
                let parts: Vec<&str> = s.split(':').collect();
                if parts.len() == 2 {
                    let device_id = parts[1].parse::<usize>()
                        .map_err(|_| anyhow!("Invalid CUDA device index: '{}'", parts[1]))?;
                    Ok(Device::new_cuda(device_id)?)
                } else {
                    Err(anyhow!("Invalid CUDA device format. Expected 'cuda:N'."))
                }
            },
            _ => Err(anyhow!("Unsupported device string: '{}'", device)),
        }
    }

    /// Initialize the library (no-op for Candle-based implementation).
    pub fn initialize() -> Result<(), anyhow::Error> {
        Ok(())
    }

    /// Creates and saves a new FastPlaid index to disk.
    pub fn create_index_from_embeddings(
        index_path: &str,
        device_str: &str,
        embedding_dim: i64,
        nbits: i64,
        embeddings: Vec<candle_core::Tensor>,
        centroids: candle_core::Tensor,
        seed: Option<u64>,
    ) -> Result<(), anyhow::Error> {
        let device = get_device(device_str)?;
        
        let embeddings: Result<Vec<_>, _> = embeddings
            .into_iter()
            .map(|tensor| tensor.to_device(&device)?.to_dtype(candle_core::DType::F16))
            .collect();
        let embeddings = embeddings?;
        
        let centroids = centroids.to_device(&device)?.to_dtype(candle_core::DType::F16)?;

        create_index(&embeddings, index_path, embedding_dim, nbits, device, centroids, seed)
    }

    /// Updates an existing FastPlaid index with new documents.
    pub fn update_index_with_embeddings(
        index_path: &str,
        device_str: &str,
        embeddings: Vec<candle_core::Tensor>,
    ) -> Result<(), anyhow::Error> {
        let device = get_device(device_str)?;

        let embeddings: Result<Vec<_>, _> = embeddings
            .into_iter()
            .map(|tensor| tensor.to_device(&device)?.to_dtype(candle_core::DType::F16))
            .collect();
        let embeddings = embeddings?;

        update_index(&embeddings, index_path, device)
    }

    /// Loads an index and performs a search in a single, high-level operation.
    pub fn load_and_search_index(
        index_path: &str,
        device_str: &str,
        queries_embeddings: candle_core::Tensor,
        search_parameters: &SearchParameters,
        show_progress: bool,
        subset: Option<Vec<Vec<i64>>>,
    ) -> Result<Vec<QueryResult>, anyhow::Error> {
        let query_tensor = queries_embeddings
            .to_device(&Device::Cpu)?
            .to_dtype(candle_core::DType::F16)?;

        let device = get_device(device_str)?;
        let loaded_index = load_index(index_path, device.clone())?;

        let results = search_index(
            &query_tensor,
            &loaded_index,
            search_parameters,
            device,
            show_progress,
            subset,
        )?;

        Ok(results)
    }

    /// Deletes documents from an existing index.
    pub fn delete_from_index_by_ids(
        index_path: &str,
        device_str: &str,
        subset: Vec<i64>,
    ) -> Result<(), anyhow::Error> {
        let device = get_device(device_str)?;
        delete_from_index(&subset, index_path, device)
    }
}

// Export native functions for non-WASM builds
#[cfg(not(target_arch = "wasm32"))]
pub use native_impl::*;


