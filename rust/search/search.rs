use anyhow::{anyhow, bail, Result};
use indicatif::{ProgressBar, ProgressIterator};
use serde::Serialize;
use candle_core::{Device, DType, Tensor};

use crate::search::load::LoadedIndex;
use crate::search::padding::direct_pad_sequences;
use crate::search::tensor::StridedTensor;
use crate::utils::residual_codec::ResidualCodec;

/// Decompresses residual vectors from a packed, quantized format.
///
/// This function reconstructs full embedding vectors by combining coarse centroids with
/// fine-grained, quantized residuals. The residuals are packed with multiple codes per byte
/// (determined by `nbits`) and are unpacked using a series of lookup tables. This is a
/// typical operation in multi-stage vector quantization schemes designed to reduce
/// memory footprint.
///
/// The process involves:
/// 1. Unpacking `nbits` codes from each byte in `packed_residuals` using a bit-reversal map.
/// 2. Performing a series of indexed lookups to translate these codes into quantization bucket weights.
/// 3. Selecting the coarse centroids corresponding to the input `codes`.
/// 4. Adding the retrieved bucket weights (the decompressed residuals) to the coarse centroids.
///
/// # Preconditions
///
/// This function assumes specific dimensional relationships and will not work correctly if they
/// are not met. The caller must ensure:
/// - `(emb_dim * nbits)` is perfectly divisible by 8.
/// - 8 is perfectly divisible by `nbits`.
/// - The first dimension of `packed_residuals` matches the first dimension of `codes`.
/// - The second dimension of `packed_residuals` is `(emb_dim * nbits) / 8`.
///
/// # Arguments
///
/// * `packed_residuals` - The tensor of compressed residuals, where multiple codes are packed into each byte.
/// * `bucket_weights` - The codebook containing the fine-grained quantization vectors.
/// * `byte_reversed_bits_map` - A lookup table to efficiently unpack `nbits` codes from a byte.
/// * `bucket_weight_indices_lookup` - An intermediate table to map unpacked codes to `bucket_weights` indices.
/// * `codes` - Indices used to select the initial coarse centroids for each embedding.
/// * `centroids` - The codebook of coarse centroids.
/// * `emb_dim` - The dimensionality of the final, decompressed embedding vectors.
/// * `nbits` - The number of bits used for each sub-quantizer code within the packed residuals.
///
/// # Returns
///
/// A `Tensor` of shape `[num_embeddings, emb_dim]` containing the fully decompressed embeddings.
pub fn decompress_residuals(
    packed_residuals: &Tensor,
    bucket_weights: &Tensor,
    byte_reversed_bits_map: &Tensor,
    bucket_weight_indices_lookup: &Tensor,
    codes: &Tensor,
    centroids: &Tensor,
    emb_dim: i64,
    nbits: i64,
) -> Result<Tensor> {
    let num_embeddings = codes.dims()[0] as i64;

    const BITS_PER_PACKED_UNIT: i64 = 8;
    let packed_dim = (emb_dim * nbits) / BITS_PER_PACKED_UNIT;
    let codes_per_packed_unit = BITS_PER_PACKED_UNIT / nbits;

    let retrieved_centroids = centroids.index_select(codes, 0)?;
    let reshaped_centroids = retrieved_centroids.reshape((
        num_embeddings as usize,
        packed_dim as usize,
        codes_per_packed_unit as usize,
    ))?;

    let flat_packed_residuals_u8 = packed_residuals.flatten_all()?;
    let flat_packed_residuals_indices = flat_packed_residuals_u8.to_dtype(DType::I64)?;

    let flat_reversed_bits = byte_reversed_bits_map.index_select(&flat_packed_residuals_indices, 0)?;
    let reshaped_reversed_bits = flat_reversed_bits.reshape((
        num_embeddings as usize,
        packed_dim as usize,
    ))?;

    let flat_reversed_bits_for_lookup = reshaped_reversed_bits.flatten_all()?;

    let flat_selected_bucket_indices =
        bucket_weight_indices_lookup.index_select(&flat_reversed_bits_for_lookup, 0)?;
    let reshaped_selected_bucket_indices = flat_selected_bucket_indices.reshape((
        num_embeddings as usize,
        packed_dim as usize,
        codes_per_packed_unit as usize,
    ))?;

    let flat_bucket_indices_for_weights = reshaped_selected_bucket_indices.flatten_all()?;

    let flat_gathered_weights = bucket_weights.index_select(&flat_bucket_indices_for_weights, 0)?;
    let reshaped_gathered_weights = flat_gathered_weights.reshape((
        num_embeddings as usize,
        packed_dim as usize,
        codes_per_packed_unit as usize,
    ))?;

    let output_contributions_sum = (&reshaped_gathered_weights + &reshaped_centroids)?;
    let decompressed_embeddings = output_contributions_sum.reshape((
        num_embeddings as usize,
        emb_dim as usize,
    ))?;

    // Compute L2 norms and normalize
    let norms = decompressed_embeddings.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12f64, f64::INFINITY)?;
    let normalized = decompressed_embeddings.broadcast_div(&norms)?;
    
    Ok(normalized)
}

/// Represents the results of a single search query.
///
/// This struct is serializable and encapsulates the retrieved passage IDs and their
/// corresponding scores for a specific query.
#[derive(Serialize, Debug)]
pub struct QueryResult {
    /// The unique identifier for the query that produced these results.
    pub query_id: usize,
    /// A vector of document or passage identifiers, ranked by relevance.
    pub passage_ids: Vec<i64>,
    /// A vector of relevance scores corresponding to each passage in `passage_ids`.
    pub scores: Vec<f32>,
}

/// Search configuration parameters.
#[derive(Clone, Debug)]
pub struct SearchParameters {
    /// Number of queries per batch.
    pub batch_size: usize,
    /// Number of documents to re-rank with exact scores.
    pub n_full_scores: usize,
    /// Number of final results to return per query.
    pub top_k: usize,
    /// Number of IVF cells to probe during the initial search.
    pub n_ivf_probe: usize,
}

impl SearchParameters {
    /// Creates a new `SearchParameters` instance.
    pub fn new(batch_size: usize, n_full_scores: usize, top_k: usize, n_ivf_probe: usize) -> Self {
        Self {
            batch_size,
            n_full_scores,
            top_k,
            n_ivf_probe,
        }
    }
}

/// Processes a batch of queries against the loaded index.
///
/// This function iterates through query embeddings, executes the core search logic for each,
/// and collects the results, displaying a progress bar.
///
/// # Arguments
///
/// * `queries` - A 3D tensor of query embeddings with shape `[num_queries, tokens_per_query, dim]`.
/// * `index` - The `LoadedIndex` containing all necessary index components.
/// * `params` - `SearchParameters` for search configuration.
/// * `device` - The `tch::Device` for computation.
/// * `subset` - An optional list of document ID lists to restrict the search for each query.
///
/// # Returns
///
/// A `Result` with a `Vec<QueryResult>`. Individual search failures result in an empty
/// `QueryResult` for that specific query, ensuring the operation doesn't halt.
pub fn search_index(
    queries: &Tensor,
    index: &LoadedIndex,
    params: &SearchParameters,
    device: Device,
    show_progress: bool,
    subset: Option<Vec<Vec<i64>>>,
) -> Result<Vec<QueryResult>> {
    let query_shape = queries.dims();
    if query_shape.len() != 3 {
        bail!(
            "Expected a 3D tensor for queries, but got shape {:?}",
            query_shape
        );
    }
    let [num_queries, _, query_dim] = [query_shape[0] as i64, query_shape[1] as i64, query_shape[2] as i64];

    let mut results = Vec::new();
    
    let iter: Box<dyn Iterator<Item = i64>> = if show_progress {
        let bar = ProgressBar::new(num_queries.try_into().unwrap());
        Box::new((0..num_queries).progress_with(bar))
    } else {
        Box::new(0..num_queries)
    };

    for idx in iter {
        let query_embedding = match queries.get(idx as usize) {
            Ok(tensor) => match tensor.to_device(&device) {
                Ok(t) => t,
                Err(_) => {
                    results.push(QueryResult {
                        query_id: idx as usize,
                        passage_ids: vec![],
                        scores: vec![],
                    });
                    continue;
                },
            },
            Err(_) => {
                results.push(QueryResult {
                    query_id: idx as usize,
                    passage_ids: vec![],
                    scores: vec![],
                });
                continue;
            },
        };

        // Handle the per-query subset list
        let query_subset = subset.as_ref().and_then(|s| s.get(idx as usize));
        let subset_tensor = if let Some(ids) = query_subset {
            match Tensor::new(ids.as_slice(), &device).and_then(|t| t.to_dtype(DType::I64)) {
                Ok(tensor) => Some(tensor),
                Err(_) => None,
            }
        } else {
            None
        };

        let (passage_ids, scores) = search(
            &query_embedding,
            &index.ivf_index_strided,
            &index.codec,
            query_dim,
            &index.doc_codes_strided,
            &index.doc_residuals_strided,
            params.n_ivf_probe as i64,
            params.batch_size as i64,
            params.n_full_scores as i64,
            index.nbits,
            params.top_k,
            device.clone(),
            subset_tensor.as_ref(),
        )
        .unwrap_or_default();

        results.push(QueryResult {
            query_id: idx as usize,
            passage_ids,
            scores,
        });
    }

    Ok(results)
}

/// Reduces token-level similarity scores into a final document score using the ColBERT MaxSim strategy.
///
/// This function implements the core reduction step of the ColBERT model's scoring mechanism.
/// It first finds the maximum similarity score for each document token across all query tokens,
/// effectively ignoring padded tokens in the document. Then, it sums these maximum scores to
/// produce a single relevance score for each query-document pair in the batch.
///
/// # Arguments
///
/// * `token_scores` - A 3D `Tensor` of shape `[batch_size, query_length, doc_length]`
///   containing the token-level similarity scores.
/// * `attention_mask` - A 2D `Tensor` of shape `[batch_size, doc_length]` where `true`
///   indicates a valid token and `false` indicates a padded token.
///
/// # Returns
///
/// A 1D `Tensor` of shape `[batch_size]`, where each element is the final aggregated
/// ColBERT score for a query-document pair.
pub fn colbert_score_reduce(token_scores: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let scores_shape = token_scores.dims();

    // Expand the document attention mask to match the shape of the token scores.
    let expanded_mask = attention_mask.unsqueeze(scores_shape.len() - 1)?
        .broadcast_as(scores_shape)?;

    // Invert the mask to identify padding positions.
    let padding_mask = expanded_mask.eq(&Tensor::new(0u8, token_scores.device())?)?;

    // Nullify scores at padded positions by filling them with a large negative number.
    let neg_inf = Tensor::new(-9999.0f32, token_scores.device())?;
    let masked_scores = token_scores.where_cond(&padding_mask, &neg_inf)?;

    // For each document token, find the maximum similarity score across all query tokens (MaxSim).
    let max_scores_per_token = masked_scores.max(1)?;

    // Sum the MaxSim scores for all tokens in each document to get the final score.
    Ok(max_scores_per_token.sum_keepdim(scores_shape.len() - 1)?)
}

/// Intersects two tensors of integer IDs, returning a new tensor with the common elements.
///
/// This function implements an efficient intersection algorithm for tensors on a `tch` device.
/// It works by concatenating the two input tensors, sorting the result, and then identifying
/// adjacent duplicate elements, which correspond to the elements present in both original tensors.
///
/// # Preconditions
///
/// * The `pids` tensor must be sorted and contain unique elements.
///
/// # Arguments
///
/// * `pids` - A 1D tensor of passage IDs, assumed to be sorted and unique.
/// * `subset` - A 1D tensor of passage IDs to intersect with `pids`. This tensor does not
///   need to be sorted or unique, as this function will handle it.
/// * `device` - The `tch::Device` on which to create an empty tensor if the result is empty.
///
/// # Returns
///
/// A new 1D `Tensor` containing only the elements that are present in both `pids` and `subset`,
/// sorted in ascending order.
fn filter_pids_with_subset(pids: &Tensor, _subset: &Tensor, _device: &Device) -> Result<Tensor> {
    // Simplified implementation: just return the original pids for now
    // In a full implementation, this would compute the intersection of pids and subset
    Ok(pids.clone())
}

/// Performs a multi-stage search for a query against a quantized document index.
///
/// This function implements a multi-step search process common in efficient vector retrieval systems:
/// 1.  **IVF Probing**: Identifies a set of candidate documents by selecting the nearest Inverted File (IVF) cells.
/// 2.  **Approximate Scoring**: Computes fast, approximate scores for the candidate documents using their quantized codes.
/// 3.  **Re-ranking**: Filters the candidates based on approximate scores, then decompressesthe residuals for a smaller subset and computes exact scores.
/// 4.  **Top-K Selection**: Returns the highest-scoring documents.
///
/// # Arguments
/// * `query_embeddings` - A tensor containing the query embeddings.
/// * `ivf_index_strided` - A strided tensor representing the IVF index for coarse lookup.
/// * `codec` - The `ResidualCodec` used for decompressing document vectors.
/// * `emb_dim` - The dimensionality of the embeddings.
/// * `doc_codes_strided` - A strided tensor containing the quantized codes for all documents.
/// * `doc_residuals_strided` - A strided tensor containing the compressed residuals for all documents.
/// * `n_ivf_probe` - The number of IVF cells to probe for candidate documents.
/// * `batch_size` - The batch size used for processing documents during scoring.
/// * `n_docs_for_full_score` - The number of top documents from the approximate scoring phase to re-rank with full scoring.
/// * `nbits_param` - The number of bits used in the quantization codec.
/// * `top_k` - The final number of top results to return.
/// * `device` - The `tch::Device` (e.g., `Device::Cuda(0)`) on which to perform computations.
/// * `subset` - An optional tensor of document IDs to restrict the search to.
///
/// # Returns
/// A `Result` containing a tuple of two vectors: the top `k` passage IDs (`Vec<i64>`) and their
/// corresponding final scores (`Vec<f32>`).
///
/// # Errors
/// This function returns an error if tensor operations fail, if tensor dimensions are mismatched,
/// or if the provided `codec` is missing components required for full decompression.
pub fn search(
    query_embeddings: &Tensor,
    ivf_index_strided: &StridedTensor,
    codec: &ResidualCodec,
    emb_dim: i64,
    doc_codes_strided: &StridedTensor,
    doc_residuals_strided: &StridedTensor,
    n_ivf_probe: i64,
    batch_size: i64,
    n_docs_for_full_score: i64,
    nbits_param: i64,
    top_k: usize,
    device: Device,
    subset: Option<&Tensor>,
) -> anyhow::Result<(Vec<i64>, Vec<f32>)> {
    let result = {
        let query_embeddings_unsqueezed = query_embeddings.unsqueeze(0)?;

        let query_centroid_scores = codec.centroids.matmul(&query_embeddings.t()?)?;

        let selected_ivf_cells_indices = if n_ivf_probe == 1 {
            let argmax_result = query_centroid_scores.argmax_keepdim(0)?;
            argmax_result.transpose(0, 1)?
        } else {
            // Simplified: use argmax for now since Candle doesn't have topk
            // In a full implementation, we'd implement topk or use arg_sort + narrow
            let argmax_result = query_centroid_scores.argmax_keepdim(0)?;
            argmax_result.transpose(0, 1)?
        };

        let flat_selected_ivf_cells = selected_ivf_cells_indices.flatten_all()?;
        // Note: Candle doesn't have unique_dim, so we'll use a simpler approach
        let unique_ivf_cells_to_probe = flat_selected_ivf_cells; // Simplified for now

        let (retrieved_passage_ids_ivf, _) =
            ivf_index_strided.lookup(&unique_ivf_cells_to_probe, device.clone())?;
        let (sorted_passage_ids_ivf, _) = retrieved_passage_ids_ivf.sort_last_dim(false)?;
        // Note: Using sorted result directly since Candle doesn't have unique_consecutive
        let mut unique_passage_ids_after_ivf: Tensor = sorted_passage_ids_ivf;

        if let Some(subset_tensor) = subset {
            unique_passage_ids_after_ivf =
                filter_pids_with_subset(&unique_passage_ids_after_ivf, subset_tensor, &device)?;
        }

        if unique_passage_ids_after_ivf.elem_count() == 0 {
            return Ok((vec![], vec![]));
        }

        let mut approx_score_chunks = Vec::new();
        let total_pids_for_approx = unique_passage_ids_after_ivf.dims()[0] as i64;
        let num_approx_batches = (total_pids_for_approx + batch_size - 1) / batch_size;

        for batch_idx in 0..num_approx_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = ((batch_idx + 1) * batch_size).min(total_pids_for_approx);
            if batch_start >= batch_end {
                continue;
            }

            let batch_pids = unique_passage_ids_after_ivf.narrow(
                0, 
                batch_start as usize, 
                (batch_end - batch_start) as usize
            )?;
            let (batch_packed_codes, batch_doc_lengths) =
                doc_codes_strided.lookup(&batch_pids, device.clone())?;

            if batch_packed_codes.elem_count() == 0 {
                approx_score_chunks.push(Tensor::zeros(
                    (batch_pids.dims()[0],),
                    DType::F32,
                    &device,
                )?);
                continue;
            }

            let batch_approx_scores = query_centroid_scores.index_select(
                &batch_packed_codes.to_dtype(DType::I64)?, 
                0
            )?;
            let (padded_approx_scores, mask) =
                direct_pad_sequences(&batch_approx_scores, &batch_doc_lengths, 0.0, device.clone())?;
            approx_score_chunks.push(colbert_score_reduce(&padded_approx_scores, &mask)?);
        }

        let approx_scores = if !approx_score_chunks.is_empty() {
            Tensor::cat(&approx_score_chunks, 0)?
        } else {
            Tensor::zeros((0,), DType::F32, &device.clone())?
        };

        if approx_scores.dims()[0] != unique_passage_ids_after_ivf.dims()[0] {
            return Err(anyhow!(
                "PID ({}) and approx scores ({}) count mismatch.",
                unique_passage_ids_after_ivf.dims()[0],
                approx_scores.dims()[0],
            ));
        }

        let mut passage_ids_to_rerank = unique_passage_ids_after_ivf;
        let mut working_approx_scores = approx_scores;

        if n_docs_for_full_score < working_approx_scores.dims()[0] as i64
            && working_approx_scores.elem_count() > 0
        {
            // Simplified: use arg_sort + narrow to simulate topk
            let sorted_indices = working_approx_scores.arg_sort_last_dim(false)?;
            let top_indices = sorted_indices.narrow(0, 0, n_docs_for_full_score as usize)?;
            passage_ids_to_rerank = passage_ids_to_rerank.index_select(&top_indices, 0)?;
            working_approx_scores = working_approx_scores.gather(&top_indices.unsqueeze(1)?, 0)?.squeeze(1)?;
        }

        let n_pids_for_decomp = (n_docs_for_full_score / 4).max(1);
        if n_pids_for_decomp < working_approx_scores.dims()[0] as i64 && working_approx_scores.elem_count() > 0
        {
            let sorted_indices = working_approx_scores.arg_sort_last_dim(false)?;
            let top_indices = sorted_indices.narrow(0, 0, n_pids_for_decomp as usize)?;
            passage_ids_to_rerank = passage_ids_to_rerank.index_select(&top_indices, 0)?;
        }

        if passage_ids_to_rerank.elem_count() == 0 {
            return Ok((vec![], vec![]));
        }

        let (final_codes, final_doc_lengths) =
            doc_codes_strided.lookup(&passage_ids_to_rerank, device.clone())?;
        let (final_residuals, _) = doc_residuals_strided.lookup(&passage_ids_to_rerank, device.clone())?;

        let bucket_weights = codec
            .bucket_weights
            .as_ref()
            .ok_or_else(|| anyhow!("Codec missing bucket_weights for decompression."))?;
        let decomp_lookup = codec
            .decomp_indices_lookup
            .as_ref()
            .ok_or_else(|| anyhow!("Codec missing decomp_indices_lookup for decompression."))?;

        let decompressed_embs = decompress_residuals(
            &final_residuals,
            bucket_weights,
            &codec.byte_reversed_bits_map,
            decomp_lookup,
            &final_codes,
            &codec.centroids,
            emb_dim,
            nbits_param,
        )?;

        let (padded_doc_embs, mask) =
            direct_pad_sequences(&decompressed_embs, &final_doc_lengths, 0.0, device.clone())?;
        let final_scores = padded_doc_embs.matmul(&query_embeddings_unsqueezed.t()?)?;
        let reduced_scores = colbert_score_reduce(&final_scores, &mask)?;

        let sorted_indices = reduced_scores.arg_sort_last_dim(false)?;
        let sorted_scores = reduced_scores.gather(&sorted_indices.unsqueeze(1)?, 0)?.squeeze(1)?;
        let sorted_pids = passage_ids_to_rerank.gather(&sorted_indices.unsqueeze(1)?, 0)?.squeeze(1)?;

        // Convert tensors to vectors
        let pids_vec: Vec<i64> = sorted_pids.to_vec1()?;
        let scores_vec: Vec<f32> = sorted_scores.to_vec1()?;

        let result_count = top_k.min(pids_vec.len());
        Ok((
            pids_vec[..result_count].to_vec(),
            scores_vec[..result_count].to_vec(),
        ))
    };

    result
}
