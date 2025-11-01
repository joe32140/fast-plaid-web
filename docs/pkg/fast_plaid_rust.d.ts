/* tslint:disable */
/* eslint-disable */
/**
 * Utility function to validate embeddings format for mxbai-edge-colbert
 */
export function validate_mxbai_embeddings(embeddings: Float32Array, expected_dim: number): boolean;
/**
 * Initialize the WASM module (called automatically)
 */
export function main(): void;
/**
 * WASM wrapper for quantized FastPlaid
 */
export class FastPlaidQuantized {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Creates a new quantized FastPlaid instance
   */
  constructor();
  /**
   * Load and quantize document embeddings
   * Training happens automatically on the provided embeddings
   */
  load_documents_quantized(embeddings_data: Float32Array, doc_info: BigInt64Array, num_centroids?: number | null): void;
  /**
   * Incrementally add new documents to the index without full rebuild
   * Uses existing codec to compress new documents and stores IVF updates as deltas
   */
  update_index_incremental(embeddings_data: Float32Array, doc_info: BigInt64Array): void;
  /**
   * Set nprobe (clusters to probe per query token)
   * PLAID default: 4 clusters per token
   * Higher values = better recall, slower search
   * Lower values = faster search, lower recall
   */
  set_nprobe(nprobe: number): void;
  /**
   * Get current nprobe setting
   */
  get_nprobe(): number;
  /**
   * Search with quantized embeddings
   */
  search(query_embeddings: Float32Array, query_shape: Uint32Array, top_k: number): string;
  get_index_info(): string;
  get_num_documents(): number;
  /**
   * Save the quantized index to binary format
   * Returns binary data that can be saved to disk
   * Note: Automatically compacts deltas before saving for optimal performance
   */
  save_index(): Uint8Array;
  /**
   * Load a precomputed quantized index from binary format
   */
  load_index(index_bytes: Uint8Array): void;
  /**
   * Manually trigger compaction of deltas into base IVF
   * Useful for forcing compaction before save or for performance tuning
   */
  compact_index(): void;
}
/**
 * WASM wrapper for FastPlaid search functionality
 */
export class FastPlaidWasm {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Creates a new FastPlaidWasm instance
   */
  constructor();
  /**
   * Loads document embeddings from JavaScript
   *
   * # Arguments
   * * `embeddings_data` - Flat array of all document embeddings concatenated
   * * `doc_info` - Array of [doc_id, num_tokens] pairs for each document
   */
  load_documents(embeddings_data: Float32Array, doc_info: BigInt64Array): void;
  /**
   * Loads a FastPlaid index from bytes (simplified version for now)
   */
  load_index(_index_bytes: Uint8Array): void;
  /**
   * Searches the loaded index with query embeddings using ColBERT MaxSim
   *
   * # Arguments
   * * `query_embeddings` - Flat array of f32 embeddings from mxbai-edge-colbert
   * * `query_shape` - Shape of the query tensor [batch_size, seq_len, embedding_dim]
   * * `top_k` - Number of top results to return
   * * `n_ivf_probe` - Number of IVF cells to probe (ignored in this implementation)
   *
   * # Returns
   * Returns a JSON string (not JsValue) to avoid externref table overflow issues
   */
  search(query_embeddings: Float32Array, query_shape: Uint32Array, top_k: number, _n_ivf_probe?: number | null): string;
  /**
   * Gets information about the loaded index
   */
  get_index_info(): string;
  /**
   * Sets the embedding dimension (useful for different models)
   */
  set_embedding_dim(dim: number): void;
  /**
   * Gets the current embedding dimension
   */
  get_embedding_dim(): number;
  /**
   * Gets the number of documents loaded
   */
  get_num_documents(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_fastplaidquantized_free: (a: number, b: number) => void;
  readonly fastplaidquantized_new: () => [number, number, number];
  readonly fastplaidquantized_load_documents_quantized: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
  readonly fastplaidquantized_update_index_incremental: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly fastplaidquantized_set_nprobe: (a: number, b: number) => void;
  readonly fastplaidquantized_get_nprobe: (a: number) => number;
  readonly fastplaidquantized_search: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
  readonly fastplaidquantized_get_index_info: (a: number) => [number, number, number, number];
  readonly fastplaidquantized_get_num_documents: (a: number) => number;
  readonly fastplaidquantized_save_index: (a: number) => [number, number, number, number];
  readonly fastplaidquantized_load_index: (a: number, b: number, c: number) => [number, number];
  readonly fastplaidquantized_compact_index: (a: number) => [number, number];
  readonly __wbg_fastplaidwasm_free: (a: number, b: number) => void;
  readonly fastplaidwasm_new: () => [number, number, number];
  readonly fastplaidwasm_load_documents: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly fastplaidwasm_load_index: (a: number, b: number, c: number) => [number, number];
  readonly fastplaidwasm_search: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
  readonly fastplaidwasm_get_index_info: (a: number) => [number, number, number, number];
  readonly fastplaidwasm_set_embedding_dim: (a: number, b: number) => void;
  readonly fastplaidwasm_get_embedding_dim: (a: number) => number;
  readonly fastplaidwasm_get_num_documents: (a: number) => number;
  readonly validate_mxbai_embeddings: (a: number, b: number, c: number) => [number, number, number];
  readonly main: () => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_3: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
