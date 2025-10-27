/* tslint:disable */
/* eslint-disable */
export class MaxSimWasm {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  /**
   * Official MaxSim: raw sum with dot product
   * Expects L2-normalized embeddings. Matches ColBERT, pylate-rs, mixedbread-ai implementations
   */
  maxsim_single(query_flat: Float32Array, query_tokens: number, doc_flat: Float32Array, doc_tokens: number, embedding_dim: number): number;
  /**
   * Normalized MaxSim: averaged score for cross-query comparison
   * Expects L2-normalized embeddings
   */
  maxsim_single_normalized(query_flat: Float32Array, query_tokens: number, doc_flat: Float32Array, doc_tokens: number, embedding_dim: number): number;
  /**
   * Official MaxSim batch: raw sum with dot product
   */
  maxsim_batch(query_flat: Float32Array, query_tokens: number, doc_flat: Float32Array, doc_tokens: Uint32Array, embedding_dim: number): Float32Array;
  /**
   * Normalized MaxSim batch: averaged with dot product
   */
  maxsim_batch_normalized(query_flat: Float32Array, query_tokens: number, doc_flat: Float32Array, doc_tokens: Uint32Array, embedding_dim: number): Float32Array;
  /**
   * Official MaxSim batch uniform: raw sum with dot product
   */
  maxsim_batch_uniform(query_flat: Float32Array, query_tokens: number, doc_flat: Float32Array, num_docs: number, doc_tokens: number, embedding_dim: number): Float32Array;
  /**
   * Normalized MaxSim batch uniform: averaged with dot product
   */
  maxsim_batch_uniform_normalized(query_flat: Float32Array, query_tokens: number, doc_flat: Float32Array, num_docs: number, doc_tokens: number, embedding_dim: number): Float32Array;
  /**
   * Official MaxSim batch zero-copy: raw sum with dot product
   */
  maxsim_batch_zero_copy(query_ptr: number, query_tokens: number, doc_ptr: number, doc_tokens_ptr: number, num_docs: number, embedding_dim: number): Float32Array;
  /**
   * Normalized MaxSim batch zero-copy: averaged with dot product
   */
  maxsim_batch_zero_copy_normalized(query_ptr: number, query_tokens: number, doc_ptr: number, doc_tokens_ptr: number, num_docs: number, embedding_dim: number): Float32Array;
  get_info(): string;
  /**
   * Load and store document embeddings in WASM memory
   * This eliminates per-search conversion overhead (following FastPlaid's pattern)
   *
   * # Arguments
   * * `embeddings_data` - Flat array of all document embeddings concatenated
   * * `doc_tokens` - Array of token counts for each document
   * * `embedding_dim` - Embedding dimension
   *
   * # Example
   * For 3 documents with [128, 256, 192] tokens each at dim=48:
   * - embeddings_data.len() = (128 + 256 + 192) * 48 = 27,648
   * - doc_tokens = [128, 256, 192]
   */
  load_documents(embeddings_data: Float32Array, doc_tokens: Uint32Array, embedding_dim: number): void;
  /**
   * Search preloaded documents with a query
   * Returns MaxSim scores for all documents
   *
   * # Arguments
   * * `query_flat` - Flat query embedding (query_tokens × embedding_dim)
   * * `query_tokens` - Number of query tokens
   *
   * # Returns
   * Float32Array of MaxSim scores (one per document)
   */
  search_preloaded(query_flat: Float32Array, query_tokens: number): Float32Array;
  /**
   * Search preloaded documents with normalized MaxSim scores
   */
  search_preloaded_normalized(query_flat: Float32Array, query_tokens: number): Float32Array;
  /**
   * Get number of loaded documents
   */
  num_documents_loaded(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_maxsimwasm_free: (a: number, b: number) => void;
  readonly maxsimwasm_new: () => number;
  readonly maxsimwasm_maxsim_single: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
  readonly maxsimwasm_maxsim_single_normalized: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
  readonly maxsimwasm_maxsim_batch: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
  readonly maxsimwasm_maxsim_batch_normalized: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
  readonly maxsimwasm_maxsim_batch_uniform: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
  readonly maxsimwasm_maxsim_batch_uniform_normalized: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
  readonly maxsimwasm_maxsim_batch_zero_copy: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly maxsimwasm_maxsim_batch_zero_copy_normalized: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly maxsimwasm_get_info: (a: number) => [number, number];
  readonly maxsimwasm_load_documents: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
  readonly maxsimwasm_search_preloaded: (a: number, b: number, c: number, d: number) => [number, number, number, number];
  readonly maxsimwasm_search_preloaded_normalized: (a: number, b: number, c: number, d: number) => [number, number, number, number];
  readonly maxsimwasm_num_documents_loaded: (a: number) => number;
  readonly __wbindgen_export_0: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
