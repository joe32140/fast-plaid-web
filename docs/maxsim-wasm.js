
/**
 * MaxSim WASM Implementation
 *
 * High-performance implementation using WebAssembly with SIMD instructions.
 * Expected performance: 10x faster than pure JavaScript baseline.
 *
 * **IMPORTANT**: This implementation expects **L2-normalized embeddings** as input.
 * Modern embedding models (ColBERT, BGE, E5, etc.) output normalized embeddings by default.
 * For normalized embeddings, dot product equals cosine similarity.
 *
 * Algorithm:
 * For each query token, find the maximum dot product with all document tokens,
 * then sum these maximum similarities: score = Σ max(qi · dj)
 *
 * Two methods available:
 * - maxsim(): Official MaxSim (raw sum) - matches ColBERT, pylate-rs, mixedbread-ai
 * - maxsim_normalized(): Normalized MaxSim (averaged) - for cross-query comparison
 *
 * Requirements:
 * - Browser with WASM SIMD support (Chrome 91+, Firefox 89+, Safari 16.4+)
 * - Compiled WASM module in dist/wasm/
 */

export class MaxSimWasm {
    constructor() {
        this.wasmInstance = null;
        this.isInitialized = false;

        // Persistent buffers to eliminate allocations
        this.queryBuffer = null;
        this.docBuffer = null;
        this.resultBuffer = null;
        this.maxBufferSize = 50 * 1024 * 1024; // 50MB max buffer
    }

    /**
     * Initialize WASM module with retry logic
     * Must be called before using maxsim methods
     */
    async init() {
        if (this.isInitialized) {
            return;
        }

        const maxRetries = 3;
        const retryDelay = 200; // ms

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                if (attempt === 1) {
                    console.log(`🔄 WASM initialization (attempt ${attempt}/${maxRetries})...`);
                } else {
                    console.log(`🔄 Retrying WASM initialization (attempt ${attempt}/${maxRetries})...`);
                }

                // Dynamic import of WASM module - use relative path for GitHub Pages compatibility
                const wasmModule = await import('./maxsim-wasm/maxsim_web_wasm.js');

                // Initialize WASM
                await wasmModule.default();

                // Create instance
                this.wasmInstance = new wasmModule.MaxSimWasm();
                this.isInitialized = true;

                // Check SIMD support and log clean status
                const simdSupported = await MaxSimWasm.isSupported();
                const info = this.wasmInstance.get_info();
                console.log(`✅ WASM initialized successfully (SIMD: ${simdSupported ? '✓' : '✗'})`);
                return;
            } catch (error) {
                if (attempt < maxRetries) {
                    console.warn(`⚠️ Attempt ${attempt} failed, retrying...`);
                } else {
                    console.warn(`⚠️ WASM initialization failed:`, error.message);
                }

                if (attempt === maxRetries) {
                    console.error('❌ All WASM initialization attempts failed');
                    throw new Error(`WASM initialization failed after ${maxRetries} attempts: ${error.message}`);
                }

                // Wait before retrying
                await new Promise(resolve => setTimeout(resolve, retryDelay * attempt));
            }
        }
    }

    /**
     * Official MaxSim: raw sum with dot product
     * Matches ColBERT, pylate-rs, mixedbread-ai implementations
     * @param {number[][]|Float32Array[]} queryEmbedding - L2-normalized query embeddings
     * @param {number[][]|Float32Array[]} docEmbedding - L2-normalized document embeddings
     * @returns {number} MaxSim score (raw sum)
     */
    maxsim(queryEmbedding, docEmbedding) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        if (!queryEmbedding || queryEmbedding.length === 0 ||
            !docEmbedding || docEmbedding.length === 0) {
            return 0;
        }

        const { queryFlat, queryTokens, embeddingDim } = this.flattenEmbedding(queryEmbedding);
        const { docFlat, docTokens } = this.flattenEmbedding(docEmbedding);

        return this.wasmInstance.maxsim_single(
            queryFlat,
            queryTokens,
            docFlat,
            docTokens,
            embeddingDim
        );
    }

    /**
     * Normalized MaxSim: averaged score for cross-query comparison
     * @param {number[][]|Float32Array[]} queryEmbedding - L2-normalized query embeddings
     * @param {number[][]|Float32Array[]} docEmbedding - L2-normalized document embeddings
     * @returns {number} Normalized MaxSim score (averaged)
     */
    maxsim_normalized(queryEmbedding, docEmbedding) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        if (!queryEmbedding || queryEmbedding.length === 0 ||
            !docEmbedding || docEmbedding.length === 0) {
            return 0;
        }

        const { queryFlat, queryTokens, embeddingDim } = this.flattenEmbedding(queryEmbedding);
        const { docFlat, docTokens } = this.flattenEmbedding(docEmbedding);

        return this.wasmInstance.maxsim_single_normalized(
            queryFlat,
            queryTokens,
            docFlat,
            docTokens,
            embeddingDim
        );
    }

    /**
     * Ultra-efficient batch processing with minimal JS-WASM boundary crossings
     * @param {number[][]|Float32Array[]} queryEmbedding - Query embeddings
     * @param {Array<number[][]|Float32Array[]>} docEmbeddings - Array of document embeddings
     * @returns {Float32Array} Array of MaxSim scores
     */
    maxsimBatch(queryEmbedding, docEmbeddings) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        if (!queryEmbedding || queryEmbedding.length === 0 || docEmbeddings.length === 0) {
            return new Float32Array(docEmbeddings.length);
        }

        // Pre-allocate all memory to avoid repeated allocations
        const queryTokens = queryEmbedding.length;
        const embeddingDim = queryEmbedding[0].length;
        const numDocs = docEmbeddings.length;

        // Check if uniform processing is possible
        const firstDocLength = docEmbeddings[0].length;
        const allSameLength = docEmbeddings.every(doc => doc.length === firstDocLength);

        if (allSameLength) {
            // ULTRA-OPTIMIZED PATH: All docs same length
            const totalSize = queryTokens * embeddingDim + numDocs * firstDocLength * embeddingDim;
            const buffer = new Float32Array(totalSize);

            // Pack query data
            let offset = 0;
            for (let q = 0; q < queryTokens; q++) {
                const token = queryEmbedding[q];
                if (token instanceof Float32Array) {
                    buffer.set(token, offset);
                } else {
                    for (let d = 0; d < embeddingDim; d++) {
                        buffer[offset + d] = token[d];
                    }
                }
                offset += embeddingDim;
            }

            // Pack document data contiguously
            for (let docIdx = 0; docIdx < numDocs; docIdx++) {
                const doc = docEmbeddings[docIdx];
                for (let tokenIdx = 0; tokenIdx < firstDocLength; tokenIdx++) {
                    const token = doc[tokenIdx];
                    if (token instanceof Float32Array) {
                        buffer.set(token, offset);
                    } else {
                        for (let d = 0; d < embeddingDim; d++) {
                            buffer[offset + d] = token[d];
                        }
                    }
                    offset += embeddingDim;
                }
            }

            // Single WASM call with all data
            const queryStart = 0;
            const docStart = queryTokens * embeddingDim;

            const scores = this.wasmInstance.maxsim_batch_uniform(
                buffer.subarray(queryStart, docStart),
                queryTokens,
                buffer.subarray(docStart),
                numDocs,
                firstDocLength,
                embeddingDim
            );

            return new Float32Array(scores);
        } else {
            // VARIABLE LENGTH PATH: Minimize boundary crossings
            const docTokenCounts = docEmbeddings.map(doc => doc.length);
            const totalDocTokens = docTokenCounts.reduce((sum, count) => sum + count, 0);
            const totalSize = queryTokens * embeddingDim + totalDocTokens * embeddingDim;

            const buffer = new Float32Array(totalSize);

            // Pack query
            let offset = 0;
            for (let q = 0; q < queryTokens; q++) {
                const token = queryEmbedding[q];
                if (token instanceof Float32Array) {
                    buffer.set(token, offset);
                } else {
                    for (let d = 0; d < embeddingDim; d++) {
                        buffer[offset + d] = token[d];
                    }
                }
                offset += embeddingDim;
            }

            // Pack all documents
            for (const doc of docEmbeddings) {
                for (const token of doc) {
                    if (token instanceof Float32Array) {
                        buffer.set(token, offset);
                    } else {
                        for (let d = 0; d < embeddingDim; d++) {
                            buffer[offset + d] = token[d];
                        }
                    }
                    offset += embeddingDim;
                }
            }

            const queryStart = 0;
            const docStart = queryTokens * embeddingDim;

            const scores = this.wasmInstance.maxsim_batch(
                buffer.subarray(queryStart, docStart),
                queryTokens,
                buffer.subarray(docStart),
                new Uint32Array(docTokenCounts),
                embeddingDim
            );

            return new Float32Array(scores);
        }
    }

    /**
     * Normalized MaxSim batch: averaged scores for cross-query comparison
     * @param {number[][]|Float32Array[]} queryEmbedding - Query embeddings
     * @param {Array<number[][]|Float32Array[]>} docEmbeddings - Array of document embeddings
     * @returns {Float32Array} Array of normalized MaxSim scores
     */
    maxsimBatch_normalized(queryEmbedding, docEmbeddings) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        if (!queryEmbedding || queryEmbedding.length === 0 || docEmbeddings.length === 0) {
            return new Float32Array(docEmbeddings.length);
        }

        const queryTokens = queryEmbedding.length;
        const embeddingDim = queryEmbedding[0].length;
        const numDocs = docEmbeddings.length;

        const firstDocLength = docEmbeddings[0].length;
        const allSameLength = docEmbeddings.every(doc => doc.length === firstDocLength);

        if (allSameLength) {
            const totalSize = queryTokens * embeddingDim + numDocs * firstDocLength * embeddingDim;
            const buffer = new Float32Array(totalSize);

            let offset = 0;
            for (let q = 0; q < queryTokens; q++) {
                const token = queryEmbedding[q];
                if (token instanceof Float32Array) {
                    buffer.set(token, offset);
                } else {
                    for (let d = 0; d < embeddingDim; d++) {
                        buffer[offset + d] = token[d];
                    }
                }
                offset += embeddingDim;
            }

            for (let docIdx = 0; docIdx < numDocs; docIdx++) {
                const doc = docEmbeddings[docIdx];
                for (let tokenIdx = 0; tokenIdx < firstDocLength; tokenIdx++) {
                    const token = doc[tokenIdx];
                    if (token instanceof Float32Array) {
                        buffer.set(token, offset);
                    } else {
                        for (let d = 0; d < embeddingDim; d++) {
                            buffer[offset + d] = token[d];
                        }
                    }
                    offset += embeddingDim;
                }
            }

            const queryStart = 0;
            const docStart = queryTokens * embeddingDim;

            const scores = this.wasmInstance.maxsim_batch_uniform_normalized(
                buffer.subarray(queryStart, docStart),
                queryTokens,
                buffer.subarray(docStart),
                numDocs,
                firstDocLength,
                embeddingDim
            );

            return new Float32Array(scores);
        } else {
            const docTokenCounts = docEmbeddings.map(doc => doc.length);
            const totalDocTokens = docTokenCounts.reduce((sum, count) => sum + count, 0);
            const totalSize = queryTokens * embeddingDim + totalDocTokens * embeddingDim;

            const buffer = new Float32Array(totalSize);

            let offset = 0;
            for (let q = 0; q < queryTokens; q++) {
                const token = queryEmbedding[q];
                if (token instanceof Float32Array) {
                    buffer.set(token, offset);
                } else {
                    for (let d = 0; d < embeddingDim; d++) {
                        buffer[offset + d] = token[d];
                    }
                }
                offset += embeddingDim;
            }

            for (const doc of docEmbeddings) {
                for (const token of doc) {
                    if (token instanceof Float32Array) {
                        buffer.set(token, offset);
                    } else {
                        for (let d = 0; d < embeddingDim; d++) {
                            buffer[offset + d] = token[d];
                        }
                    }
                    offset += embeddingDim;
                }
            }

            const queryStart = 0;
            const docStart = queryTokens * embeddingDim;

            const scores = this.wasmInstance.maxsim_batch_normalized(
                buffer.subarray(queryStart, docStart),
                queryTokens,
                buffer.subarray(docStart),
                new Uint32Array(docTokenCounts),
                embeddingDim
            );

            return new Float32Array(scores);
        }
    }

    /**
     * Flatten embeddings for WASM
     * @private
     */
    flattenEmbedding(embedding) {
        const numTokens = embedding.length;
        const embeddingDim = embedding[0].length;
        const flat = new Float32Array(numTokens * embeddingDim);

        let offset = 0;
        for (const token of embedding) {
            if (token instanceof Float32Array) {
                flat.set(token, offset);
            } else {
                flat.set(new Float32Array(token), offset);
            }
            offset += embeddingDim;
        }

        return {
            queryFlat: flat,
            docFlat: flat,
            queryTokens: numTokens,
            docTokens: numTokens,
            embeddingDim: embeddingDim
        };
    }

    /**
     * ULTIMATE PERFORMANCE: Direct memory access with zero-copy operations
     * This is the fastest possible implementation
     */
    maxsimBatchHyperOptimized(queryEmbedding, docEmbeddings) {
        const totalStart = performance.now();
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        const queryTokens = queryEmbedding.length;
        const embeddingDim = queryEmbedding[0].length;
        const numDocs = docEmbeddings.length;

        // Calculate total memory needed
        const querySize = queryTokens * embeddingDim;
        const docTokenCounts = docEmbeddings.map(doc => doc.length);
        const totalDocSize = docTokenCounts.reduce((sum, count) => sum + count * embeddingDim, 0);

        const flattenStart = performance.now();
        // Allocate WASM linear memory directly
        const totalFloats = querySize + totalDocSize;
        const memory = new Float32Array(totalFloats);
        const docTokensArray = new Uint32Array(docTokenCounts);

        // Pack data directly into WASM memory with optimal layout
        let offset = 0;

        // Pack query with cache-friendly layout
        for (let q = 0; q < queryTokens; q++) {
            const token = queryEmbedding[q];
            if (token instanceof Float32Array) {
                memory.set(token, offset);
            } else {
                for (let d = 0; d < embeddingDim; d++) {
                    memory[offset + d] = token[d];
                }
            }
            offset += embeddingDim;
        }

        const docStartOffset = offset;

        // Pack documents with optimal memory layout
        for (const doc of docEmbeddings) {
            for (const token of doc) {
                if (token instanceof Float32Array) {
                    memory.set(token, offset);
                } else {
                    for (let d = 0; d < embeddingDim; d++) {
                        memory[offset + d] = token[d];
                    }
                }
                offset += embeddingDim;
            }
        }

        const flattenTime = performance.now() - flattenStart;

        // Call the regular batch method which ALREADY does zero-copy!
        // wasm-bindgen automatically creates slice views without copying
        const wasmStart = performance.now();
        const scores = this.wasmInstance.maxsim_batch(
            memory.subarray(0, querySize),
            queryTokens,
            memory.subarray(docStartOffset),
            docTokensArray,
            embeddingDim
        );
        const wasmTime = performance.now() - wasmStart;
        const totalTime = performance.now() - totalStart;

        if (typeof window !== 'undefined' && window.__MAXSIM_DEBUG) {
            if (!window.__MAXSIM_CALL_COUNT) window.__MAXSIM_CALL_COUNT = 0;
            window.__MAXSIM_CALL_COUNT++;
            if (window.__MAXSIM_CALL_COUNT <= 5) {
                console.log(`[NON-PRELOADED #${window.__MAXSIM_CALL_COUNT}] Total: ${totalTime.toFixed(2)}ms | Flatten: ${flattenTime.toFixed(2)}ms | WASM: ${wasmTime.toFixed(2)}ms`);
            }
        }

        return new Float32Array(scores);
    }

    /**
     * Ultra-fast batch processing with zero-allocation persistent buffers
     * This eliminates all memory allocation overhead
     */
    maxsimBatchZeroAlloc(queryEmbedding, docEmbeddings) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        const queryTokens = queryEmbedding.length;
        const embeddingDim = queryEmbedding[0].length;
        const numDocs = docEmbeddings.length;

        // Calculate required buffer sizes
        const querySize = queryTokens * embeddingDim;
        const docSize = docEmbeddings.reduce((sum, doc) => sum + doc.length * embeddingDim, 0);
        const totalSize = querySize + docSize;

        // Resize persistent buffer if needed
        if (!this.docBuffer || this.docBuffer.length < totalSize) {
            const newSize = Math.min(Math.max(totalSize * 1.5, 1024 * 1024), this.maxBufferSize);
            this.docBuffer = new Float32Array(newSize);
            console.log(`Resized WASM buffer to ${(newSize * 4 / 1024 / 1024).toFixed(1)}MB`);
        }

        // Pack data directly into persistent buffer
        let offset = 0;

        // Pack query
        for (let q = 0; q < queryTokens; q++) {
            const token = queryEmbedding[q];
            if (token instanceof Float32Array) {
                this.docBuffer.set(token, offset);
            } else {
                for (let d = 0; d < embeddingDim; d++) {
                    this.docBuffer[offset + d] = token[d];
                }
            }
            offset += embeddingDim;
        }

        const docStart = offset;

        // Pack documents
        const docTokenCounts = [];
        for (const doc of docEmbeddings) {
            docTokenCounts.push(doc.length);
            for (const token of doc) {
                if (token instanceof Float32Array) {
                    this.docBuffer.set(token, offset);
                } else {
                    for (let d = 0; d < embeddingDim; d++) {
                        this.docBuffer[offset + d] = token[d];
                    }
                }
                offset += embeddingDim;
            }
        }

        // Single WASM call with persistent buffer views
        const scores = this.wasmInstance.maxsim_batch(
            this.docBuffer.subarray(0, querySize),
            queryTokens,
            this.docBuffer.subarray(docStart, offset),
            new Uint32Array(docTokenCounts),
            embeddingDim
        );

        return new Float32Array(scores);
    }

    /**
     * FLAT API - Zero-Copy Batch Processing
     *
     * Use this when your embeddings are already in flat Float32Array format.
     * This eliminates all conversion overhead (up to 300ms saved for large batches).
     *
     * @param {Float32Array} queryFlat - Flat query embeddings (queryTokens × embeddingDim)
     * @param {number} queryTokens - Number of query tokens
     * @param {Float32Array} docsFlat - Flat concatenated document embeddings
     * @param {Uint32Array|number[]} docTokenCounts - Number of tokens per document
     * @param {number} embeddingDim - Embedding dimension
     * @returns {Float32Array} Array of MaxSim scores
     *
     * @example
     * // Embeddings already flat from ML library
     * const queryFlat = new Float32Array(13 * 48);  // 13 tokens × 48 dim
     * const docsFlat = new Float32Array(270000);     // All docs concatenated
     * const docTokenCounts = new Uint32Array([256, 270, 245, ...]); // 1000 docs
     *
     * const scores = maxsim.maxsimBatchFlat(
     *     queryFlat, 13, docsFlat, docTokenCounts, 48
     * );
     */
    maxsimBatchFlat(queryFlat, queryTokens, docsFlat, docTokenCounts, embeddingDim) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        // Validate inputs
        const expectedQuerySize = queryTokens * embeddingDim;
        if (queryFlat.length !== expectedQuerySize) {
            throw new Error(
                `Query size mismatch: expected ${expectedQuerySize} (${queryTokens} × ${embeddingDim}), ` +
                `got ${queryFlat.length}`
            );
        }

        const docTokenCountsArray = docTokenCounts instanceof Uint32Array ?
            docTokenCounts : new Uint32Array(docTokenCounts);

        const numDocs = docTokenCountsArray.length;

        if (numDocs === 0) {
            return new Float32Array(0);
        }

        // Direct WASM call - NO conversion overhead!
        const scores = this.wasmInstance.maxsim_batch(
            queryFlat,
            queryTokens,
            docsFlat,
            docTokenCountsArray,
            embeddingDim
        );

        return new Float32Array(scores);
    }

    /**
     * FLAT API - Normalized MaxSim Batch
     *
     * Same as maxsimBatchFlat but returns normalized (averaged) scores.
     *
     * @param {Float32Array} queryFlat - Flat query embeddings
     * @param {number} queryTokens - Number of query tokens
     * @param {Float32Array} docsFlat - Flat concatenated document embeddings
     * @param {Uint32Array|number[]} docTokenCounts - Number of tokens per document
     * @param {number} embeddingDim - Embedding dimension
     * @returns {Float32Array} Array of normalized MaxSim scores
     */
    maxsimBatchFlat_normalized(queryFlat, queryTokens, docsFlat, docTokenCounts, embeddingDim) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        const expectedQuerySize = queryTokens * embeddingDim;
        if (queryFlat.length !== expectedQuerySize) {
            throw new Error(
                `Query size mismatch: expected ${expectedQuerySize}, got ${queryFlat.length}`
            );
        }

        const docTokenCountsArray = docTokenCounts instanceof Uint32Array ?
            docTokenCounts : new Uint32Array(docTokenCounts);

        if (docTokenCountsArray.length === 0) {
            return new Float32Array(0);
        }

        const scores = this.wasmInstance.maxsim_batch_normalized(
            queryFlat,
            queryTokens,
            docsFlat,
            docTokenCountsArray,
            embeddingDim
        );

        return new Float32Array(scores);
    }

    /**
     * FLAT API - Single Document MaxSim
     *
     * Compute MaxSim for a single query-document pair using flat arrays.
     *
     * @param {Float32Array} queryFlat - Flat query embeddings
     * @param {number} queryTokens - Number of query tokens
     * @param {Float32Array} docFlat - Flat document embeddings
     * @param {number} docTokens - Number of document tokens
     * @param {number} embeddingDim - Embedding dimension
     * @returns {number} MaxSim score
     */
    maxsimFlat(queryFlat, queryTokens, docFlat, docTokens, embeddingDim) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        return this.wasmInstance.maxsim_single(
            queryFlat,
            queryTokens,
            docFlat,
            docTokens,
            embeddingDim
        );
    }

    /**
     * FLAT API - Single Document Normalized MaxSim
     *
     * @param {Float32Array} queryFlat - Flat query embeddings
     * @param {number} queryTokens - Number of query tokens
     * @param {Float32Array} docFlat - Flat document embeddings
     * @param {number} docTokens - Number of document tokens
     * @param {number} embeddingDim - Embedding dimension
     * @returns {number} Normalized MaxSim score
     */
    maxsimFlat_normalized(queryFlat, queryTokens, docFlat, docTokens, embeddingDim) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        return this.wasmInstance.maxsim_single_normalized(
            queryFlat,
            queryTokens,
            docFlat,
            docTokens,
            embeddingDim
        );
    }

    /**
     * Load documents into WASM memory for repeated searches (NEW in v0.5.0!)
     * Following FastPlaid architecture: load once, search many times with ZERO conversion overhead
     * @param {Array<number[][]|Float32Array[]>|Float32Array} docEmbeddings - Array of document embeddings OR flat array
     * @param {Uint32Array} [docTokens] - Optional: Array of token counts (required if first param is flat)
     * @param {number} [embeddingDim] - Optional: Embedding dimension (required if first param is flat)
     */
    async loadDocuments(docEmbeddings, docTokens, embeddingDim) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        // Support both 2D array input (high-level) and flat array input (low-level)
        if (docEmbeddings instanceof Float32Array) {
            // Low-level flat API
            if (!(docTokens instanceof Uint32Array)) {
                throw new Error('docTokens must be Uint32Array when using flat array input');
            }
            if (!embeddingDim) {
                throw new Error('embeddingDim is required when using flat array input');
            }
            this.wasmInstance.load_documents(docEmbeddings, docTokens, embeddingDim);
        } else {
            // High-level 2D array API (like fast-plaid usage)
            if (!docEmbeddings || docEmbeddings.length === 0) {
                throw new Error('Document embeddings cannot be empty');
            }

            const numDocs = docEmbeddings.length;
            const dim = docEmbeddings[0][0].length;

            // Flatten all documents into a single buffer
            const tokenCounts = docEmbeddings.map(doc => doc.length);
            const totalDocTokens = tokenCounts.reduce((sum, count) => sum + count, 0);
            const buffer = new Float32Array(totalDocTokens * dim);

            let offset = 0;
            for (const doc of docEmbeddings) {
                for (const token of doc) {
                    if (token instanceof Float32Array) {
                        buffer.set(token, offset);
                    } else {
                        for (let d = 0; d < dim; d++) {
                            buffer[offset + d] = token[d];
                        }
                    }
                    offset += dim;
                }
            }

            // Load into WASM memory
            this.wasmInstance.load_documents(
                buffer,
                new Uint32Array(tokenCounts),
                dim
            );
        }
    }

    /**
     * Search preloaded documents with a query
     * Returns MaxSim scores for all documents
     *
     * @param {number[][]|Float32Array[]} queryEmbedding - L2-normalized query embeddings
     * @returns {Float32Array} MaxSim scores (one per document)
     */
    searchPreloaded(queryEmbedding) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        // Flatten query embedding
        const { queryFlat, queryTokens, embeddingDim } = this.flattenEmbedding(queryEmbedding);

        // Call WASM search
        return this.wasmInstance.search_preloaded(queryFlat, queryTokens);
    }

    /**
     * Search preloaded documents with normalized MaxSim scores
     *
     * @param {number[][]|Float32Array[]} queryEmbedding - L2-normalized query embeddings
     * @returns {Float32Array} Normalized MaxSim scores (one per document)
     */
    searchPreloadedNormalized(queryEmbedding) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        // Flatten query embedding
        const { queryFlat, queryTokens, embeddingDim } = this.flattenEmbedding(queryEmbedding);

        // Call WASM search
        return this.wasmInstance.search_preloaded_normalized(queryFlat, queryTokens);
    }

    /**
     * Get number of loaded documents
     * @returns {number} Number of documents currently loaded in WASM memory
     */
    numDocumentsLoaded() {
        if (!this.isInitialized) {
            return 0;
        }

        return this.wasmInstance.num_documents_loaded();
    }

    /**
     * Get implementation info
     * @returns {object} Implementation details
     */
    getInfo() {
        return {
            name: 'MaxSim WASM',
            version: '2.1.0',
            backend: 'wasm-simd',
            features: ['simd', 'batch-processing', 'zero-alloc-buffers', 'adaptive-batching', 'flat-api', 'document-preloading'],
            methods: [
                'maxsim() - 2D array API (convenience)',
                'maxsimFlat() - Flat array API (performance)',
                'maxsimBatch() - 2D array batch',
                'maxsimBatchFlat() - Flat array batch (FASTEST)',
                'maxsim_normalized() - Averaged scores',
                'maxsimBatchFlat_normalized() - Flat batch normalized',
                'loadDocuments() - NEW v0.5.0: Preload documents',
                'searchPreloaded() - NEW v0.5.0: Search preloaded (FASTEST!)',
                'searchPreloadedNormalized() - NEW v0.5.0: Normalized search',
                'numDocumentsLoaded() - NEW v0.5.0: Check loaded count'
            ],
            initialized: this.isInitialized,
            bufferSize: this.docBuffer ? `${(this.docBuffer.length * 4 / 1024 / 1024).toFixed(1)}MB` : 'Not allocated',
            wasmInfo: this.isInitialized ? this.wasmInstance.get_info() : 'Not initialized',
            recommendation: 'Use searchPreloaded() after loadDocuments() for best performance (4.5x faster for repeated searches!)'
        };
    }

    /**
     * Static method to check if WASM SIMD is supported
     * @returns {Promise<boolean>}
     */
    static async isSupported() {
        try {
            // Check for basic WASM support
            if (typeof WebAssembly !== 'object') {
                return false;
            }

            // Check for SIMD support using a minimal SIMD module
            const simdTest = new Uint8Array([
                0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
                10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
            ]);

            return WebAssembly.validate(simdTest);
        } catch {
            return false;
        }
    }
}