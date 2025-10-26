/**
 * MaxSim WASM Loader for FastPlaid Demo
 * Handles WASM initialization with correct paths
 */

export class MaxSimLoader {
    constructor() {
        this.wasmInstance = null;
        this.isInitialized = false;
    }

    async init() {
        if (this.isInitialized) {
            return;
        }

        const maxRetries = 5;
        const retryDelay = 500; // ms

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                if (attempt === 1) {
                    console.log('🔄 Loading WASM MaxSim module...');
                } else {
                    console.log(`🔄 Retrying WASM MaxSim initialization (attempt ${attempt}/${maxRetries})...`);
                }

                // Force garbage collection if possible (helps with WASM memory issues)
                if (window.gc) {
                    window.gc();
                }

                // Wait a bit to let memory settle
                if (attempt > 1) {
                    await new Promise(resolve => setTimeout(resolve, retryDelay * Math.pow(2, attempt - 2)));
                }

                // Import the WASM module from the copied location
                const wasmModule = await import('./maxsim-wasm/maxsim_web_wasm.js');

                // Initialize WASM
                await wasmModule.default();

                // Create instance
                this.wasmInstance = new wasmModule.MaxSimWasm();
                this.isInitialized = true;

                // Check SIMD support
                const simdSupported = await this.checkSimdSupport();
                console.log(`✅ WASM MaxSim initialized successfully (attempt ${attempt}/${maxRetries})`);
                console.log(`   SIMD support: ${simdSupported ? '✓ ENABLED' : '✗ DISABLED (performance will be degraded)'}`);

                if (!simdSupported) {
                    console.warn('⚠️ WASM SIMD not supported - performance will be 10-13x slower!');
                    console.warn('   Try using Chrome 91+, Firefox 89+, or Safari 16.4+');
                }
                return;
            } catch (error) {
                if (attempt < maxRetries) {
                    console.warn(`⚠️ WASM MaxSim initialization attempt ${attempt} failed: ${error.message}`);

                    // If it's a table grow error, suggest memory cleanup
                    if (error.message.includes('Table.grow')) {
                        console.warn('💡 WASM table grow error detected - may need more memory. Retrying...');
                    }
                } else {
                    console.error('❌ WASM MaxSim initialization failed after all retries:', error);
                    throw new Error(`WASM MaxSim initialization failed after ${maxRetries} attempts: ${error.message}`);
                }
            }
        }
    }

    /**
     * Check if WASM SIMD is supported
     */
    async checkSimdSupport() {
        try {
            return await WebAssembly.validate(new Uint8Array([
                0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3,
                2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
            ]));
        } catch (e) {
            return false;
        }
    }

    /**
     * Official MaxSim: raw sum with dot product
     * Matches ColBERT, pylate-rs, mixedbread-ai implementations
     */
    maxsim(queryEmbedding, docEmbedding) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        // Debug: Check input structure
        if (!this._debugLogged) {
            console.log('🔍 WASM MaxSim Debug:');
            console.log('  queryEmbedding type:', Array.isArray(queryEmbedding) ? 'Array' : typeof queryEmbedding);
            console.log('  queryEmbedding.length:', queryEmbedding.length);
            console.log('  queryEmbedding[0] type:', Array.isArray(queryEmbedding[0]) ? 'Array' : typeof queryEmbedding[0]);
            console.log('  queryEmbedding[0].length:', queryEmbedding[0]?.length);
            console.log('  docEmbedding type:', Array.isArray(docEmbedding) ? 'Array' : typeof docEmbedding);
            console.log('  docEmbedding.length:', docEmbedding.length);
            console.log('  docEmbedding[0] type:', Array.isArray(docEmbedding[0]) ? 'Array' : typeof docEmbedding[0]);
            console.log('  docEmbedding[0].length:', docEmbedding[0]?.length);
            this._debugLogged = true;
        }

        // Convert to Float32Array if needed
        const t0 = performance.now();
        const queryFlat = this._flattenEmbedding(queryEmbedding);
        const docFlat = this._flattenEmbedding(docEmbedding);
        const flattenTime = performance.now() - t0;

        const queryTokens = queryEmbedding.length;
        const docTokens = docEmbedding.length;
        const dim = queryEmbedding[0].length;

        if (!this._debugLogged2) {
            console.log('  queryFlat.length:', queryFlat.length, 'expected:', queryTokens * dim);
            console.log('  docFlat.length:', docFlat.length, 'expected:', docTokens * dim);
            console.log('  Flatten time:', flattenTime.toFixed(3), 'ms');
            this._debugLogged2 = true;
        }

        // Call WASM maxsim_single (note: WASM API uses maxsim_single, not maxsim)
        const t1 = performance.now();
        const result = this.wasmInstance.maxsim_single(queryFlat, queryTokens, docFlat, docTokens, dim);
        const wasmTime = performance.now() - t1;

        if (!this._debugLogged3) {
            console.log('  WASM call time:', wasmTime.toFixed(3), 'ms');
            console.log('  Total overhead:', flattenTime.toFixed(3), 'ms');
            this._debugLogged3 = true;
        }

        return result;
    }

    /**
     * Normalized MaxSim: averaged for cross-query comparison
     */
    maxsim_normalized(queryEmbedding, docEmbedding) {
        if (!this.isInitialized) {
            throw new Error('WASM not initialized. Call init() first.');
        }

        const queryFlat = this._flattenEmbedding(queryEmbedding);
        const docFlat = this._flattenEmbedding(docEmbedding);

        const queryTokens = queryEmbedding.length;
        const docTokens = docEmbedding.length;
        const dim = queryEmbedding[0].length;

        return this.wasmInstance.maxsim_single_normalized(queryFlat, queryTokens, docFlat, docTokens, dim);
    }

    /**
     * Batch MaxSim computation
     */
    maxsimBatch(queryEmbedding, docEmbeddings) {
        return docEmbeddings.map(doc => this.maxsim(queryEmbedding, doc));
    }

    /**
     * Batch normalized MaxSim computation
     */
    maxsimBatch_normalized(queryEmbedding, docEmbeddings) {
        return docEmbeddings.map(doc => this.maxsim_normalized(queryEmbedding, doc));
    }

    _flattenEmbedding(embedding) {
        if (embedding instanceof Float32Array) {
            return embedding;
        }

        // If it's a 2D array, flatten it
        const dim = embedding[0].length;
        const tokens = embedding.length;
        const flat = new Float32Array(tokens * dim);

        for (let i = 0; i < tokens; i++) {
            for (let j = 0; j < dim; j++) {
                flat[i * dim + j] = embedding[i][j];
            }
        }

        return flat;
    }
}
