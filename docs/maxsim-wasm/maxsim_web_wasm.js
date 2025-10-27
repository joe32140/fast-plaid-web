let wasm;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedFloat32ArrayMemory0 = null;

function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let WASM_VECTOR_LEN = 0;

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_export_0.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

const MaxSimWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_maxsimwasm_free(ptr >>> 0, 1));

export class MaxSimWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MaxSimWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_maxsimwasm_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.maxsimwasm_new();
        this.__wbg_ptr = ret >>> 0;
        MaxSimWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Official MaxSim: raw sum with dot product
     * Expects L2-normalized embeddings. Matches ColBERT, pylate-rs, mixedbread-ai implementations
     * @param {Float32Array} query_flat
     * @param {number} query_tokens
     * @param {Float32Array} doc_flat
     * @param {number} doc_tokens
     * @param {number} embedding_dim
     * @returns {number}
     */
    maxsim_single(query_flat, query_tokens, doc_flat, doc_tokens, embedding_dim) {
        const ptr0 = passArrayF32ToWasm0(query_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(doc_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_maxsim_single(this.__wbg_ptr, ptr0, len0, query_tokens, ptr1, len1, doc_tokens, embedding_dim);
        return ret;
    }
    /**
     * Normalized MaxSim: averaged score for cross-query comparison
     * Expects L2-normalized embeddings
     * @param {Float32Array} query_flat
     * @param {number} query_tokens
     * @param {Float32Array} doc_flat
     * @param {number} doc_tokens
     * @param {number} embedding_dim
     * @returns {number}
     */
    maxsim_single_normalized(query_flat, query_tokens, doc_flat, doc_tokens, embedding_dim) {
        const ptr0 = passArrayF32ToWasm0(query_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(doc_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_maxsim_single_normalized(this.__wbg_ptr, ptr0, len0, query_tokens, ptr1, len1, doc_tokens, embedding_dim);
        return ret;
    }
    /**
     * Official MaxSim batch: raw sum with dot product
     * @param {Float32Array} query_flat
     * @param {number} query_tokens
     * @param {Float32Array} doc_flat
     * @param {Uint32Array} doc_tokens
     * @param {number} embedding_dim
     * @returns {Float32Array}
     */
    maxsim_batch(query_flat, query_tokens, doc_flat, doc_tokens, embedding_dim) {
        const ptr0 = passArrayF32ToWasm0(query_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(doc_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(doc_tokens, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_maxsim_batch(this.__wbg_ptr, ptr0, len0, query_tokens, ptr1, len1, ptr2, len2, embedding_dim);
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
    /**
     * Normalized MaxSim batch: averaged with dot product
     * @param {Float32Array} query_flat
     * @param {number} query_tokens
     * @param {Float32Array} doc_flat
     * @param {Uint32Array} doc_tokens
     * @param {number} embedding_dim
     * @returns {Float32Array}
     */
    maxsim_batch_normalized(query_flat, query_tokens, doc_flat, doc_tokens, embedding_dim) {
        const ptr0 = passArrayF32ToWasm0(query_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(doc_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(doc_tokens, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_maxsim_batch_normalized(this.__wbg_ptr, ptr0, len0, query_tokens, ptr1, len1, ptr2, len2, embedding_dim);
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
    /**
     * Official MaxSim batch uniform: raw sum with dot product
     * @param {Float32Array} query_flat
     * @param {number} query_tokens
     * @param {Float32Array} doc_flat
     * @param {number} num_docs
     * @param {number} doc_tokens
     * @param {number} embedding_dim
     * @returns {Float32Array}
     */
    maxsim_batch_uniform(query_flat, query_tokens, doc_flat, num_docs, doc_tokens, embedding_dim) {
        const ptr0 = passArrayF32ToWasm0(query_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(doc_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_maxsim_batch_uniform(this.__wbg_ptr, ptr0, len0, query_tokens, ptr1, len1, num_docs, doc_tokens, embedding_dim);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Normalized MaxSim batch uniform: averaged with dot product
     * @param {Float32Array} query_flat
     * @param {number} query_tokens
     * @param {Float32Array} doc_flat
     * @param {number} num_docs
     * @param {number} doc_tokens
     * @param {number} embedding_dim
     * @returns {Float32Array}
     */
    maxsim_batch_uniform_normalized(query_flat, query_tokens, doc_flat, num_docs, doc_tokens, embedding_dim) {
        const ptr0 = passArrayF32ToWasm0(query_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(doc_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_maxsim_batch_uniform_normalized(this.__wbg_ptr, ptr0, len0, query_tokens, ptr1, len1, num_docs, doc_tokens, embedding_dim);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Official MaxSim batch zero-copy: raw sum with dot product
     * @param {number} query_ptr
     * @param {number} query_tokens
     * @param {number} doc_ptr
     * @param {number} doc_tokens_ptr
     * @param {number} num_docs
     * @param {number} embedding_dim
     * @returns {Float32Array}
     */
    maxsim_batch_zero_copy(query_ptr, query_tokens, doc_ptr, doc_tokens_ptr, num_docs, embedding_dim) {
        const ret = wasm.maxsimwasm_maxsim_batch_zero_copy(this.__wbg_ptr, query_ptr, query_tokens, doc_ptr, doc_tokens_ptr, num_docs, embedding_dim);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Normalized MaxSim batch zero-copy: averaged with dot product
     * @param {number} query_ptr
     * @param {number} query_tokens
     * @param {number} doc_ptr
     * @param {number} doc_tokens_ptr
     * @param {number} num_docs
     * @param {number} embedding_dim
     * @returns {Float32Array}
     */
    maxsim_batch_zero_copy_normalized(query_ptr, query_tokens, doc_ptr, doc_tokens_ptr, num_docs, embedding_dim) {
        const ret = wasm.maxsimwasm_maxsim_batch_zero_copy_normalized(this.__wbg_ptr, query_ptr, query_tokens, doc_ptr, doc_tokens_ptr, num_docs, embedding_dim);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string}
     */
    get_info() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.maxsimwasm_get_info(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
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
     * @param {Float32Array} embeddings_data
     * @param {Uint32Array} doc_tokens
     * @param {number} embedding_dim
     */
    load_documents(embeddings_data, doc_tokens, embedding_dim) {
        const ptr0 = passArrayF32ToWasm0(embeddings_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(doc_tokens, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_load_documents(this.__wbg_ptr, ptr0, len0, ptr1, len1, embedding_dim);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
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
     * @param {Float32Array} query_flat
     * @param {number} query_tokens
     * @returns {Float32Array}
     */
    search_preloaded(query_flat, query_tokens) {
        const ptr0 = passArrayF32ToWasm0(query_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_search_preloaded(this.__wbg_ptr, ptr0, len0, query_tokens);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Search preloaded documents with normalized MaxSim scores
     * @param {Float32Array} query_flat
     * @param {number} query_tokens
     * @returns {Float32Array}
     */
    search_preloaded_normalized(query_flat, query_tokens) {
        const ptr0 = passArrayF32ToWasm0(query_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.maxsimwasm_search_preloaded_normalized(this.__wbg_ptr, ptr0, len0, query_tokens);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Get number of loaded documents
     * @returns {number}
     */
    num_documents_loaded() {
        const ret = wasm.maxsimwasm_num_documents_loaded(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) MaxSimWasm.prototype[Symbol.dispose] = MaxSimWasm.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_wbindgenthrow_451ec1a8469d7eb6 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_0;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('maxsim_web_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
