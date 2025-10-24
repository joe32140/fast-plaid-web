/**
 * Real integration layer for mixedbread-ai/mxbai-edge-colbert-v0-17m model
 * Uses pylate-rs WASM library to load and run the actual Hugging Face model
 */

// Import pylate-rs (will be loaded via ES modules)
let ColBERT = null;

class MxbaiEdgeColbertIntegration {
    constructor() {
        this.model = null;
        this.modelLoaded = false;
        // NOTE: mxbai-edge-colbert-v0-17m outputs 48 dims (with 2_Dense projection)
        // pylate-rs now supports 2_Dense layers!
        this.embeddingDim = 48; // Correct output with 2_Dense: 256->512->48
        this.maxSequenceLength = 512; // Typical max length for ColBERT models
        // Use mixedbread-ai/mxbai-edge-colbert-v0-17m exclusively
        this.modelRepo = 'mixedbread-ai/mxbai-edge-colbert-v0-17m';
        this.fallbackModels = []; // No fallback - use mxbai only

        // Performance tracking
        this.timings = {
            modelEncoding: [],
            queryEncoding: [],
            searchTime: [],
            totalSearchTime: [],
            indexTime: []
        };

        // Index size and memory tracking
        this.indexMemory = {
            fastPlaid: {
                totalBytes: 0,
                embeddingsBytes: 0,
                metadataBytes: 0,
                documentCount: 0,
                embeddingDim: 0
            },
            directMaxSim: {
                totalBytes: 0,
                embeddingsBytes: 0,
                metadataBytes: 0,
                documentCount: 0,
                embeddingDim: 0
            }
        };

        // Required files for pylate-rs ColBERT models
        // Note: pylate-rs ColBERT now supports multi-stage Dense layers!
        // mxbai-edge-colbert-v0-17m has 1_Dense (256->512) and 2_Dense (512->48)
        // With 2_Dense support, we get correct 48-dim embeddings
        this.requiredFiles = [
            'tokenizer.json',
            'model.safetensors',
            'config.json',
            'config_sentence_transformers.json',
            '1_Dense/model.safetensors',
            '1_Dense/config.json',
            '2_Dense/model.safetensors',  // Second dense layer: 512 -> 48 (NOW SUPPORTED!)
            '2_Dense/config.json',
            'special_tokens_map.json',
        ];

        // Alternative file structure for some models
        this.alternativeFiles = [
            'tokenizer.json',
            'pytorch_model.bin', // Some models use .bin instead of .safetensors
            'config.json',
            'sentence_bert_config.json', // Alternative name
            '1_Dense/pytorch_model.bin',
            '1_Dense/config.json',
            '2_Dense/pytorch_model.bin',  // Second dense layer (alternative format)
            '2_Dense/config.json',
            'tokenizer_config.json', // Alternative to special_tokens_map.json
        ];
    }

    /**
     * Initialize pylate-rs and load the ColBERT model
     */
    async initializeModel() {
        console.log('🚀 Loading real ColBERT model with pylate-rs...');

        try {
            // Import pylate-rs WASM module
            console.log('📦 Importing pylate-rs WASM module...');
            const pylateModule = await import('./node_modules/pylate-rs/pylate_rs.js');
            console.log('🔧 Initializing WASM...');
            await pylateModule.default(); // Initialize WASM
            ColBERT = pylateModule.ColBERT;

            console.log('✅ pylate-rs WASM module loaded successfully');
            console.log('🔍 Available ColBERT class:', ColBERT);

            // Load the actual model from Hugging Face
            await this.loadModelFromHuggingFace();

            this.modelLoaded = true;
            this.simulationMode = false;
            console.log(`🎉 Real ColBERT model (${this.modelRepo}) loaded successfully!`);
            return true;

        } catch (error) {
            console.error('❌ Failed to initialize real ColBERT model:', error);
            console.error('Error details:', error.stack);

            // For demo purposes, let's try to continue with simulation
            // In production, you might want to show an error to the user
            console.log('🔄 Falling back to simulation mode for demo...');
            await this.initializeSimulationMode();
            return true;
        }
    }

    /**
     * Fallback simulation mode if real model loading fails
     */
    async initializeSimulationMode() {
        console.log('🎭 Initializing simulation mode...');
        await new Promise(resolve => setTimeout(resolve, 1000));
        this.modelLoaded = true;
        this.simulationMode = true;
        console.log('✅ Simulation mode ready');
    }

    /**
     * Load model files from Hugging Face Hub with fallback models
     */
    async loadModelFromHuggingFace() {
        const modelsToTry = [this.modelRepo, ...this.fallbackModels];

        for (const modelRepo of modelsToTry) {
            try {
                console.log(`📥 Trying to load ${modelRepo}...`);
                await this.loadSingleModel(modelRepo);
                this.modelRepo = modelRepo; // Update to the working model
                console.log(`✅ Successfully loaded ${modelRepo}`);
                return;
            } catch (error) {
                console.warn(`❌ Failed to load ${modelRepo}:`, error.message);
                if (modelRepo === modelsToTry[modelsToTry.length - 1]) {
                    throw new Error(`Failed to load any model. Last error: ${error.message}`);
                }
            }
        }
    }

    /**
     * Load a single model from Hugging Face Hub
     */
    async loadSingleModel(modelRepo) {
        const fetchAllFiles = async (basePath, fileList) => {
            console.log(`🔍 Fetching files from ${basePath}...`);
            console.log(`📋 File list:`, fileList);

            const responses = await Promise.all(
                fileList.map(async (file) => {
                    const url = `${basePath}/${file}`;
                    console.log(`📄 Fetching ${file}...`);
                    const response = await fetch(url);
                    if (!response.ok) {
                        throw new Error(`File not found: ${url} (${response.status})`);
                    }
                    return response;
                })
            );

            console.log(`✅ All files fetched successfully`);
            return Promise.all(
                responses.map(res => res.arrayBuffer().then(b => new Uint8Array(b)))
            );
        };

        // For browser demo, skip local and go directly to Hugging Face
        console.log('🌐 Downloading from Hugging Face Hub...');

        let modelFiles;
        try {
            // Try primary file structure first
            console.log('🧪 Trying primary file structure...');
            modelFiles = await fetchAllFiles(
                `https://huggingface.co/${modelRepo}/resolve/main`,
                this.requiredFiles
            );
            console.log('📥 Downloaded model with primary file structure');
        } catch (primaryError) {
            console.warn('⚠️ Primary file structure failed:', primaryError.message);
            try {
                // Try alternative file structure
                console.log('🧪 Trying alternative file structure...');
                modelFiles = await fetchAllFiles(
                    `https://huggingface.co/${modelRepo}/resolve/main`,
                    this.alternativeFiles
                );
                console.log('📥 Downloaded model with alternative file structure');
            } catch (alternativeError) {
                console.error(`❌ Both file structures failed for ${modelRepo}`);
                console.error('Primary error:', primaryError.message);
                console.error('Alternative error:', alternativeError.message);
                throw new Error(`Failed to download ${modelRepo}: ${alternativeError.message}`);
            }
        }

        const [
            tokenizer,
            model,
            config,
            stConfig,
            dense,
            denseConfig,
            dense2,         // NEW: 2_Dense weights
            dense2Config,   // NEW: 2_Dense config
            tokensConfig,
        ] = modelFiles;

        // Initialize the ColBERT model with pylate-rs
        console.log('🔧 Initializing ColBERT model with 2_Dense support...');
        console.log('🔍 Model files being passed to ColBERT constructor:', {
            model: model ? 'present' : 'missing',
            dense: dense ? 'present' : 'missing',
            dense2: dense2 ? 'present (2_Dense!)' : 'missing',  // NEW
            tokenizer: tokenizer ? 'present' : 'missing',
            config: config ? 'present' : 'missing',
            stConfig: stConfig ? 'present' : 'missing',
            denseConfig: denseConfig ? 'present' : 'missing',
            dense2Config: dense2Config ? 'present (2_Dense!)' : 'missing',  // NEW
            tokensConfig: tokensConfig ? 'present' : 'missing'
        });

        try {
            // Debug: Check all config files
            console.log('🔍 DEBUG: Checking config files...');

            const configText = new TextDecoder().decode(config);
            const configObj = JSON.parse(configText);
            console.log('🔍 config.json architectures:', configObj.architectures);
            console.log('🔍 config.json size:', config.byteLength, 'bytes');

            const denseConfigText = new TextDecoder().decode(denseConfig);
            const denseConfigObj = JSON.parse(denseConfigText);
            console.log('🔍 1_Dense/config.json:', denseConfigObj);

            if (dense2Config) {
                const dense2ConfigText = new TextDecoder().decode(dense2Config);
                const dense2ConfigObj = JSON.parse(dense2ConfigText);
                console.log('🔍 2_Dense/config.json:', dense2ConfigObj);
            }

            // OVERRIDE: Enable query expansion with 32 tokens
            console.log('🔧 Enabling query expansion...');
            const stConfigText = new TextDecoder().decode(stConfig);
            const stConfigObj = JSON.parse(stConfigText);
            console.log('🔍 Original config_sentence_transformers.json:', stConfigObj);

            // Enable query expansion and set query length to 32
            // Note: attend_to_expansion_tokens = false means [MASK] tokens are used during
            // encoding but not included in output. This makes queries more distinct while
            // keeping output compact.
            stConfigObj.do_query_expansion = true;
            stConfigObj.query_length = 32;

            console.log('✅ Modified config_sentence_transformers.json:', stConfigObj);

            // Re-encode the modified config
            const modifiedStConfigText = JSON.stringify(stConfigObj);
            const modifiedStConfig = new TextEncoder().encode(modifiedStConfigText);

            // WASM signature: from_bytes(weights, dense_weights, dense2_weights, tokenizer, config,
            //                            sentence_transformers_config, dense_config, dense2_config,
            //                            special_tokens_map, batch_size)
            console.log('🔧 Parameter sizes:');
            console.log('  model:', model?.byteLength, 'bytes');
            console.log('  dense:', dense?.byteLength, 'bytes');
            console.log('  dense2:', dense2?.byteLength, 'bytes');
            console.log('  tokenizer:', tokenizer?.byteLength, 'bytes');
            console.log('  config:', config?.byteLength, 'bytes');
            console.log('  stConfig:', stConfig?.byteLength, 'bytes');
            console.log('  denseConfig:', denseConfig?.byteLength, 'bytes');
            console.log('  dense2Config:', dense2Config?.byteLength, 'bytes');
            console.log('  tokensConfig:', tokensConfig?.byteLength, 'bytes');

            // Try old signature first (8 params - no 2_Dense)
            console.log('🧪 Trying OLD signature (without 2_Dense support)...');
            try {
                this.model = new ColBERT(
                    model,            // weights
                    dense,            // dense_weights
                    tokenizer,        // tokenizer
                    config,           // config
                    modifiedStConfig, // sentence_transformers_config (with query expansion enabled!)
                    denseConfig,      // dense_config
                    tokensConfig,     // special_tokens_map
                    32                // batch_size
                );
                console.log('✅ OLD signature worked! WASM does NOT have 2_Dense support yet');
                console.warn('⚠️ Output will be 512-dim (not 48-dim) - need to rebuild WASM with 2_Dense');
                this.embeddingDim = 512; // Update to actual output
            } catch (oldError) {
                console.log('❌ OLD signature failed:', oldError);
                console.log('🧪 Trying NEW signature (with 2_Dense)...');
                this.model = new ColBERT(
                    model,            // weights (model.safetensors)
                    dense,            // dense_weights (1_Dense/model.safetensors)
                    dense2,           // dense2_weights (2_Dense/model.safetensors)
                    tokenizer,        // tokenizer (tokenizer.json)
                    config,           // config (config.json)
                    modifiedStConfig, // sentence_transformers_config (with query expansion enabled!)
                    denseConfig,      // dense_config (1_Dense/config.json)
                    dense2Config,     // dense2_config (2_Dense/config.json)
                    tokensConfig,     // special_tokens_map (special_tokens_map.json)
                    32                // batch_size
                );
                console.log('✅ NEW signature worked! WASM has 2_Dense support!');
                console.log('📊 Expected output dimension: 48 (with 2_Dense: 256→512→48)');
            }
        } catch (error) {
            console.error('❌ ColBERT constructor failed:', error);
            console.error('Error type:', typeof error);
            console.error('Error toString:', String(error));
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
            throw error;
        }
    }

    /**
     * Encode text into ColBERT embeddings using real model or simulation
     * @param {string} text - Input text to encode
     * @param {boolean} isQuery - Whether this is a query (true) or document (false)
     * @returns {Promise<Object>} - Token-level embeddings with metadata
     */
    async encodeText(text, isQuery = true) {
        if (!this.modelLoaded) {
            throw new Error('Model not loaded. Call initializeModel() first.');
        }

        const textType = isQuery ? 'query' : 'document';
        console.log(`🔤 Encoding ${textType}: "${text.substring(0, 50)}..."`);

        const startTime = performance.now();

        if (this.model && !this.simulationMode) {
            // Use real pylate-rs model
            try {
                // Use the correct pylate-rs API based on your working code
                const rawResult = await this.model.encode({
                    sentences: [text],
                    is_query: isQuery
                });

                console.log(`✅ Raw ${textType} result received`);
                console.log(`🔍 Result type: ${typeof rawResult}`);
                console.log(`🔍 Result structure:`, rawResult);

                // DEBUG: Deep inspection of the result structure
                if (rawResult && rawResult.embeddings) {
                    console.log(`🔍 DEBUG: rawResult.embeddings is Array:`, Array.isArray(rawResult.embeddings));
                    console.log(`🔍 DEBUG: rawResult.embeddings.length:`, rawResult.embeddings.length);
                    if (rawResult.embeddings[0]) {
                        console.log(`🔍 DEBUG: rawResult.embeddings[0] is Array:`, Array.isArray(rawResult.embeddings[0]));
                        console.log(`🔍 DEBUG: rawResult.embeddings[0].length:`, rawResult.embeddings[0].length);
                        if (rawResult.embeddings[0][0]) {
                            console.log(`🔍 DEBUG: rawResult.embeddings[0][0] is Array:`, Array.isArray(rawResult.embeddings[0][0]));
                            console.log(`🔍 DEBUG: rawResult.embeddings[0][0].length:`, rawResult.embeddings[0][0].length);
                            console.log(`🔍 DEBUG: First token vector (first 10 values):`, rawResult.embeddings[0][0].slice(0, 10));
                        }
                    }
                }

                // Handle ColBERT result format based on your working code
                let embeddings;
                if (Array.isArray(rawResult)) {
                    console.log('✅ Using direct array result');
                    embeddings = rawResult;
                } else if (rawResult && rawResult.embeddings && Array.isArray(rawResult.embeddings)) {
                    console.log('✅ Using result.embeddings array, length:', rawResult.embeddings.length);
                    // Unwrap the first sentence's embedding (we passed one sentence)
                    embeddings = rawResult.embeddings[0];
                    console.log('🔍 Unwrapped embeddings length:', embeddings?.length);
                } else if (rawResult && rawResult.data && Array.isArray(rawResult.data)) {
                    console.log('✅ Using result.data array');
                    embeddings = rawResult.data[0]; // Unwrap first sentence
                } else if (rawResult && Array.isArray(rawResult[0])) {
                    console.log('✅ Using result[0] array');
                    embeddings = rawResult[0];
                } else {
                    console.error('❌ Unknown result format:', rawResult);
                    throw new Error('Unknown ColBERT result format');
                }

                // Validate the embeddings structure
                if (!Array.isArray(embeddings) || embeddings.length === 0) {
                    console.error('❌ Invalid embeddings structure:', embeddings);
                    throw new Error('ColBERT returned empty or invalid embeddings');
                }

                console.log(`✅ Final ${textType} embeddings: array with ${embeddings.length} token vectors`);
                console.log('🔍 First token vector length:', embeddings[0]?.length);

                // Flatten the token vectors into a single array for FastPlaid compatibility
                const flatEmbeddings = embeddings.flat();

                // Calculate token count and dimensions
                const numTokens = embeddings.length;
                const tokenDim = embeddings[0]?.length || this.embeddingDim;

                // Update embedding dimension if it changed
                if (tokenDim !== this.embeddingDim) {
                    console.log(`🔧 Updating embeddingDim from ${this.embeddingDim} to ${tokenDim}`);
                    this.embeddingDim = tokenDim;
                }

                console.log(`📊 Token count: ${numTokens}, Token dimension: ${tokenDim}`);

                const encodingTime = performance.now() - startTime;
                this.timings.modelEncoding.push(encodingTime);
                // Silent per-document encoding (see final index time instead)

                return {
                    embeddings: new Float32Array(flatEmbeddings),
                    shape: [1, numTokens, tokenDim],
                    numTokens: numTokens,
                    tokenDim: tokenDim,
                    isReal: true,
                    encodingTime: encodingTime
                };
            } catch (error) {
                console.error(`❌ Real ${textType} encoding failed, falling back to simulation:`, error);
                console.error('Error details:', error);
                return this.simulateEncoding(text, startTime);
            }
        } else {
            // Use simulation mode
            return this.simulateEncoding(text, startTime);
        }
    }

    /**
     * Simulate encoding for demo purposes
     */
    simulateEncoding(text, startTime) {
        if (!startTime) startTime = performance.now();
        const tokens = this.tokenizeText(text);
        const numTokens = Math.min(tokens.length, 32); // Limit to 32 tokens for demo
        const embeddings = new Float32Array(numTokens * this.embeddingDim);

        // Generate normalized random embeddings that look like real ColBERT embeddings
        for (let i = 0; i < numTokens; i++) {
            let norm = 0;
            for (let j = 0; j < this.embeddingDim; j++) {
                const val = (Math.random() - 0.5) * 0.2; // Small random values
                embeddings[i * this.embeddingDim + j] = val;
                norm += val * val;
            }

            // Normalize the embedding vector
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (let j = 0; j < this.embeddingDim; j++) {
                    embeddings[i * this.embeddingDim + j] /= norm;
                }
            }
        }

        const encodingTime = performance.now() - startTime;
        this.timings.modelEncoding.push(encodingTime);
        console.log(`🎭 Simulated embeddings: ${numTokens} tokens × ${this.embeddingDim} dimensions`);
        console.log(`⏱️ Simulation encoding time: ${encodingTime.toFixed(2)}ms`);

        return {
            embeddings: embeddings,
            shape: [1, numTokens, this.embeddingDim],
            numTokens: numTokens,
            isReal: false,
            encodingTime: encodingTime
        };
    }

    /**
     * Simple tokenization (placeholder)
     * In reality, this would use the model's tokenizer
     */
    tokenizeText(text) {
        // Simple word-based tokenization for demo
        const words = text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 0);

        // Add special tokens
        return ['[CLS]', ...words, '[SEP]'];
    }

    /**
     * Load a specific model by ID
     */
    async loadSpecificModel(modelId) {
        console.log(`🎯 Loading specific model: ${modelId}`);
        this.modelRepo = modelId;
        this.simulationMode = false;
        this.model = null;

        try {
            await this.loadSingleModel(modelId);
            console.log(`✅ Successfully loaded ${modelId}`);
        } catch (error) {
            console.warn(`❌ Failed to load ${modelId}, falling back to simulation:`, error.message);
            this.simulationMode = true;
        }
    }

    /**
     * Create embeddings for all documents using real model or simulation
     */
    async createDocumentIndex(documents) {
        const indexStartTime = performance.now();
        console.log(`📚 Creating document index for ${documents.length} documents...`);

        const documentEmbeddings = [];
        for (const doc of documents) {
            const fullText = `${doc.title} ${doc.content}`;
            // Encode as document (is_query: false)
            const result = await this.encodeText(fullText, false);
            documentEmbeddings.push({
                id: doc.id,
                title: doc.title,
                content: doc.content, // Include original content
                embeddings: result.embeddings,
                shape: result.shape,
                numTokens: result.numTokens,
                isReal: result.isReal,
                encodingTime: result.encodingTime
            });
        }

        // Create FastPlaid index from document embeddings
        await this.createFastPlaidIndex(documentEmbeddings);

        const indexTime = performance.now() - indexStartTime;
        this.timings.indexTime.push(indexTime);

        const realCount = documentEmbeddings.filter(doc => doc.isReal).length;
        const simCount = documentEmbeddings.length - realCount;

        // Calculate memory usage for Direct MaxSim
        // Direct MaxSim stores embeddings as JavaScript arrays
        let totalEmbeddingsLength = 0;
        for (const doc of documentEmbeddings) {
            totalEmbeddingsLength += doc.embeddings.length;
        }

        // JavaScript Float32Arrays use 4 bytes per element, regular arrays also store as numbers (8 bytes each)
        // We'll estimate based on Float32Array (optimistic) vs number arrays (pessimistic)
        this.indexMemory.directMaxSim.embeddingsBytes = totalEmbeddingsLength * 8; // JS numbers are 8 bytes (64-bit float)

        // Metadata includes: id, title, content strings, numTokens, isReal, etc.
        // Rough estimate: 200 bytes per document for metadata overhead
        this.indexMemory.directMaxSim.metadataBytes = documentEmbeddings.length * 200;

        this.indexMemory.directMaxSim.totalBytes = this.indexMemory.directMaxSim.embeddingsBytes +
                                                     this.indexMemory.directMaxSim.metadataBytes;
        this.indexMemory.directMaxSim.documentCount = documentEmbeddings.length;
        this.indexMemory.directMaxSim.embeddingDim = documentEmbeddings[0]?.embeddings.length / documentEmbeddings[0]?.numTokens || 0;

        console.log(`✅ Document index created: ${realCount} real embeddings, ${simCount} simulated in ${indexTime.toFixed(2)}ms`);
        console.log(`📊 Direct MaxSim Memory: ${(this.indexMemory.directMaxSim.totalBytes / 1024 / 1024).toFixed(2)} MB`);

        return documentEmbeddings;
    }

    /**
     * Create FastPlaid index from document embeddings
     */
    async createFastPlaidIndex(documentEmbeddings) {
        console.log('🏗️ Creating FastPlaid index from document embeddings...');

        if (!window.fastPlaid) {
            console.warn('⚠️ FastPlaid WASM not available, skipping index creation');
            return;
        }

        try {
            // Prepare data for WASM
            // 1. Flatten all document embeddings into a single Float32Array
            // 2. Create doc_info array with [id, num_tokens] pairs

            let totalEmbeddings = 0;
            for (const doc of documentEmbeddings) {
                totalEmbeddings += doc.embeddings.length;
            }

            console.log(`📊 Preparing ${documentEmbeddings.length} documents, ${totalEmbeddings} total embeddings`);

            const allEmbeddings = new Float32Array(totalEmbeddings);
            const docInfo = new BigInt64Array(documentEmbeddings.length * 2); // [id, num_tokens] pairs

            let offset = 0;
            for (let i = 0; i < documentEmbeddings.length; i++) {
                const doc = documentEmbeddings[i];

                // Copy embeddings
                allEmbeddings.set(doc.embeddings, offset);
                offset += doc.embeddings.length;

                // Set doc info (using BigInt)
                docInfo[i * 2] = BigInt(doc.id);
                docInfo[i * 2 + 1] = BigInt(doc.numTokens);
            }

            console.log(`✅ Prepared ${allEmbeddings.length} embeddings, doc_info: ${docInfo.length} entries`);

            // Load documents into WASM
            window.fastPlaid.load_documents(allEmbeddings, docInfo);

            // Calculate memory usage for FastPlaid index
            this.indexMemory.fastPlaid.embeddingsBytes = allEmbeddings.length * 4; // Float32 = 4 bytes
            this.indexMemory.fastPlaid.metadataBytes = docInfo.length * 8; // BigInt64 = 8 bytes
            this.indexMemory.fastPlaid.totalBytes = this.indexMemory.fastPlaid.embeddingsBytes +
                                                     this.indexMemory.fastPlaid.metadataBytes;
            this.indexMemory.fastPlaid.documentCount = documentEmbeddings.length;
            this.indexMemory.fastPlaid.embeddingDim = documentEmbeddings[0]?.embeddings.length / documentEmbeddings[0]?.numTokens || 0;

            console.log(`✅ FastPlaid index created with ${documentEmbeddings.length} documents`);
            console.log(`📊 FastPlaid Memory: ${(this.indexMemory.fastPlaid.totalBytes / 1024 / 1024).toFixed(2)} MB`);

        } catch (error) {
            console.error('❌ Failed to create FastPlaid index:', error);
            console.error('Error details:', error.stack);
            // Continue without FastPlaid index - will fall back to direct MaxSim
        }
    }

    /**
     * Create FastPlaid Quantized index from document embeddings (4-bit compression)
     */
    async createFastPlaidQuantizedIndex(documentEmbeddings) {
        console.log('🗜️ Creating FastPlaid Quantized index (4-bit) from document embeddings...');

        if (!window.fastPlaidQuantized) {
            console.warn('⚠️ FastPlaidQuantized WASM not available, skipping index creation');
            return;
        }

        try {
            // Prepare data for WASM
            // 1. Flatten all document embeddings into a single Float32Array
            // 2. Create doc_info array with [id, num_tokens] pairs

            let totalEmbeddings = 0;
            for (const doc of documentEmbeddings) {
                totalEmbeddings += doc.embeddings.length;
            }

            console.log(`📊 Preparing ${documentEmbeddings.length} documents, ${totalEmbeddings} total embeddings for 4-bit quantization`);

            const allEmbeddings = new Float32Array(totalEmbeddings);
            const docInfo = new BigInt64Array(documentEmbeddings.length * 2); // [id, num_tokens] pairs

            let offset = 0;
            for (let i = 0; i < documentEmbeddings.length; i++) {
                const doc = documentEmbeddings[i];

                // Copy embeddings
                allEmbeddings.set(doc.embeddings, offset);
                offset += doc.embeddings.length;

                // Set doc info (using BigInt)
                docInfo[i * 2] = BigInt(doc.id);
                docInfo[i * 2 + 1] = BigInt(doc.numTokens);
            }

            console.log(`✅ Prepared ${allEmbeddings.length} embeddings, doc_info: ${docInfo.length} entries`);

            // Load documents into WASM with 4-bit quantization
            window.fastPlaidQuantized.load_documents_quantized(allEmbeddings, docInfo);

            // Calculate memory usage for FastPlaid Quantized index
            // 4-bit quantization gives roughly 8x compression
            const estimatedCompressedBytes = allEmbeddings.length * 4 / 8; // ~8x compression
            this.indexMemory.fastPlaid.embeddingsBytes = estimatedCompressedBytes;
            this.indexMemory.fastPlaid.metadataBytes = docInfo.length * 8; // BigInt64 = 8 bytes
            this.indexMemory.fastPlaid.totalBytes = this.indexMemory.fastPlaid.embeddingsBytes +
                                                     this.indexMemory.fastPlaid.metadataBytes;
            this.indexMemory.fastPlaid.documentCount = documentEmbeddings.length;
            this.indexMemory.fastPlaid.embeddingDim = documentEmbeddings[0]?.embeddings.length / documentEmbeddings[0]?.numTokens || 0;

            console.log(`✅ FastPlaid Quantized (4-bit) index created with ${documentEmbeddings.length} documents`);
            console.log(`📊 FastPlaid Quantized Memory: ${(this.indexMemory.fastPlaid.totalBytes / 1024 / 1024).toFixed(2)} MB`);
            console.log(`🗜️ Compression ratio: ~8x (from ${(allEmbeddings.length * 4 / 1024 / 1024).toFixed(2)} MB to ${(estimatedCompressedBytes / 1024 / 1024).toFixed(2)} MB)`);

        } catch (error) {
            console.error('❌ Failed to create FastPlaid Quantized index:', error);
            console.error('Error details:', error.stack);
            // Continue without FastPlaid index - will fall back to direct MaxSim
        }
    }

    /**
     * Create a sample document index for demonstration (100 documents)
     */
    createSampleDocuments() {
        return [
            // Core ML/AI Documents (1-20)
            {
                id: 1,
                title: "Introduction to Machine Learning",
                content: "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It encompasses supervised, unsupervised, and reinforcement learning paradigms."
            },
            {
                id: 2,
                title: "Deep Learning Algorithms Overview",
                content: "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data, including convolutional and recurrent neural networks. These architectures have revolutionized computer vision and natural language processing."
            },
            {
                id: 3,
                title: "Neural Networks and Backpropagation",
                content: "Backpropagation is the fundamental algorithm for training neural networks, using gradient descent to minimize the loss function. It efficiently computes gradients through the chain rule across multiple layers."
            },
            {
                id: 4,
                title: "Supervised Learning Techniques",
                content: "Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data, including classification and regression tasks. Popular methods include decision trees, support vector machines, and neural networks."
            },
            {
                id: 5,
                title: "Unsupervised Learning Methods",
                content: "Unsupervised learning discovers hidden patterns in data without labeled examples, including clustering, dimensionality reduction, and association rules. K-means, PCA, and autoencoders are common techniques."
            },
            {
                id: 6,
                title: "Natural Language Processing with Transformers",
                content: "Transformer models like BERT and GPT have revolutionized natural language processing through attention mechanisms and pre-training on large text corpora. They excel at understanding context and generating coherent text."
            },
            {
                id: 7,
                title: "Computer Vision and Convolutional Networks",
                content: "Convolutional neural networks excel at image recognition tasks by learning hierarchical features through convolution and pooling operations. They have achieved human-level performance on many visual recognition benchmarks."
            },
            {
                id: 8,
                title: "Reinforcement Learning Fundamentals",
                content: "Reinforcement learning trains agents to make decisions in environments by learning from rewards and penalties through trial and error. Q-learning and policy gradient methods are fundamental approaches."
            },
            {
                id: 9,
                title: "ColBERT: Efficient and Effective Passage Retrieval",
                content: "ColBERT is a ranking model that adapts deep bidirectional representations for efficient retrieval via late interaction. It uses token-level embeddings and MaxSim operations to achieve state-of-the-art effectiveness with high efficiency."
            },
            {
                id: 10,
                title: "Dense Passage Retrieval for Open-Domain QA",
                content: "Dense passage retrieval uses dense representations learned by encoding passages into low-dimensional continuous space. This approach significantly outperforms sparse retrieval methods like BM25 for question answering tasks."
            },
            {
                id: 11,
                title: "PLAID: An Efficient Engine for Late Interaction Retrieval",
                content: "PLAID accelerates ColBERT's late interaction through centroid interaction and decompression optimizations. It reduces storage requirements while maintaining retrieval quality through efficient indexing strategies."
            },
            {
                id: 12,
                title: "mixedbread-ai: Embedding Models for Retrieval",
                content: "mixedbread-ai develops high-quality embedding models optimized for retrieval tasks. Their mxbai-embed and mxbai-rerank models achieve strong performance on MTEB benchmarks while being efficient for production use."
            },
            {
                id: 13,
                title: "Pylate: Python Library for Late Interaction",
                content: "Pylate is a Python library that implements ColBERT and other late interaction models for efficient neural information retrieval. It provides easy-to-use APIs for indexing and searching large document collections."
            },
            {
                id: 14,
                title: "FastPlaid: Accelerated PLAID Implementation",
                content: "FastPlaid is an optimized implementation of the PLAID indexing system, designed for high-performance retrieval with ColBERT models. It uses advanced quantization and compression techniques for memory efficiency."
            },
            {
                id: 15,
                title: "Sentence Transformers and Semantic Search",
                content: "Sentence Transformers provide an easy method to compute dense vector representations for sentences and paragraphs. They enable semantic search by finding similar sentences based on meaning rather than keyword matching."
            },
            {
                id: 16,
                title: "Vector Databases and Similarity Search",
                content: "Vector databases like Pinecone, Weaviate, and Chroma are designed to store and query high-dimensional embeddings efficiently. They use approximate nearest neighbor algorithms like HNSW and IVF for fast similarity search."
            },
            {
                id: 17,
                title: "Retrieval-Augmented Generation (RAG)",
                content: "RAG combines retrieval systems with generative models to provide factual and up-to-date responses. It retrieves relevant documents and uses them as context for generating accurate answers to user queries."
            },
            {
                id: 18,
                title: "BERT and Bidirectional Encoder Representations",
                content: "BERT revolutionized NLP by introducing bidirectional training of transformers. It learns deep bidirectional representations by jointly conditioning on both left and right context in all layers."
            },
            {
                id: 19,
                title: "Attention Mechanisms in Neural Networks",
                content: "Attention mechanisms allow models to focus on relevant parts of the input when making predictions. Self-attention, as used in transformers, enables modeling of long-range dependencies in sequences."
            },
            {
                id: 20,
                title: "WebAssembly for Machine Learning",
                content: "WebAssembly (WASM) enables running machine learning models efficiently in web browsers. It provides near-native performance for compute-intensive tasks like neural network inference and vector operations."
            },

            // Advanced AI Topics (21-40)
            {
                id: 21,
                title: "Graph Neural Networks and Geometric Deep Learning",
                content: "Graph neural networks extend deep learning to non-Euclidean data structures like graphs and manifolds. They enable learning on social networks, molecular structures, and knowledge graphs through message passing and attention mechanisms."
            },
            {
                id: 22,
                title: "Federated Learning and Privacy-Preserving AI",
                content: "Federated learning enables training machine learning models across decentralized data sources without centralizing sensitive information. It preserves privacy while enabling collaborative learning across organizations and devices."
            },
            {
                id: 23,
                title: "Multimodal AI and Cross-Modal Understanding",
                content: "Multimodal AI systems process and understand information from multiple modalities like text, images, audio, and video. These systems enable applications like image captioning, visual question answering, and cross-modal retrieval."
            },
            {
                id: 24,
                title: "Explainable AI and Model Interpretability",
                content: "Explainable AI focuses on making machine learning models more transparent and interpretable. Techniques like LIME, SHAP, and attention visualization help understand model decisions and build trust in AI systems."
            },
            {
                id: 25,
                title: "AutoML and Neural Architecture Search",
                content: "Automated machine learning (AutoML) automates the process of model selection, hyperparameter tuning, and feature engineering. Neural architecture search discovers optimal network architectures for specific tasks and constraints."
            },

            // Fun & Cute Documents (26-50)
            {
                id: 26,
                title: "The Secret Life of Pandas: Data Science Edition",
                content: "Pandas aren't just adorable bears - they're also the backbone of data science! The pandas library helps data scientists wrangle, clean, and analyze data with the same grace that real pandas munch bamboo. Both are essential for their respective ecosystems."
            },
            {
                id: 27,
                title: "Why Cats Would Make Terrible Machine Learning Engineers",
                content: "Cats would be awful at machine learning because they'd knock over the training data, sleep on the keyboard during model training, and only work when they feel like it. However, they'd excel at anomaly detection - they always know when something's not quite right."
            },
            {
                id: 28,
                title: "The Great Debugging Adventure: A Developer's Quest",
                content: "Once upon a time, in a land of infinite loops and mysterious segfaults, a brave developer embarked on a quest to find the legendary bug that had been crashing their code. Armed with print statements and rubber ducks, they ventured forth into the dark forest of legacy code."
            },
            {
                id: 29,
                title: "Coffee-Driven Development: A Programmer's Guide",
                content: "Coffee is the fuel that powers the software industry. Studies show that code quality is directly proportional to caffeine intake, with optimal performance achieved at exactly 3.7 cups per day. Side effects may include jittery typing and the ability to see bugs in your sleep."
            },
            {
                id: 30,
                title: "The Zen of Python: Beautiful Code Philosophy",
                content: "Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex. The Zen of Python teaches us that code should be poetry - elegant, readable, and bringing joy to those who encounter it. Namespaces are one honking great idea!"
            },
            {
                id: 31,
                title: "Rubber Duck Debugging: The Quack Method",
                content: "Rubber duck debugging is a real technique where programmers explain their code to a rubber duck. The act of verbalizing the problem often leads to the solution. The duck doesn't judge, doesn't interrupt, and never asks for a raise. Perfect colleague!"
            },
            {
                id: 32,
                title: "The Mysterious Case of the Missing Semicolon",
                content: "Detective JavaScript was on the case of the missing semicolon that had brought an entire application to its knees. After hours of investigation, the culprit was found hiding at the end of line 247, causing chaos throughout the codebase. Justice was served with a simple keystroke."
            },
            {
                id: 33,
                title: "Unicorns in Tech: Mythical Creatures and Billion-Dollar Startups",
                content: "In the tech world, unicorns are rare startups valued at over $1 billion. Unlike their mythical counterparts, these unicorns don't have horns or magical powers, but they do have the ability to make investors very happy and competitors very nervous."
            },
            {
                id: 34,
                title: "The Art of Naming Variables: A Comedy of Errors",
                content: "Variable naming is an art form. From the classic 'temp' and 'data' to the mysterious 'x' and 'foo', developers have created a rich tapestry of confusing identifiers. Future you will thank present you for using descriptive names like 'userAccountBalance' instead of 'uab'."
            },
            {
                id: 35,
                title: "Stack Overflow: The Developer's Best Friend",
                content: "Stack Overflow is where developers go to find answers, copy code snippets, and occasionally contribute back to the community. It's estimated that 90% of all software is held together by Stack Overflow answers and prayer. The other 10% is just prayer."
            },

            // Tech Culture & Humor (36-60)
            {
                id: 36,
                title: "The Evolution of Programming Languages: From COBOL to Rust",
                content: "Programming languages evolve like species, with some going extinct (RIP COBOL... well, mostly) and others thriving in new environments. Rust promises memory safety, Go offers simplicity, and JavaScript continues to run everywhere, whether you want it to or not."
            },
            {
                id: 37,
                title: "Git Commit Messages: A Window into the Developer's Soul",
                content: "Git commit messages reveal the true state of a developer's mind: 'Fixed bug', 'Actually fixed bug', 'Fixed bug for real this time', 'I hate everything', 'Added semicolon', 'Removed semicolon', 'Why does this work?', 'Magic, do not touch'."
            },
            {
                id: 38,
                title: "The Twelve Days of DevOps: A Holiday Carol",
                content: "On the first day of DevOps, my manager gave to me: a pipeline that's failing. On the second day: two broken builds. On the third day: three merge conflicts. By the twelfth day, you have a partridge in a Docker tree and a very stressed-out team."
            },
            {
                id: 39,
                title: "Agile Methodology: The Art of Organized Chaos",
                content: "Agile development is like jazz - it looks chaotic from the outside, but there's actually a structure to the madness. Daily standups, sprint planning, and retrospectives create a rhythm that somehow produces working software. The magic is in the collaboration."
            },
            {
                id: 40,
                title: "The Cloud: Someone Else's Computer",
                content: "The cloud is just someone else's computer, but it's a really nice computer with excellent uptime, global distribution, and someone else handling the maintenance. It's like having a really reliable friend who never asks you to help them move."
            },
            {
                id: 41,
                title: "Microservices: Breaking Things into Smaller Pieces to Break",
                content: "Microservices architecture breaks monolithic applications into smaller, independent services. This means instead of one big thing that can break, you now have dozens of small things that can break in creative and unexpected ways. Progress!"
            },
            {
                id: 42,
                title: "The Blockchain: A Solution Looking for a Problem",
                content: "Blockchain technology is like a hammer - when you have one, everything starts to look like a nail. From cryptocurrency to supply chain management to digital art, blockchain enthusiasts believe it can solve any problem, even ones that don't exist yet."
            },
            {
                id: 43,
                title: "Artificial Intelligence: Teaching Machines to Think Like Humans",
                content: "AI researchers are working hard to make machines think like humans. The irony is that humans often don't think very logically, so we're essentially teaching computers to be inconsistent, biased, and occasionally brilliant. We're making great progress!"
            },
            {
                id: 44,
                title: "The Internet of Things: When Your Toaster Needs a Software Update",
                content: "The Internet of Things connects everyday objects to the internet, creating a world where your refrigerator can order milk and your toaster can get hacked. It's the future we never knew we needed, complete with security vulnerabilities in kitchen appliances."
            },
            {
                id: 45,
                title: "Quantum Computing: Schrödinger's Algorithm",
                content: "Quantum computing harnesses the weird properties of quantum mechanics to solve certain problems exponentially faster. Until you measure the result, your algorithm is simultaneously working and broken. It's like regular programming, but with more physics and existential dread."
            },

            // Programming Wisdom & Philosophy (46-70)
            {
                id: 46,
                title: "The Principle of Least Astonishment in Software Design",
                content: "Good software should behave in ways that don't surprise users. If clicking 'Save' launches a rocket to Mars, you've probably violated this principle. Design interfaces that match user expectations, unless you're actually building rocket launch software."
            },
            {
                id: 47,
                title: "Technical Debt: The Credit Card of Software Development",
                content: "Technical debt is like a credit card for code - it lets you ship features quickly now, but you'll pay interest later in the form of bugs, maintenance headaches, and developer tears. The minimum payment is never enough."
            },
            {
                id: 48,
                title: "The Bus Factor: Planning for the Inevitable",
                content: "The bus factor is the number of team members who need to get hit by a bus before your project is doomed. A bus factor of one means you're in trouble. Good documentation and knowledge sharing increase your bus factor and decrease your anxiety."
            },
            {
                id: 49,
                title: "Code Reviews: The Art of Constructive Criticism",
                content: "Code reviews are like peer editing for programmers. They catch bugs, improve code quality, and occasionally hurt feelings. The best reviews are thorough but kind, focusing on the code rather than the coder. 'This could be clearer' beats 'This is garbage.'"
            },
            {
                id: 50,
                title: "The Mythical Man-Month: Why Nine Women Can't Make a Baby in One Month",
                content: "Adding more programmers to a late project makes it later. This counterintuitive truth from Fred Brooks explains why throwing people at problems doesn't always work. Communication overhead grows quadratically with team size, while productivity doesn't."
            },

            // Modern Tech Trends (51-75)
            {
                id: 51,
                title: "Serverless Computing: Servers That Aren't Really Serverless",
                content: "Serverless computing means you don't manage servers, but servers still exist - they're just someone else's problem. It's like ordering takeout instead of cooking: you still need a kitchen, but it's not in your house and you don't have to clean it."
            },
            {
                id: 52,
                title: "Container Orchestration: Herding Digital Cats",
                content: "Container orchestration with Kubernetes is like being a shepherd for thousands of digital sheep, except the sheep are Docker containers and they occasionally decide to restart themselves at 3 AM. The shepherd (you) needs lots of coffee."
            },
            {
                id: 53,
                title: "Progressive Web Apps: The Best of Both Worlds",
                content: "Progressive Web Apps combine the reach of the web with the functionality of native apps. They work offline, send push notifications, and can be installed on devices. It's like having a website that thinks it's an app and mostly succeeds."
            },
            {
                id: 54,
                title: "Edge Computing: Bringing the Cloud Closer to Home",
                content: "Edge computing moves processing closer to where data is generated, reducing latency and improving performance. Instead of sending everything to distant data centers, we process data at the edge of the network. It's like having a mini-cloud in your neighborhood."
            },
            {
                id: 55,
                title: "WebAssembly: Running Native Code in Browsers",
                content: "WebAssembly lets you run code written in languages like C, C++, and Rust in web browsers at near-native speed. It's like having a universal translator for programming languages, enabling high-performance applications to run anywhere the web reaches."
            },
            {
                id: 56,
                title: "GraphQL: Asking for Exactly What You Need",
                content: "GraphQL is like ordering à la carte instead of getting a fixed menu. Clients can request exactly the data they need, nothing more, nothing less. It reduces over-fetching and under-fetching, making APIs more efficient and developers happier."
            },
            {
                id: 57,
                title: "JAMstack: JavaScript, APIs, and Markup",
                content: "JAMstack architecture pre-builds pages and serves them from CDNs, using JavaScript for dynamic functionality and APIs for server-side operations. It's fast, secure, and scalable - like having a sports car that's also a tank."
            },
            {
                id: 58,
                title: "Low-Code/No-Code: Democratizing Software Development",
                content: "Low-code and no-code platforms let non-programmers build applications using visual interfaces and drag-and-drop components. It's like LEGO for software - anyone can build something, though you might still need an architect for the really complex stuff."
            },
            {
                id: 59,
                title: "DevSecOps: Security as Code",
                content: "DevSecOps integrates security practices into the development pipeline from the start. Instead of bolting security on at the end, it's baked into every step. It's like having a security guard who's also a developer and knows where all the vulnerabilities hide."
            },
            {
                id: 60,
                title: "Observability: Knowing What Your System is Thinking",
                content: "Observability goes beyond monitoring to understand the internal state of systems based on their external outputs. It's like being a mind reader for your applications - you can tell what they're thinking even when they're not talking."
            },

            // Data Science & Analytics (61-80)
            {
                id: 61,
                title: "Data Lakes vs Data Warehouses: The Storage Wars",
                content: "Data lakes store raw data in its native format, while data warehouses store structured, processed data. It's like the difference between a messy garage where you throw everything and a well-organized closet. Both have their place, depending on your needs."
            },
            {
                id: 62,
                title: "ETL vs ELT: The Great Data Pipeline Debate",
                content: "ETL (Extract, Transform, Load) processes data before storing it, while ELT (Extract, Load, Transform) stores raw data first and processes it later. It's like deciding whether to wash dishes before putting them away or after taking them out of the cupboard."
            },
            {
                id: 63,
                title: "Real-Time Analytics: The Need for Speed",
                content: "Real-time analytics processes data as it arrives, providing immediate insights. It's like having a crystal ball that shows you what's happening right now instead of what happened yesterday. Great for fraud detection, less useful for predicting the weather next week."
            },
            {
                id: 64,
                title: "A/B Testing: The Scientific Method for Product Development",
                content: "A/B testing compares two versions of something to see which performs better. It's like having a controlled experiment for every product decision. Version A gets the red button, Version B gets the blue button, and statistics tell you which one users prefer."
            },
            {
                id: 65,
                title: "Data Visualization: Making Numbers Tell Stories",
                content: "Good data visualization turns spreadsheets into stories, revealing patterns and insights that would be invisible in raw data. It's like being a translator between the language of numbers and the language of humans. Bar charts are the universal translators."
            },

            // Cybersecurity & Privacy (66-85)
            {
                id: 66,
                title: "Zero Trust Security: Trust No One, Verify Everything",
                content: "Zero Trust security assumes that threats exist both inside and outside the network perimeter. Every user and device must be verified before accessing resources. It's like having a bouncer at every door in your house, even the bathroom."
            },
            {
                id: 67,
                title: "Encryption: The Art of Secret Keeping",
                content: "Encryption scrambles data so only authorized parties can read it. It's like writing in a secret code that only you and your friends know. The math behind it is complex, but the concept is simple: keep secrets secret."
            },
            {
                id: 68,
                title: "Social Engineering: Hacking Humans Instead of Computers",
                content: "Social engineering exploits human psychology rather than technical vulnerabilities. It's often easier to trick someone into giving you their password than to crack it. The weakest link in security is usually the human holding the other end of the chain."
            },
            {
                id: 69,
                title: "GDPR and Privacy by Design: Making Privacy a Feature",
                content: "GDPR and similar regulations make privacy a legal requirement, not just a nice-to-have. Privacy by design builds data protection into systems from the ground up. It's like having privacy as a core feature rather than an afterthought."
            },
            {
                id: 70,
                title: "Bug Bounties: Paying Hackers to Break Your Stuff",
                content: "Bug bounty programs pay security researchers to find vulnerabilities in your systems. It's like hiring burglars to test your locks, except it's legal and they help you fix the problems they find. Everyone wins, except the actual bad guys."
            },

            // Future Tech & Speculation (71-90)
            {
                id: 71,
                title: "Brain-Computer Interfaces: Thinking Your Way to the Internet",
                content: "Brain-computer interfaces could let us control devices with our thoughts. Imagine typing by thinking, browsing the web with your mind, or debugging code telepathically. The future might be hands-free, but hopefully not bug-free."
            },
            {
                id: 72,
                title: "Augmented Reality: Overlaying Digital on Physical",
                content: "Augmented reality adds digital information to the real world, like having a heads-up display for life. You could see code reviews floating above your coffee cup or debug information hovering over broken hardware. Reality, but with more features."
            },
            {
                id: 73,
                title: "Digital Twins: Virtual Copies of Everything",
                content: "Digital twins are virtual replicas of physical systems that update in real-time. You could have a digital copy of your data center, your car, or even yourself. It's like having a save game for reality that you can experiment with safely."
            },
            {
                id: 74,
                title: "Autonomous Systems: Teaching Machines to Make Decisions",
                content: "Autonomous systems make decisions without human intervention, from self-driving cars to automated trading systems. They're like having a really smart assistant who never sleeps, never gets tired, and occasionally makes spectacular mistakes."
            },
            {
                id: 75,
                title: "Synthetic Biology: Programming Life Itself",
                content: "Synthetic biology applies engineering principles to biological systems, essentially programming living organisms. It's like software development, but instead of debugging code, you're debugging DNA. The bugs are literally bugs."
            },

            // Meta & Philosophical (76-90)
            {
                id: 76,
                title: "The Singularity: When AI Becomes Smarter Than Us",
                content: "The technological singularity is the hypothetical point when AI surpasses human intelligence and begins improving itself. It's either the beginning of a golden age or the plot of every sci-fi movie ever made. Probably both."
            },
            {
                id: 77,
                title: "Digital Minimalism: Less is More in the Digital Age",
                content: "Digital minimalism advocates for intentional technology use, focusing on tools that truly add value to your life. It's like Marie Kondo for your digital life - if that app doesn't spark joy, delete it. Your attention span will thank you."
            },
            {
                id: 78,
                title: "The Attention Economy: Your Focus is the Product",
                content: "In the attention economy, human attention is the scarce resource that companies compete for. Social media platforms, news sites, and apps are all vying for your eyeballs. You're not the customer; you're the product being sold to advertisers."
            },
            {
                id: 79,
                title: "Digital Transformation: More Than Just Buying Computers",
                content: "Digital transformation isn't just about adopting new technology; it's about fundamentally changing how organizations operate and deliver value. It's like renovating your house while you're still living in it - messy, but necessary."
            },
            {
                id: 80,
                title: "The Future of Work: Humans and Machines Together",
                content: "The future of work isn't about humans versus machines; it's about humans with machines. AI will automate routine tasks, freeing humans to focus on creative, strategic, and interpersonal work. We'll be cyborgs, but with better job satisfaction."
            },

            // Bonus Fun Documents (81-100)
            {
                id: 81,
                title: "The Great Tabs vs Spaces Debate: A Holy War",
                content: "The tabs versus spaces debate has raged for decades, dividing programmers into two camps. Tabs are efficient and customizable, spaces are consistent and predictable. The real answer is: use whatever your team agrees on, and use a formatter to enforce it."
            },
            {
                id: 82,
                title: "Vim vs Emacs: The Editor Wars Continue",
                content: "Vim and Emacs users have been fighting for supremacy since the dawn of computing. Vim users can exit their editor (eventually), Emacs users have an operating system that happens to edit text. Both are powerful, both have steep learning curves."
            },
            {
                id: 83,
                title: "The Rubber Duck's Guide to Problem Solving",
                content: "From the perspective of a rubber duck: 'I sit quietly on desks, listening to programmers explain their code. I don't judge, I don't interrupt, I just listen. Somehow, this helps them find solutions. I'm basically a therapist, but waterproof.'"
            },
            {
                id: 84,
                title: "404 Not Found: The Internet's Most Famous Error",
                content: "The 404 error is the internet's way of saying 'I looked everywhere, but I can't find what you're looking for.' It's named after room 404 at CERN where the original web servers were kept. Now it's a cultural icon and the subject of countless creative error pages."
            },
            {
                id: 85,
                title: "Hello World: Every Programmer's First Words",
                content: "'Hello, World!' is traditionally the first program every programmer writes. It's like a baby's first words, but for code. Simple, universal, and the beginning of a beautiful relationship between human and computer. Some relationships last longer than others."
            },
            {
                id: 86,
                title: "The Infinite Scroll: A UX Pattern That Never Ends",
                content: "Infinite scroll keeps loading content as you scroll down, creating an endless stream of information. It's great for engagement, terrible for productivity. It's like a digital black hole that sucks in your time and attention. Resistance is futile."
            },
            {
                id: 87,
                title: "Dark Mode: The Developer's Natural Habitat",
                content: "Dark mode reduces eye strain and saves battery life, but more importantly, it makes you look like a hacker in movies. Light mode is for morning people and designers. Dark mode is for night owls and people who take their coding seriously."
            },
            {
                id: 88,
                title: "The Cookie Monster's Guide to Web Tracking",
                content: "Web cookies track user behavior across sites, like leaving breadcrumbs through the internet forest. Some cookies are helpful (remembering your login), others are creepy (following you everywhere). The Cookie Monster would approve of the name, if not the implementation."
            },
            {
                id: 89,
                title: "Responsive Design: Making Websites Work on Everything",
                content: "Responsive design ensures websites work on devices from smartwatches to billboards. It's like designing clothes that fit everyone from toddlers to giants. CSS media queries are the magic that makes it work, though sometimes it feels more like dark magic."
            },
            {
                id: 90,
                title: "The Cloud Native Mindset: Born in the Cloud",
                content: "Cloud native applications are designed specifically for cloud environments, embracing microservices, containers, and dynamic orchestration. They're like digital natives - they've never known a world without the internet, and they're perfectly adapted to their environment."
            },
            {
                id: 91,
                title: "API First Design: Building the Plumbing First",
                content: "API-first design means building the API before the user interface, ensuring that all functionality is accessible programmatically. It's like building the plumbing before the bathroom - not glamorous, but essential for everything else to work properly."
            },
            {
                id: 92,
                title: "The Twelve-Factor App: A Methodology for Modern Applications",
                content: "The Twelve-Factor App methodology provides guidelines for building scalable, maintainable applications. It covers everything from configuration management to logging. It's like a recipe for good software, though following recipes doesn't guarantee you won't burn dinner."
            },
            {
                id: 93,
                title: "Chaos Engineering: Breaking Things on Purpose",
                content: "Chaos engineering deliberately introduces failures into systems to test their resilience. It's like having a toddler in your data center - things will break, but you'll learn how to make them more robust. Netflix's Chaos Monkey is the most famous practitioner."
            },
            {
                id: 94,
                title: "The Gartner Hype Cycle: Riding the Waves of Technology",
                content: "The Gartner Hype Cycle tracks how technologies move from initial excitement through disillusionment to eventual productivity. Every new tech goes through this cycle - from 'This will change everything!' to 'This is useless!' to 'This is actually pretty useful.'"
            },
            {
                id: 95,
                title: "Conway's Law: Organizations Design Systems That Mirror Their Structure",
                content: "Conway's Law states that organizations design systems that mirror their communication structure. If you have four teams, you'll build a system with four components. It's like organizational DNA expressing itself in code architecture."
            },
            {
                id: 96,
                title: "The Peter Principle: Rising to Your Level of Incompetence",
                content: "The Peter Principle suggests that people in hierarchies rise to their level of incompetence. Great programmers become mediocre managers, great managers become terrible executives. It explains a lot about corporate life and why good technical people sometimes make bad leaders."
            },
            {
                id: 97,
                title: "Parkinson's Law: Work Expands to Fill Available Time",
                content: "Parkinson's Law states that work expands to fill the time available for its completion. Give someone a week to write a function, and they'll take a week. Give them a day, and they'll finish in a day. Deadlines are productivity's best friend."
            },
            {
                id: 98,
                title: "The Dunning-Kruger Effect: Confidence vs Competence",
                content: "The Dunning-Kruger effect describes how people with limited knowledge overestimate their competence. In programming, it's the difference between 'How hard can it be?' and 'Oh god, what have I done?' The more you learn, the more you realize you don't know."
            },
            {
                id: 99,
                title: "Imposter Syndrome: Feeling Like a Fraud",
                content: "Imposter syndrome is the feeling that you're not qualified for your job and that everyone will eventually figure out you're a fraud. It's incredibly common in tech, where the pace of change makes everyone feel like they're constantly behind. You're probably more competent than you think."
            },
            {
                id: 100,
                title: "The Joy of Coding: Why We Do What We Do",
                content: "Despite the bugs, the deadlines, and the occasional existential crisis, there's something magical about creating software. It's the joy of solving puzzles, building something from nothing, and occasionally making the world a little bit better. That's why we code, even when the code doesn't cooperate."
            }
        ];
    }

    /**
     * Create 100 documents with diverse content including cute/fun topics
     */
    create100Documents() {
        const docs = [
            // Original 20 ML/AI documents
            {
                id: 1,
                title: "Introduction to Machine Learning",
                content: "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It encompasses supervised, unsupervised, and reinforcement learning paradigms."
            },
            {
                id: 2,
                title: "Deep Learning Algorithms Overview",
                content: "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data, including convolutional and recurrent neural networks. These architectures have revolutionized computer vision and natural language processing."
            },
            {
                id: 3,
                title: "Neural Networks and Backpropagation",
                content: "Backpropagation is the fundamental algorithm for training neural networks, using gradient descent to minimize the loss function. It efficiently computes gradients through the chain rule across multiple layers."
            },
            {
                id: 4,
                title: "Supervised Learning Techniques",
                content: "Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data, including classification and regression tasks. Popular methods include decision trees, support vector machines, and neural networks."
            },
            {
                id: 5,
                title: "Unsupervised Learning Methods",
                content: "Unsupervised learning discovers hidden patterns in data without labeled examples, including clustering, dimensionality reduction, and association rules. K-means, PCA, and autoencoders are common techniques."
            },
            {
                id: 6,
                title: "Natural Language Processing with Transformers",
                content: "Transformer models like BERT and GPT have revolutionized natural language processing through attention mechanisms and pre-training on large text corpora. They excel at understanding context and generating coherent text."
            },
            {
                id: 7,
                title: "Computer Vision and Convolutional Networks",
                content: "Convolutional neural networks excel at image recognition tasks by learning hierarchical features through convolution and pooling operations. They have achieved human-level performance on many visual recognition benchmarks."
            },
            {
                id: 8,
                title: "Reinforcement Learning Fundamentals",
                content: "Reinforcement learning trains agents to make decisions in environments by learning from rewards and penalties through trial and error. Q-learning and policy gradient methods are fundamental approaches."
            },
            {
                id: 9,
                title: "ColBERT: Efficient and Effective Passage Retrieval",
                content: "ColBERT is a ranking model that adapts deep bidirectional representations for efficient retrieval via late interaction. It uses token-level embeddings and MaxSim operations to achieve state-of-the-art effectiveness with high efficiency."
            },
            {
                id: 10,
                title: "Dense Passage Retrieval for Open-Domain QA",
                content: "Dense passage retrieval uses dense representations learned by encoding passages into low-dimensional continuous space. This approach significantly outperforms sparse retrieval methods like BM25 for question answering tasks."
            },
            {
                id: 11,
                title: "PLAID: An Efficient Engine for Late Interaction Retrieval",
                content: "PLAID accelerates ColBERT's late interaction through centroid interaction and decompression optimizations. It reduces storage requirements while maintaining retrieval quality through efficient indexing strategies."
            },
            {
                id: 12,
                title: "mixedbread-ai: Embedding Models for Retrieval",
                content: "mixedbread-ai develops high-quality embedding models optimized for retrieval tasks. Their mxbai-embed and mxbai-rerank models achieve strong performance on MTEB benchmarks while being efficient for production use."
            },
            {
                id: 13,
                title: "Pylate: Python Library for Late Interaction",
                content: "Pylate is a Python library that implements ColBERT and other late interaction models for efficient neural information retrieval. It provides easy-to-use APIs for indexing and searching large document collections."
            },
            {
                id: 14,
                title: "FastPlaid: Accelerated PLAID Implementation",
                content: "FastPlaid is an optimized implementation of the PLAID indexing system, designed for high-performance retrieval with ColBERT models. It uses advanced quantization and compression techniques for memory efficiency."
            },
            {
                id: 15,
                title: "Sentence Transformers and Semantic Search",
                content: "Sentence Transformers provide an easy method to compute dense vector representations for sentences and paragraphs. They enable semantic search by finding similar sentences based on meaning rather than keyword matching."
            },
            {
                id: 16,
                title: "Vector Databases and Similarity Search",
                content: "Vector databases like Pinecone, Weaviate, and Chroma are designed to store and query high-dimensional embeddings efficiently. They use approximate nearest neighbor algorithms like HNSW and IVF for fast similarity search."
            },
            {
                id: 17,
                title: "Retrieval-Augmented Generation (RAG)",
                content: "RAG combines retrieval systems with generative models to provide factual and up-to-date responses. It retrieves relevant documents and uses them as context for generating accurate answers to user queries."
            },
            {
                id: 18,
                title: "BERT and Bidirectional Encoder Representations",
                content: "BERT revolutionized NLP by introducing bidirectional training of transformers. It learns deep bidirectional representations by jointly conditioning on both left and right context in all layers."
            },
            {
                id: 19,
                title: "Attention Mechanisms in Neural Networks",
                content: "Attention mechanisms allow models to focus on relevant parts of the input when making predictions. Self-attention, as used in transformers, enables modeling of long-range dependencies in sequences."
            },
            {
                id: 20,
                title: "WebAssembly for Machine Learning",
                content: "WebAssembly (WASM) enables running machine learning models efficiently in web browsers. It provides near-native performance for compute-intensive tasks like neural network inference and vector operations."
            }
        ];

        // Add 80 more diverse and fun documents
        const additionalDocs = [
            // Cute Animals (21-30)
            { id: 21, title: "Why Cats Purr: The Science Behind Feline Happiness", content: "Cats purr at frequencies between 20-50 Hz, which has been shown to promote bone healing and reduce pain. This adorable behavior serves multiple purposes including communication, self-soothing, and even healing. Kittens start purring within days of birth!" },
            { id: 22, title: "The Secret Life of Penguins: Antarctic Adventures", content: "Emperor penguins can dive up to 500 meters deep and hold their breath for 20 minutes while hunting for fish. These tuxedo-wearing birds huddle together in groups of thousands to survive Antarctic winters, taking turns being on the outside of the huddle." },
            { id: 23, title: "Golden Retrievers: The Ultimate Good Boys", content: "Golden Retrievers were originally bred in Scotland for retrieving waterfowl during hunting. Their gentle mouths can carry eggs without breaking them, and their friendly nature makes them perfect therapy dogs. They literally smile when happy!" },
            { id: 24, title: "Octopus Intelligence: Eight-Armed Geniuses", content: "Octopuses have three hearts, blue blood, and can solve complex puzzles. Each of their eight arms has its own brain, and they can change color and texture to camouflage perfectly with their surroundings. Some species use tools!" },
            { id: 25, title: "Baby Elephants and Their Trunk Training", content: "Baby elephants spend years learning to control their trunks, which contain over 40,000 muscles. They often trip over their trunks while learning to walk and suck their trunks for comfort, just like human babies suck their thumbs." },
            { id: 26, title: "Dolphins: The Ocean's Comedians", content: "Dolphins have names for each other (signature whistles) and can recognize themselves in mirrors. They play games, surf waves for fun, and have been observed using sea sponges as tools to protect their noses while foraging." },
            { id: 27, title: "Pandas: The Bamboo-Eating Teddy Bears", content: "Giant pandas spend 14 hours a day eating bamboo and can consume up to 40 pounds daily. Despite being bears, they're terrible at digesting bamboo and only absorb 17% of its nutrients. Baby pandas are pink, hairless, and smaller than a mouse!" },
            { id: 28, title: "Owls: Silent Hunters of the Night", content: "Owls have asymmetrical ear openings that help them pinpoint sounds in 3D space. Their flight is completely silent thanks to special feather structures, and they can rotate their heads 270 degrees because they have 14 neck vertebrae." },
            { id: 29, title: "Sea Otters: The Ocean's Tool Users", content: "Sea otters hold hands while sleeping to prevent drifting apart, use rocks as tools to crack open shellfish, and have the densest fur in the animal kingdom with up to 1 million hairs per square inch. They're basically aquatic teddy bears!" },
            { id: 30, title: "Hummingbirds: Tiny Flying Jewels", content: "Hummingbirds can fly backwards, upside down, and hover in place. Their hearts beat up to 1,260 times per minute, and they must eat every 10-15 minutes to survive. The smallest species weighs less than a penny!" },

            // Food & Cooking (31-40)
            { id: 31, title: "The Science of Perfect Pizza Dough", content: "Perfect pizza dough requires a balance of gluten development, fermentation time, and hydration. The Maillard reaction creates the golden crust, while proper fermentation develops complex flavors. Neapolitan pizza cooks at 900°F in just 90 seconds!" },
            { id: 32, title: "Why Chocolate Makes Us Happy", content: "Chocolate contains phenylethylamine and anandamide, compounds that trigger the release of endorphins and serotonin. Dark chocolate with 70% cacao has antioxidants that rival blueberries. The melting point of cocoa butter is just below body temperature!" },
            { id: 33, title: "The Art of French Croissants", content: "A proper croissant requires 81 layers of butter and dough created through a process called lamination. The butter must be the exact right temperature - too cold and it breaks, too warm and it melts. Each fold doubles the number of layers!" },
            { id: 34, title: "Sourdough: The Ancient Bread Revolution", content: "Sourdough starter is a living ecosystem of wild yeast and bacteria that can be maintained for decades. The fermentation process breaks down gluten and creates lactic acid, giving sourdough its distinctive tang and making it easier to digest." },
            { id: 35, title: "The Chemistry of Caramelization", content: "Caramelization occurs when sugars are heated above 320°F, breaking down into hundreds of different compounds that create complex flavors and aromas. This process is responsible for the golden color and rich taste of everything from crème brûlée to roasted coffee." },
            { id: 36, title: "Umami: The Fifth Taste", content: "Umami, discovered by Japanese scientist Kikunae Ikeda, is triggered by glutamates found in foods like tomatoes, cheese, mushrooms, and seaweed. It's the savory taste that makes foods like parmesan and soy sauce so addictive and satisfying." },
            { id: 37, title: "The Perfect Cup of Coffee", content: "Coffee extraction is a delicate balance of grind size, water temperature (195-205°F), and brewing time. The golden ratio is 1:15-1:17 coffee to water. Over-extraction leads to bitterness, while under-extraction results in sourness." },
            { id: 38, title: "Ice Cream Science: Crystals and Creaminess", content: "Perfect ice cream requires controlling ice crystal formation through proper churning and stabilizers. The ideal serving temperature is 6-10°F. Liquid nitrogen ice cream freezes so quickly that it creates incredibly small ice crystals for ultimate smoothness." },
            { id: 39, title: "The Magic of Fermentation", content: "Fermentation transforms simple ingredients into complex foods like kimchi, wine, and cheese. Beneficial bacteria and yeasts break down sugars and proteins, creating new flavors, preserving food, and adding probiotics that benefit gut health." },
            { id: 40, title: "Molecular Gastronomy: Food Meets Science", content: "Molecular gastronomy uses scientific principles to create surprising textures and presentations. Spherification creates caviar-like pearls, liquid nitrogen flash-freezes ingredients, and hydrocolloids create foams and gels that challenge our expectations." },

            // Space & Astronomy (41-50)
            { id: 41, title: "Black Holes: The Universe's Ultimate Mystery", content: "Black holes are regions where gravity is so strong that nothing, not even light, can escape. They warp space-time itself, and at their center lies a singularity where physics as we know it breaks down. Some are billions of times more massive than our Sun!" },
            { id: 42, title: "The International Space Station: Humanity's Orbital Home", content: "The ISS orbits Earth every 90 minutes at 17,500 mph, experiencing 16 sunrises and sunsets daily. Astronauts conduct hundreds of experiments in microgravity, from growing crystals to studying how flames behave without gravity." },
            { id: 43, title: "Mars: The Red Planet's Secrets", content: "Mars has the largest volcano in the solar system (Olympus Mons) and a canyon system that dwarfs the Grand Canyon. Evidence suggests it once had flowing water and a thicker atmosphere. Today, NASA's rovers search for signs of ancient microbial life." },
            { id: 44, title: "The James Webb Space Telescope: Eyes on the Cosmos", content: "JWST can see the first galaxies that formed after the Big Bang, observe exoplanet atmospheres, and peer through cosmic dust clouds. Its mirrors are made of gold-plated beryllium and must be kept at -370°F to detect infrared light from distant objects." },
            { id: 45, title: "Neutron Stars: The Densest Objects in the Universe", content: "Neutron stars are so dense that a teaspoon would weigh 6 billion tons on Earth. They spin up to 700 times per second and have magnetic fields trillions of times stronger than Earth's. Some emit beams of radiation like cosmic lighthouses." },
            { id: 46, title: "The Search for Exoplanets", content: "Over 5,000 exoplanets have been discovered, including some in the 'Goldilocks zone' where liquid water could exist. The Kepler Space Telescope found planets made of diamond, worlds with glass rain, and systems with multiple suns like Tatooine." },
            { id: 47, title: "Saturn's Rings: A Cosmic Light Show", content: "Saturn's rings are made of billions of ice and rock particles ranging from tiny grains to house-sized chunks. They're incredibly thin - if Saturn were the size of a basketball, the rings would be thinner than paper. Some rings have 'spokes' that rotate with the planet." },
            { id: 48, title: "The Voyager Missions: Humanity's Farthest Reach", content: "Voyager 1 and 2, launched in 1977, have traveled beyond our solar system into interstellar space. They carry golden records with sounds and images from Earth, including music by Bach and Chuck Berry, as messages to potential alien civilizations." },
            { id: 49, title: "Jupiter: The Solar System's Guardian", content: "Jupiter acts as a cosmic vacuum cleaner, protecting inner planets from asteroids and comets with its massive gravity. It has at least 79 moons, including Europa which may harbor an ocean beneath its icy surface that could contain more water than all Earth's oceans." },
            { id: 50, title: "The Big Bang: How Everything Began", content: "The universe began 13.8 billion years ago from a singularity smaller than an atom. In the first second, it expanded faster than light, cooled enough for protons and neutrons to form, and set the stage for the creation of all matter and energy we see today." },

            // Technology & Gaming (51-60)
            { id: 51, title: "The Evolution of Video Game Graphics", content: "From Pong's simple pixels to photorealistic ray tracing, video game graphics have evolved exponentially. Modern GPUs can render millions of polygons per second, simulate realistic lighting, and create virtual worlds indistinguishable from reality." },
            { id: 52, title: "How WiFi Works: Invisible Internet Magic", content: "WiFi uses radio waves at 2.4GHz and 5GHz frequencies to transmit data. Your router converts internet signals into radio waves, which your device's antenna receives and converts back to data. The latest WiFi 6 can reach speeds of 9.6 Gbps!" },
            { id: 53, title: "The History of the Computer Mouse", content: "The computer mouse was invented in 1964 by Douglas Engelbart and was originally called an 'X-Y Position Indicator for a Display System.' The first mouse was made of wood and had only one button. Today's optical mice use LED lights and sensors to track movement." },
            { id: 54, title: "Quantum Computing: The Future of Processing", content: "Quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously, potentially solving certain problems exponentially faster than classical computers. They could revolutionize cryptography, drug discovery, and artificial intelligence." },
            { id: 55, title: "The Internet: How 4 Billion People Connect", content: "The internet is a network of networks, connecting billions of devices worldwide through fiber optic cables, satellites, and wireless signals. Data travels at nearly the speed of light, and a single fiber optic cable can carry 10 terabits per second." },
            { id: 56, title: "Smartphone Cameras: Computational Photography", content: "Modern smartphone cameras use AI and computational photography to create stunning images. Features like Night Mode, Portrait Mode, and HDR combine multiple exposures and use machine learning to enhance photos beyond what the hardware alone could achieve." },
            { id: 57, title: "The Rise of Electric Vehicles", content: "Electric vehicles use lithium-ion batteries and electric motors that are 90% efficient compared to 30% for gasoline engines. Regenerative braking captures energy when slowing down, and the latest EVs can travel over 400 miles on a single charge." },
            { id: 58, title: "3D Printing: Manufacturing Revolution", content: "3D printing builds objects layer by layer from digital designs, using materials from plastic to metal to living cells. It's revolutionizing manufacturing by enabling rapid prototyping, custom medical implants, and on-demand production of complex parts." },
            { id: 59, title: "Virtual Reality: Stepping Into Digital Worlds", content: "VR headsets track head movement and display stereoscopic images to create immersive experiences. Modern VR can achieve 90+ FPS with sub-20ms latency to prevent motion sickness. Applications range from gaming to medical training to virtual tourism." },
            { id: 60, title: "The Story of Open Source Software", content: "Open source software, where code is freely available and modifiable, powers most of the internet. Linux runs on billions of devices, from smartphones to supercomputers. The collaborative development model has created some of the world's most important software." },

            // Nature & Environment (61-70)
            { id: 61, title: "The Amazon Rainforest: Earth's Lungs", content: "The Amazon produces 20% of the world's oxygen and contains 10% of all known species. It has its own weather system, creating rain clouds that travel across South America. One tree can be home to over 400 species of insects!" },
            { id: 62, title: "Coral Reefs: Underwater Cities", content: "Coral reefs support 25% of all marine species despite covering less than 1% of the ocean floor. They're built by tiny animals called polyps that have a symbiotic relationship with algae. The Great Barrier Reef can be seen from space!" },
            { id: 63, title: "The Northern Lights: Nature's Light Show", content: "Aurora borealis occurs when charged particles from the sun interact with Earth's magnetic field and atmosphere. The colors depend on altitude and gas type - oxygen creates green and red, while nitrogen produces blue and purple. They can reach 400 miles high!" },
            { id: 64, title: "Mushrooms: The Internet of the Forest", content: "Fungi create vast underground networks called mycorrhizae that connect trees and plants, sharing nutrients and information. Some fungal networks span thousands of acres. The largest living organism on Earth is a honey fungus in Oregon covering 2,400 acres!" },
            { id: 65, title: "Bees: The Tiny Heroes of Agriculture", content: "Bees pollinate one-third of everything we eat, from apples to almonds. A single bee visits 2,000 flowers per day and flies up to 6 miles from the hive. They communicate through the 'waggle dance' to tell other bees where to find the best flowers." },
            { id: 66, title: "The Deep Ocean: Earth's Final Frontier", content: "We've explored less than 5% of our oceans. The deep sea contains bioluminescent creatures, underwater mountains taller than Everest, and ecosystems that thrive around volcanic vents without sunlight. The pressure at the deepest point could crush a human instantly." },
            { id: 67, title: "Redwood Trees: The Giants of the Forest", content: "Coast redwoods are the tallest trees on Earth, reaching over 380 feet high and living for over 2,000 years. They can absorb fog through their needles and create their own rain. Their bark is fire-resistant and can be up to 12 inches thick." },
            { id: 68, title: "The Water Cycle: Nature's Recycling System", content: "Every drop of water on Earth has been recycled countless times through evaporation, condensation, and precipitation. The water you drink today might have been in a dinosaur, a cloud, or an ocean millions of years ago. It's the ultimate renewable resource!" },
            { id: 69, title: "Photosynthesis: How Plants Eat Sunlight", content: "Plants convert sunlight, carbon dioxide, and water into glucose and oxygen through photosynthesis. This process captures about 100 terawatts of solar energy annually - six times more than human civilization consumes. It's the foundation of almost all life on Earth." },
            { id: 70, title: "Migration: Nature's Greatest Journeys", content: "Arctic terns migrate 44,000 miles annually from Arctic to Antarctic and back. Monarch butterflies travel 3,000 miles across generations, with great-great-grandchildren returning to the same trees their ancestors left. Salmon return to their exact birthplace to spawn." },

            // Arts & Culture (71-80)
            { id: 71, title: "The Science of Music: Why Songs Make Us Feel", content: "Music activates multiple brain regions simultaneously, releasing dopamine and triggering emotional responses. Certain chord progressions create tension and resolution, while rhythm synchronizes with our heartbeat. Music therapy can help treat depression and improve memory." },
            { id: 72, title: "The Golden Ratio in Art and Nature", content: "The golden ratio (1.618) appears in art, architecture, and nature from the Parthenon to sunflower spirals. Artists like Leonardo da Vinci used it to create pleasing proportions. It's found in flower petals, nautilus shells, and even human facial features." },
            { id: 73, title: "The History of Animation: From Drawings to Digital", content: "Animation began with simple flipbooks and evolved through hand-drawn cells to computer graphics. Disney's Snow White used 200,000 drawings, while modern Pixar films use millions of computer calculations per frame. Each second of animation can take weeks to create." },
            { id: 74, title: "Street Art: From Vandalism to Gallery Walls", content: "Street art has evolved from simple graffiti tags to complex murals that transform urban landscapes. Artists like Banksy use stencils and social commentary to create thought-provoking works. Many cities now commission street artists to beautify neighborhoods." },
            { id: 75, title: "The Psychology of Color", content: "Colors affect our emotions and behavior in measurable ways. Red increases heart rate and appetite (why it's used in restaurants), blue promotes calm and productivity, and green reduces eye strain. Marketing teams spend millions studying color psychology." },
            { id: 76, title: "Dance: The Universal Language", content: "Every culture has developed forms of dance, from ballet's precise technique to hip-hop's street origins. Dancing releases endorphins, improves coordination, and builds social bonds. Professional dancers train as intensively as Olympic athletes." },
            { id: 77, title: "The Magic of Theater: Live Performance", content: "Theater creates unique shared experiences between performers and audiences. Each performance is different, with actors feeding off audience energy. The 'fourth wall' concept allows audiences to observe intimate moments while remaining invisible to the characters." },
            { id: 78, title: "Photography: Capturing Light and Time", content: "Photography literally means 'drawing with light.' From camera obscura to digital sensors, it's about controlling light exposure. The decisive moment, composition rules like the rule of thirds, and post-processing all contribute to creating compelling images." },
            { id: 79, title: "The Art of Storytelling", content: "Humans are wired for stories - they help us make sense of the world and connect with others. Good stories follow patterns like the hero's journey, create emotional arcs, and use techniques like foreshadowing and metaphor to engage audiences." },
            { id: 80, title: "Fashion: Wearable Art and Cultural Expression", content: "Fashion reflects cultural values, social status, and individual identity. Designers use color, texture, and silhouette to create emotional responses. Sustainable fashion is growing as consumers become aware of the industry's environmental impact." },

            // Fun & Quirky (81-100)
            { id: 81, title: "Why We Yawn: The Contagious Mystery", content: "Yawning might help cool the brain, increase alertness, or synchronize group behavior. It's contagious even across species - dogs yawn when their owners do. Psychopaths are less likely to catch yawns, suggesting it's linked to empathy." },
            { id: 82, title: "The Science of Laughter", content: "Laughter triggers the release of endorphins, boosts immune function, and burns calories. We laugh 30 times more when with others than alone. Babies laugh before they can speak, and laughter is universal across all human cultures." },
            { id: 83, title: "Why Bubble Wrap is So Satisfying to Pop", content: "Popping bubble wrap releases tension and provides sensory satisfaction. The sound triggers a mild stress response followed by relief. Studies show it can reduce anxiety and improve focus. The inventor originally intended it as wallpaper!" },
            { id: 84, title: "The Mystery of Déjà Vu", content: "Déjà vu affects 60-70% of people and might be caused by a delay in neural processing, making the present feel like a memory. It's more common when tired or stressed. Some theories suggest it's a 'glitch' in how the brain processes time and memory." },
            { id: 85, title: "Why Songs Get Stuck in Your Head", content: "Earworms (stuck songs) happen when your brain's auditory cortex gets stuck in a loop. Songs with simple, repetitive melodies are most likely to stick. Chewing gum or listening to the full song can help break the cycle." },
            { id: 86, title: "The Science of Procrastination", content: "Procrastination is a battle between the limbic system (seeking immediate pleasure) and the prefrontal cortex (planning for the future). Breaking tasks into smaller pieces and removing distractions can help overcome the procrastination habit." },
            { id: 87, title: "Why We Love Cute Things", content: "The 'baby schema' (large eyes, round face, small nose) triggers nurturing instincts across species. This cuteness response helped our ancestors care for helpless infants. It's why we find puppies, kittens, and cartoon characters irresistible." },
            { id: 88, title: "The Psychology of Superstitions", content: "Superstitions give us a sense of control over uncertain situations. Athletes' rituals, lucky charms, and avoiding black cats all serve to reduce anxiety. Even pigeons can develop superstitious behaviors in laboratory settings." },
            { id: 89, title: "Why Time Flies When You're Having Fun", content: "When engaged in enjoyable activities, we pay less attention to time passing. Boredom makes us hyper-aware of time. Age also affects time perception - years feel shorter as we get older because each year becomes a smaller fraction of our total experience." },
            { id: 90, title: "The Art of Napping", content: "The perfect nap is 10-20 minutes long, avoiding deep sleep phases that cause grogginess. Napping can improve alertness, creativity, and memory consolidation. Some cultures embrace siesta time, recognizing the natural afternoon energy dip." },
            { id: 91, title: "Why We Get Goosebumps", content: "Goosebumps are an evolutionary leftover from when we had more body hair. They're triggered by cold, fear, or emotional experiences like music. The scientific term is 'cutis anserina' (goose skin), and they're controlled by the sympathetic nervous system." },
            { id: 92, title: "The Science of Hiccups", content: "Hiccups are caused by involuntary diaphragm spasms. They might be an evolutionary remnant from when our ancestors had both gills and lungs. Most remedies work by interrupting the nerve signals or resetting the diaphragm rhythm." },
            { id: 93, title: "Why We Dream", content: "Dreams might help process emotions, consolidate memories, and solve problems. REM sleep, when most vivid dreams occur, is crucial for learning and mental health. We forget 95% of our dreams within minutes of waking up." },
            { id: 94, title: "The Power of Placebo", content: "Placebo effects can cause real physiological changes, from pain relief to improved immune function. The brain's expectation of healing can trigger actual healing responses. Even when people know they're taking a placebo, it can still work!" },
            { id: 95, title: "Why We Love Horror Movies", content: "Horror movies trigger controlled fear in a safe environment, releasing adrenaline and endorphins. Some people are 'sensation seekers' who enjoy intense experiences. Watching with others creates social bonding through shared emotional experiences." },
            { id: 96, title: "The Science of Happiness", content: "Happiness is influenced by genetics (50%), circumstances (10%), and intentional activities (40%). Gratitude, exercise, social connections, and acts of kindness all boost happiness. Money improves happiness up to about $75,000 annually, then levels off." },
            { id: 97, title: "Why We Collect Things", content: "Collecting satisfies needs for completion, control, and social connection. It can reduce anxiety and provide a sense of accomplishment. From stamps to sneakers, collections create order in our lives and connect us with like-minded communities." },
            { id: 98, title: "The Mystery of Left-Handedness", content: "About 10% of people are left-handed, and it's more common in men. Left-handedness might be linked to creativity and different brain organization. In a right-handed world, lefties have adapted to be more flexible and innovative." },
            { id: 99, title: "Why We Love Puzzles", content: "Puzzles activate the brain's reward system, releasing dopamine when we solve them. They improve problem-solving skills, memory, and can reduce stress. The satisfaction of fitting pieces together taps into our innate desire to create order from chaos." },
            { id: 100, title: "The Science of Serendipity", content: "Serendipity - finding something valuable while looking for something else - has led to discoveries like penicillin, Post-it notes, and X-rays. Cultivating curiosity, staying open to unexpected connections, and embracing 'happy accidents' can increase serendipitous discoveries." }
        ];

        return docs;
    }

    /**
     * Search using FastPlaid indexing
     */
    async searchWithFastPlaid(queryResult, documents, topK) {
        const t_js_start = performance.now();

        // Determine which FastPlaid instance to use based on quantization mode
        const useQuantization = window.useQuantization || false;
        const fastPlaidInstance = useQuantization ? window.fastPlaidQuantized : window.fastPlaid;
        const modeName = useQuantization ? 'FastPlaid Quantized (4-bit)' : 'FastPlaid Uncompressed';

        // Get the FastPlaid instance from the global scope
        if (!fastPlaidInstance) {
            console.warn(`⚠️ ${modeName} WASM not available, falling back to direct MaxSim`);
            return await this.searchWithDirectMaxSim(queryResult, documents, topK);
        }

        try {
            // Prepare query embeddings for FastPlaid
            const t_prep = performance.now();
            const queryShape = new Uint32Array([1, queryResult.numTokens, queryResult.tokenDim || this.embeddingDim]);
            const t_prep_done = performance.now();

            // Call FastPlaid WASM search (quantized or uncompressed)
            const t_wasm_call = performance.now();
            console.log(`🔍 Using ${modeName} for search`);
            const searchResults = fastPlaidInstance.search(
                queryResult.embeddings,
                queryShape,
                topK,
                10 // n_ivf_probe
            );
            const t_wasm_done = performance.now();

            // Parse JSON string response from WASM
            const t_parse = performance.now();
            let parsedResults;
            if (typeof searchResults === 'string') {
                parsedResults = JSON.parse(searchResults);
            } else {
                parsedResults = searchResults;
            }
            const t_parse_done = performance.now();

            // Convert FastPlaid results to our format
            const t_convert = performance.now();
            if (parsedResults && parsedResults.length > 0) {
                const result = parsedResults[0]; // First query result
                const results = [];

                for (let i = 0; i < Math.min(result.passage_ids.length, topK); i++) {
                    const docId = result.passage_ids[i];
                    const score = result.scores[i];

                    // Find the document by ID
                    const doc = documents.find(d => d.id === docId);
                    if (doc) {
                        results.push({
                            id: doc.id,
                            title: doc.title,
                            content: doc.content || '',
                            score: score,
                            isReal: queryResult.isReal && doc.isReal,
                            numTokens: doc.numTokens,
                            encodingTime: doc.encodingTime
                        });
                    }
                }
                const t_convert_done = performance.now();

                // JavaScript profiling
                console.log(`⏱️ JavaScript (${modeName}) Profiling:`);
                console.log(`   JS Preparation:  ${(t_prep_done - t_prep).toFixed(2)}ms`);
                console.log(`   WASM Call:       ${(t_wasm_done - t_wasm_call).toFixed(2)}ms (includes WASM execution)`);
                console.log(`   JSON Parse:      ${(t_parse_done - t_parse).toFixed(2)}ms`);
                console.log(`   Result Convert:  ${(t_convert_done - t_convert).toFixed(2)}ms`);
                console.log(`   ═══════════════`);
                console.log(`   Total JS:        ${(t_convert_done - t_js_start).toFixed(2)}ms`);

                return results;
            } else {
                console.warn('⚠️ FastPlaid returned no results, falling back to direct MaxSim');
                const fallbackResults = await this.searchWithDirectMaxSim(queryResult, documents, topK);
                // Mark results as fallback
                fallbackResults._usedFallback = true;
                return fallbackResults;
            }

        } catch (error) {
            console.error('❌ FastPlaid search failed:', error);
            console.log('🔄 Falling back to direct MaxSim...');
            const fallbackResults = await this.searchWithDirectMaxSim(queryResult, documents, topK);
            // Mark results as fallback
            fallbackResults._usedFallback = true;
            return fallbackResults;
        }
    }

    /**
     * Search using direct MaxSim computation between embeddings
     */
    async searchWithDirectMaxSim(queryResult, documents, topK) {
        console.log('🎯 Using direct MaxSim computation...');

        const scores = [];
        for (const doc of documents) {
            const score = this.calculateMaxSimScore(
                queryResult.embeddings,
                doc.embeddings,
                queryResult.numTokens,
                doc.numTokens
            );

            scores.push({
                id: doc.id,
                title: doc.title,
                content: doc.content || '', // Include content
                score: score,
                isReal: queryResult.isReal && doc.isReal,
                numTokens: doc.numTokens,
                encodingTime: doc.encodingTime
            });
        }

        // Sort by score and return top K
        scores.sort((a, b) => b.score - a.score);
        return scores.slice(0, topK);
    }

    /**
     * Calculate ColBERT MaxSim scoring between query and document
     */
    calculateMaxSimScore(queryEmbeddings, docEmbeddings, queryTokens, docTokens) {
        let totalScore = 0;
        const queryDim = this.embeddingDim;

        // For each query token, find max similarity with any document token
        for (let q = 0; q < queryTokens; q++) {
            let maxSim = -Infinity;

            for (let d = 0; d < docTokens; d++) {
                let dotProduct = 0;
                for (let i = 0; i < queryDim; i++) {
                    dotProduct += queryEmbeddings[q * queryDim + i] * docEmbeddings[d * queryDim + i];
                }
                maxSim = Math.max(maxSim, dotProduct);
            }

            totalScore += maxSim;
        }

        return totalScore; // SUM of MaxSim scores (official ColBERT implementation)
    }

    /**
     * Perform end-to-end search: encode query, search index, return results
     * @param {string} query - Search query
     * @param {Array} documents - Document index
     * @param {number} topK - Number of results to return
     * @param {boolean} useFastPlaid - Whether to use FastPlaid or direct maxsim
     */
    async searchDocuments(query, documents, topK = 5, useFastPlaid = true) {
        console.log(`🔍 Searching for: "${query}"`);
        console.log(`🔧 Search method: ${useFastPlaid ? 'FastPlaid' : 'Direct MaxSim'}`);

        const totalStartTime = performance.now();

        // 1. Encode the query (is_query: true)
        const queryStartTime = performance.now();
        const queryResult = await this.encodeText(query, true);
        const queryEncodingTime = performance.now() - queryStartTime;
        this.timings.queryEncoding.push(queryEncodingTime);

        // 2. Search phase
        const searchStartTime = performance.now();
        let results;

        if (useFastPlaid) {
            results = await this.searchWithFastPlaid(queryResult, documents, topK);
        } else {
            results = await this.searchWithDirectMaxSim(queryResult, documents, topK);
        }

        const searchTime = performance.now() - searchStartTime;
        const totalTime = performance.now() - totalStartTime;

        this.timings.searchTime.push(searchTime);
        this.timings.totalSearchTime.push(totalTime);

        // Determine actual method used (accounting for fallbacks)
        let actualMethod;
        const useQuantization = window.useQuantization || false;
        if (useFastPlaid && results._usedFallback) {
            actualMethod = 'Direct MaxSim (FastPlaid fallback)';
        } else if (useFastPlaid && useQuantization) {
            actualMethod = 'FastPlaid Quantized (4-bit)';
        } else if (useFastPlaid) {
            actualMethod = 'FastPlaid Uncompressed (f32)';
        } else {
            actualMethod = 'Direct MaxSim';
        }

        // Add timing information to results
        const realResults = results.filter(r => r.isReal).length;
        console.log(`✅ Found ${results.length} results (${realResults} using real embeddings) via ${actualMethod}`);
        console.log(`⏱️ Query encoding: ${queryEncodingTime.toFixed(2)}ms`);
        console.log(`⏱️ Search time: ${searchTime.toFixed(2)}ms`);
        console.log(`⏱️ Total time: ${totalTime.toFixed(2)}ms`);

        // Clean up the fallback marker before returning
        if (results._usedFallback) {
            delete results._usedFallback;
        }

        return {
            results: results,
            timings: {
                queryEncoding: queryEncodingTime,
                searchTime: searchTime,
                totalTime: totalTime,
                method: actualMethod
            }
        };
    }

    /**
     * Get model status information
     */
    getModelStatus() {
        return {
            loaded: this.modelLoaded,
            modelRepo: this.modelRepo,
            simulationMode: this.simulationMode || false,
            hasRealModel: this.model !== null,
            embeddingDim: this.embeddingDim,
            pylateLoaded: ColBERT !== null
        };
    }

    /**
     * Get performance statistics
     */
    getPerformanceStats() {
        const calculateStats = (times) => {
            if (times.length === 0) return { avg: 0, min: 0, max: 0, count: 0 };
            const avg = times.reduce((a, b) => a + b, 0) / times.length;
            const min = Math.min(...times);
            const max = Math.max(...times);
            return { avg: avg.toFixed(2), min: min.toFixed(2), max: max.toFixed(2), count: times.length };
        };

        return {
            modelEncoding: calculateStats(this.timings.modelEncoding),
            queryEncoding: calculateStats(this.timings.queryEncoding),
            searchTime: calculateStats(this.timings.searchTime),
            totalSearchTime: calculateStats(this.timings.totalSearchTime),
            indexTime: calculateStats(this.timings.indexTime),
            memory: this.getIndexMemoryStats()
        };
    }

    /**
     * Get index memory statistics
     */
    getIndexMemoryStats() {
        const formatBytes = (bytes) => {
            if (bytes === 0) return '0 B';
            const mb = bytes / 1024 / 1024;
            return `${mb.toFixed(2)} MB`;
        };

        const calculateSavings = () => {
            const fastPlaidTotal = this.indexMemory.fastPlaid.totalBytes;
            const directMaxSimTotal = this.indexMemory.directMaxSim.totalBytes;

            if (fastPlaidTotal === 0 || directMaxSimTotal === 0) return null;

            const savings = directMaxSimTotal - fastPlaidTotal;
            const savingsPercent = (savings / directMaxSimTotal) * 100;

            return {
                absolute: formatBytes(savings),
                percent: savingsPercent.toFixed(1)
            };
        };

        return {
            fastPlaid: {
                total: formatBytes(this.indexMemory.fastPlaid.totalBytes),
                totalBytes: this.indexMemory.fastPlaid.totalBytes,
                embeddings: formatBytes(this.indexMemory.fastPlaid.embeddingsBytes),
                metadata: formatBytes(this.indexMemory.fastPlaid.metadataBytes),
                documentCount: this.indexMemory.fastPlaid.documentCount,
                embeddingDim: this.indexMemory.fastPlaid.embeddingDim
            },
            directMaxSim: {
                total: formatBytes(this.indexMemory.directMaxSim.totalBytes),
                totalBytes: this.indexMemory.directMaxSim.totalBytes,
                embeddings: formatBytes(this.indexMemory.directMaxSim.embeddingsBytes),
                metadata: formatBytes(this.indexMemory.directMaxSim.metadataBytes),
                documentCount: this.indexMemory.directMaxSim.documentCount,
                embeddingDim: this.indexMemory.directMaxSim.embeddingDim
            },
            savings: calculateSavings()
        };
    }

    /**
     * Reset performance statistics
     */
    resetPerformanceStats() {
        this.timings = {
            modelEncoding: [],
            queryEncoding: [],
            searchTime: [],
            totalSearchTime: [],
            indexTime: []
        };
    }

    /**
     * Test pylate-rs basic functionality
     */
    async testPylateRs() {
        console.log('🧪 Testing pylate-rs basic functionality...');

        if (!ColBERT) {
            throw new Error('ColBERT class not available');
        }

        console.log('✅ ColBERT class is available');
        console.log('🔍 ColBERT constructor:', ColBERT.toString());

        return true;
    }

    /**
     * Try to load specifically the mxbai model with detailed debugging
     */
    async tryMxbaiModel() {
        console.log('🚀 Trying specifically mxbai-edge-colbert-v0-17m...');

        try {
            await this.testPylateRs();

            // Check what files are actually available for mxbai model
            const modelRepo = 'mixedbread-ai/mxbai-edge-colbert-v0-17m';
            const basePath = `https://huggingface.co/${modelRepo}/resolve/main`;

            console.log('🔍 Checking available files for mxbai model...');

            // Check each required file individually
            const fileStatus = {};
            for (const file of this.requiredFiles) {
                try {
                    const response = await fetch(`${basePath}/${file}`, { method: 'HEAD' });
                    fileStatus[file] = response.ok ? '✅' : `❌ ${response.status}`;
                } catch (e) {
                    fileStatus[file] = `❌ ${e.message}`;
                }
            }

            console.log('📋 mxbai file availability:', fileStatus);

            // Count available files
            const availableFiles = Object.entries(fileStatus).filter(([_, status]) => status === '✅');
            console.log(`📊 Available files: ${availableFiles.length}/${this.requiredFiles.length}`);

            if (availableFiles.length < this.requiredFiles.length) {
                throw new Error(`mxbai-edge-colbert-v0-17m missing required files. Available: ${availableFiles.length}/${this.requiredFiles.length}`);
            }

            // Load mxbai model (no fallback)
            await this.loadSingleModel(modelRepo);
            this.modelRepo = modelRepo;

            this.simulationMode = false;
            console.log(`🎉 Model loaded successfully: ${this.modelRepo}`);
            console.log(`📊 Model config dimension: 48 (2_Dense output)`);
            console.log(`📊 Actual pylate-rs dimension: 512 (1_Dense output only)`);
            console.log(`⚠️ Note: pylate-rs doesn't support 2_Dense layer`);
            return true;
        } catch (error) {
            console.error('❌ Model loading failed:', error);
            return false;
        }
    }

    /**
     * Force retry loading the real model (for debugging)
     */
    async forceRealModel() {
        console.log('🔄 Force retrying real model loading...');

        // First test if pylate-rs is working
        try {
            await this.testPylateRs();
        } catch (error) {
            console.error('❌ pylate-rs test failed:', error);
            return false;
        }

        this.simulationMode = false;
        this.model = null;

        try {
            await this.loadModelFromHuggingFace();
            console.log('✅ Force retry successful!');
            return true;
        } catch (error) {
            console.error('❌ Force retry failed:', error);
            this.simulationMode = true;
            return false;
        }
    }
}

// Export for use in the demo
window.MxbaiEdgeColbertIntegration = MxbaiEdgeColbertIntegration;