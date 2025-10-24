/**
 * Loader for precomputed binary index created by build_precomputed_index.py
 * Handles both Direct MaxSim (float32 embeddings) and FastPlaid (quantized) indices
 */

export class PrecomputedIndexLoader {
    constructor(basePath = './data') {
        this.basePath = basePath;
        this.papers = null;
        this.embeddings = null;
        this.metadata = null;
    }

    /**
     * Load papers metadata (titles, abstracts, etc.)
     */
    async loadPapersMetadata() {
        console.log('📄 Loading papers metadata...');
        const response = await fetch(`${this.basePath}/papers_metadata.json`);
        this.papers = await response.json();
        console.log(`✅ Loaded ${this.papers.length} papers`);
        return this.papers;
    }

    /**
     * Load binary embeddings for Direct MaxSim
     * Format: [num_papers:u32][embedding_dim:u32][paper1_num_tokens:u32][paper1_embeddings:f32[]]...
     */
    async loadDirectMaxSimEmbeddings() {
        console.log('📦 Loading Direct MaxSim embeddings (binary)...');
        const startTime = performance.now();

        const [binResponse, metaResponse] = await Promise.all([
            fetch(`${this.basePath}/embeddings.bin`),
            fetch(`${this.basePath}/embeddings_meta.json`)
        ]);

        const binData = await binResponse.arrayBuffer();
        this.metadata = await metaResponse.json();

        const view = new DataView(binData);
        let offset = 0;

        // Read header
        const numPapers = view.getUint32(offset, true);
        offset += 4;
        const embeddingDim = view.getUint32(offset, true);
        offset += 4;

        console.log(`   Papers: ${numPapers}, Embedding dim: ${embeddingDim}`);

        // Read embeddings for each paper
        const embeddings = [];
        for (let i = 0; i < numPapers; i++) {
            const numTokens = view.getUint32(offset, true);
            offset += 4;

            const embedding = new Float32Array(numTokens * embeddingDim);
            for (let j = 0; j < numTokens * embeddingDim; j++) {
                embedding[j] = view.getFloat32(offset, true);
                offset += 4;
            }

            embeddings.push({
                numTokens,
                embedding
            });
        }

        this.embeddings = embeddings;

        const loadTime = (performance.now() - startTime).toFixed(2);
        const sizeMB = (binData.byteLength / 1024 / 1024).toFixed(2);
        console.log(`✅ Loaded embeddings in ${loadTime}ms (${sizeMB} MB)`);

        return {
            embeddings,
            metadata: this.metadata
        };
    }

    /**
     * Load FastPlaid 4-bit quantized index with clustering
     * Format: [total_tokens:u32][embedding_dim:u32][num_papers:u32][num_clusters:u32]
     *         [min_vals:f32[]][max_vals:f32[]][doc_boundaries:u32[]]
     *         [centroids:f32[]][cluster_labels:u16[]][clusters][packed_4bit:u8[]]
     */
    async loadFastPlaid4BitIndex() {
        console.log('🗜️ Loading FastPlaid 4-bit index...');
        const startTime = performance.now();

        const [indexResponse, metaResponse] = await Promise.all([
            fetch(`${this.basePath}/fastplaid_4bit.bin`),
            fetch(`${this.basePath}/fastplaid_meta.json`)
        ]);

        const indexData = await indexResponse.arrayBuffer();
        const indexMeta = await metaResponse.json();

        const view = new DataView(indexData);
        let offset = 0;

        // Read header
        const totalTokens = view.getUint32(offset, true);
        offset += 4;
        const embeddingDim = view.getUint32(offset, true);
        offset += 4;
        const numPapers = view.getUint32(offset, true);
        offset += 4;
        const numClusters = view.getUint32(offset, true);
        offset += 4;

        console.log(`   Total tokens: ${totalTokens.toLocaleString()}`);
        console.log(`   Embedding dim: ${embeddingDim}`);
        console.log(`   Papers: ${numPapers}`);
        console.log(`   Clusters: ${numClusters}`);

        // Read min/max values for dequantization
        const minVals = new Float32Array(embeddingDim);
        const maxVals = new Float32Array(embeddingDim);

        for (let i = 0; i < embeddingDim; i++) {
            minVals[i] = view.getFloat32(offset, true);
            offset += 4;
        }
        for (let i = 0; i < embeddingDim; i++) {
            maxVals[i] = view.getFloat32(offset, true);
            offset += 4;
        }

        // Read document boundaries
        const docBoundaries = new Uint32Array(numPapers + 1);
        for (let i = 0; i < numPapers + 1; i++) {
            docBoundaries[i] = view.getUint32(offset, true);
            offset += 4;
        }

        // Read cluster centroids
        const centroids = new Float32Array(numClusters * embeddingDim);
        for (let i = 0; i < numClusters * embeddingDim; i++) {
            centroids[i] = view.getFloat32(offset, true);
            offset += 4;
        }

        // Read cluster labels
        const clusterLabels = new Uint16Array(numPapers);
        for (let i = 0; i < numPapers; i++) {
            clusterLabels[i] = view.getUint16(offset, true);
            offset += 2;
        }

        // Read cluster → documents mapping
        const clusters = [];
        for (let c = 0; c < numClusters; c++) {
            const clusterSize = view.getUint32(offset, true);
            offset += 4;
            const docIds = [];
            for (let i = 0; i < clusterSize; i++) {
                docIds.push(view.getUint32(offset, true));
                offset += 4;
            }
            clusters.push(docIds);
        }

        // Read packed 4-bit embeddings
        const packed4bit = new Uint8Array(indexData, offset);

        const loadTime = (performance.now() - startTime).toFixed(2);
        const sizeMB = (indexData.byteLength / 1024 / 1024).toFixed(2);
        console.log(`✅ Loaded FastPlaid 4-bit index in ${loadTime}ms (${sizeMB} MB)`);

        return {
            totalTokens,
            embeddingDim,
            numPapers,
            numClusters,
            minVals,
            maxVals,
            docBoundaries,
            centroids,
            clusterLabels,
            clusters,
            packed4bit,
            metadata: indexMeta
        };
    }

    /**
     * Dequantize embeddings from 4-bit to float32
     * Unpacks 2 values per byte and dequantizes from 0-15 range
     */
    dequantize4bit(packed4bit, startToken, numTokens, embeddingDim, minVals, maxVals) {
        const dequantized = new Float32Array(numTokens * embeddingDim);

        // Calculate starting position in packed array
        const startIdx = startToken * embeddingDim;
        const totalValues = numTokens * embeddingDim;

        for (let i = 0; i < totalValues; i++) {
            const globalIdx = startIdx + i;
            const byteIdx = Math.floor(globalIdx / 2);
            const isUpperNibble = (globalIdx % 2) === 1;

            // Unpack 4-bit value
            const byte = packed4bit[byteIdx];
            const quantVal = isUpperNibble ? (byte >> 4) : (byte & 0x0F);

            // Dequantize from 0-15 to original range
            const dim = i % embeddingDim;
            dequantized[i] = (quantVal / 15.0) * (maxVals[dim] - minVals[dim]) + minVals[dim];
        }

        return dequantized;
    }

    /**
     * Get embedding for a specific document from 4-bit quantized index
     */
    getDocumentEmbedding4bit(fastPlaidIndex, docId) {
        const startToken = fastPlaidIndex.docBoundaries[docId];
        const endToken = fastPlaidIndex.docBoundaries[docId + 1];
        const numTokens = endToken - startToken;

        return this.dequantize4bit(
            fastPlaidIndex.packed4bit,
            startToken,
            numTokens,
            fastPlaidIndex.embeddingDim,
            fastPlaidIndex.minVals,
            fastPlaidIndex.maxVals
        );
    }
}
