// Paper Abstracts Dataset Loader for FastPlaid GitHub Pages Demo
// Optimized for loading large collections of paper abstracts efficiently

/**
 * Paper abstract structure
 * @typedef {Object} PaperAbstract
 * @property {number} id - Unique paper ID
 * @property {string} title - Paper title
 * @property {string} abstract - Paper abstract text
 * @property {string[]} authors - List of authors
 * @property {string} venue - Publication venue (conference/journal)
 * @property {number} year - Publication year
 * @property {string[]} keywords - Keywords/topics
 * @property {string} arxiv_id - arXiv ID (if available)
 * @property {string} doi - DOI (if available)
 */

export class PaperAbstractsLoader {
    constructor(mxbaiIntegration) {
        this.mxbaiIntegration = mxbaiIntegration;
        this.papers = [];
        this.embeddings = null;
    }

    /**
     * Load papers from a JSON file
     * Expected format: Array of PaperAbstract objects
     */
    async loadFromJSON(jsonUrl) {
        console.log(`📄 Loading papers from ${jsonUrl}...`);
        const startTime = performance.now();

        const response = await fetch(jsonUrl);
        if (!response.ok) {
            throw new Error(`Failed to load papers: ${response.statusText}`);
        }

        const data = await response.json();
        this.papers = Array.isArray(data) ? data : data.papers || [];

        const loadTime = performance.now() - startTime;
        console.log(`✅ Loaded ${this.papers.length} papers in ${loadTime.toFixed(2)}ms`);

        return this.papers;
    }

    /**
     * Load papers from a compressed JSONL.gz file (for large datasets)
     * Each line is a JSON object representing one paper
     */
    async loadFromJSONLGz(gzUrl) {
        console.log(`📦 Loading compressed papers from ${gzUrl}...`);
        const startTime = performance.now();

        const response = await fetch(gzUrl);
        if (!response.ok) {
            throw new Error(`Failed to load papers: ${response.statusText}`);
        }

        // Decompress using browser's built-in DecompressionStream
        const decompressedStream = response.body
            .pipeThrough(new DecompressionStream('gzip'));

        const reader = decompressedStream.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        const papers = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete lines
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const paper = JSON.parse(line);
                        papers.push(paper);
                    } catch (e) {
                        console.warn('Failed to parse line:', line.substring(0, 100));
                    }
                }
            }
        }

        // Process final line
        if (buffer.trim()) {
            try {
                const paper = JSON.parse(buffer);
                papers.push(paper);
            } catch (e) {
                console.warn('Failed to parse final line');
            }
        }

        this.papers = papers;
        const loadTime = performance.now() - startTime;
        console.log(`✅ Loaded ${this.papers.length} papers in ${loadTime.toFixed(2)}ms`);

        return this.papers;
    }

    /**
     * Load pre-computed embeddings from a binary file
     * Format: [num_papers, num_tokens, embedding_dim] as Int32 header,
     *         followed by Float32 embeddings
     */
    async loadPrecomputedEmbeddings(binUrl) {
        console.log(`🧮 Loading pre-computed embeddings from ${binUrl}...`);
        const startTime = performance.now();

        const response = await fetch(binUrl);
        if (!response.ok) {
            throw new Error(`Failed to load embeddings: ${response.statusText}`);
        }

        const buffer = await response.arrayBuffer();

        // Read header: [num_papers, avg_tokens, embedding_dim]
        const headerView = new Int32Array(buffer, 0, 3);
        const numPapers = headerView[0];
        const avgTokens = headerView[1];
        const embeddingDim = headerView[2];

        console.log(`   Papers: ${numPapers}, Avg Tokens: ${avgTokens}, Dim: ${embeddingDim}`);

        // Read embeddings
        const embeddingsView = new Float32Array(buffer, 12); // Skip 12-byte header

        this.embeddings = {
            numPapers,
            avgTokens,
            embeddingDim,
            data: embeddingsView,
            totalSize: buffer.byteLength,
        };

        const loadTime = performance.now() - startTime;
        console.log(`✅ Loaded embeddings in ${loadTime.toFixed(2)}ms`);
        console.log(`   Total size: ${(buffer.byteLength / 1_000_000).toFixed(2)} MB`);

        return this.embeddings;
    }

    /**
     * Compute embeddings on-the-fly using mxbai-edge-colbert
     * This is slower but doesn't require pre-computed embeddings
     */
    async computeEmbeddings(progressCallback = null) {
        if (!this.mxbaiIntegration) {
            throw new Error('MxbaiIntegration not provided');
        }

        console.log(`🔢 Computing embeddings for ${this.papers.length} papers...`);
        const startTime = performance.now();

        const embeddings = [];
        const batchSize = 10; // Process 10 papers at a time

        for (let i = 0; i < this.papers.length; i += batchSize) {
            const batch = this.papers.slice(i, Math.min(i + batchSize, this.papers.length));

            for (const paper of batch) {
                // Combine title and abstract for embedding
                const text = `${paper.title}. ${paper.abstract}`;

                try {
                    const emb = await this.mxbaiIntegration.encodeText(text, false); // is_query=false for documents
                    embeddings.push({
                        paperId: paper.id,
                        embedding: emb.embeddings,  // Use 'embeddings' property
                        numTokens: emb.numTokens,
                        isReal: emb.isReal,
                    });
                } catch (e) {
                    console.warn(`Failed to encode paper ${paper.id}:`, e);
                    // Add placeholder
                    embeddings.push({
                        paperId: paper.id,
                        embedding: new Float32Array(96 * 70).fill(0),  // Adjusted for actual embedding dimension
                        numTokens: 70,
                        isReal: false,
                    });
                }
            }

            if (progressCallback) {
                const progress = Math.min(i + batchSize, this.papers.length);
                progressCallback(progress, this.papers.length);
            }
        }

        const computeTime = performance.now() - startTime;
        console.log(`✅ Computed embeddings in ${computeTime.toFixed(2)}ms`);
        console.log(`   Avg time per paper: ${(computeTime / this.papers.length).toFixed(2)}ms`);

        this.embeddings = {
            papers: embeddings,
            computeTime,
        };

        return this.embeddings;
    }

    /**
     * Create FastPlaid index from loaded papers
     */
    async createIndex(useFastPlaidQuantized = false) {
        if (!this.papers || this.papers.length === 0) {
            throw new Error('No papers loaded. Call loadFromJSON() or loadFromJSONLGz() first.');
        }

        console.log(`🏗️ Creating FastPlaid index for ${this.papers.length} papers...`);
        const startTime = performance.now();

        // Prepare embeddings data
        let embeddingsFlat;
        let docInfo;

        if (this.embeddings && this.embeddings.data) {
            // Use pre-computed embeddings
            embeddingsFlat = this.embeddings.data;

            // Build doc_info: [id, num_tokens] pairs
            // Use BigInt64Array because Rust uses i64
            const avgTokens = this.embeddings.avgTokens;
            docInfo = new BigInt64Array(this.papers.length * 2);
            for (let i = 0; i < this.papers.length; i++) {
                docInfo[i * 2] = BigInt(this.papers[i].id);
                docInfo[i * 2 + 1] = BigInt(avgTokens); // Assuming uniform token count
            }
        } else if (this.embeddings && this.embeddings.papers) {
            // Use computed embeddings
            const papers = this.embeddings.papers;

            // Flatten embeddings
            const totalFloats = papers.reduce(
                (sum, p) => sum + (p.embedding ? p.embedding.length : 0),
                0
            );
            embeddingsFlat = new Float32Array(totalFloats);
            // Use BigInt64Array because Rust uses i64
            docInfo = new BigInt64Array(papers.length * 2);

            let offset = 0;
            for (let i = 0; i < papers.length; i++) {
                const paper = papers[i];

                // Skip papers without embeddings
                if (!paper.embedding || paper.embedding.length === 0) {
                    console.warn(`⚠️ Paper ${paper.paperId} has no embedding, skipping`);
                    continue;
                }

                embeddingsFlat.set(paper.embedding, offset);

                docInfo[i * 2] = BigInt(paper.paperId);
                docInfo[i * 2 + 1] = BigInt(paper.numTokens);

                offset += paper.embedding.length;
            }
        } else {
            throw new Error('No embeddings available. Call loadPrecomputedEmbeddings() or computeEmbeddings() first.');
        }

        // Load into FastPlaid
        if (useFastPlaidQuantized && window.fastPlaidQuantized) {
            // Use quantized version (8x compression)
            await window.fastPlaidQuantized.load_documents_quantized(
                embeddingsFlat,
                docInfo,
                256 // num_centroids
            );
            console.log('✅ Loaded into FastPlaidQuantized (4-bit compression)');
        } else if (window.fastPlaid) {
            // Use standard version
            await window.fastPlaid.load_documents(embeddingsFlat, docInfo);
            console.log('✅ Loaded into FastPlaid (f32)');
        } else {
            throw new Error('FastPlaid WASM not initialized');
        }

        const indexTime = performance.now() - startTime;
        console.log(`✅ Created index in ${indexTime.toFixed(2)}ms`);

        return {
            numPapers: this.papers.length,
            indexTime,
            quantized: useFastPlaidQuantized,
        };
    }

    /**
     * Search papers by query using FastPlaid index
     */
    async search(query, topK = 10, useDirect = false) {
        if (useDirect) {
            return this.searchDirect(query, topK);
        }
        if (!this.mxbaiIntegration) {
            throw new Error('MxbaiIntegration not provided');
        }

        console.log(`🔍 Searching for: "${query}"`);
        const startTime = performance.now();

        // Encode query
        const queryStart = performance.now();
        const queryEmb = await this.mxbaiIntegration.encodeText(query);
        const queryTime = performance.now() - queryStart;

        // Search using FastPlaid (use quantized or uncompressed based on initialization)
        const searchStart = performance.now();
        const useQuantization = window.useQuantization || false;
        const fastPlaid = useQuantization ? window.fastPlaidQuantized : window.fastPlaid;
        const modeName = useQuantization ? 'FastPlaid Quantized (4-bit)' : 'FastPlaid Uncompressed (f32)';

        if (!fastPlaid) {
            throw new Error('FastPlaid not initialized');
        }

        console.log(`🔍 Using ${modeName} for search`);

        // Use the correct property name - it's 'embeddings' not 'embedding'
        const embeddingsArray = queryEmb.embeddings || queryEmb.embedding;
        const embeddingDim = queryEmb.tokenDim || (embeddingsArray.length / queryEmb.numTokens);

        const resultsJson = await fastPlaid.search(
            embeddingsArray,
            new Uint32Array([1, queryEmb.numTokens, embeddingDim]), // [batch_size, num_tokens, embedding_dim]
            topK
        );

        const results = JSON.parse(resultsJson);
        const searchTime = performance.now() - searchStart;

        // Map results to papers
        const paperResults = results[0].passage_ids.map((paperId, idx) => {
            const paper = this.papers.find(p => p.id === paperId);
            return {
                ...paper,
                score: results[0].scores[idx],
                rank: idx + 1,
            };
        });

        const totalTime = performance.now() - startTime;

        return {
            query,
            results: paperResults,
            timings: {
                queryEncoding: queryTime,
                search: searchTime,
                total: totalTime,
            },
        };
    }

    /**
     * Search papers using direct MaxSim (no index)
     */
    async searchDirect(query, topK = 10) {
        if (!this.mxbaiIntegration) {
            throw new Error('MxbaiIntegration not provided');
        }

        console.log(`🔍 Searching for: "${query}" (Direct MaxSim)`);
        const startTime = performance.now();

        // Encode query
        const queryStart = performance.now();
        const queryEmb = await this.mxbaiIntegration.encodeText(query, true);
        const queryTime = performance.now() - queryStart;

        // Direct MaxSim computation
        const searchStart = performance.now();
        console.log(`📊 Using Direct MaxSim (no index)`);

        const queryEmbeddings = queryEmb.embeddings || queryEmb.embedding;
        const queryNumTokens = queryEmb.numTokens;
        const queryTokenDim = queryEmb.tokenDim || (queryEmbeddings.length / queryNumTokens);

        // Reshape query embeddings to 2D array [num_tokens, dim]
        const queryTokens = [];
        for (let i = 0; i < queryNumTokens; i++) {
            const tokenStart = i * queryTokenDim;
            queryTokens.push(queryEmbeddings.slice(tokenStart, tokenStart + queryTokenDim));
        }

        // Compute MaxSim scores for all documents
        const scores = [];
        const docEmbeddings = this.embeddings.papers;

        for (const docEmb of docEmbeddings) {
            if (!docEmb.embedding || docEmb.embedding.length === 0) {
                continue;
            }

            const docNumTokens = docEmb.numTokens;
            const docTokenDim = docEmb.embedding.length / docNumTokens;

            // Reshape document embeddings to 2D array [num_tokens, dim]
            const docTokens = [];
            for (let i = 0; i < docNumTokens; i++) {
                const tokenStart = i * docTokenDim;
                docTokens.push(docEmb.embedding.slice(tokenStart, tokenStart + docTokenDim));
            }

            // Compute MaxSim: sum of max similarity for each query token
            let maxSimScore = 0;
            for (const qToken of queryTokens) {
                let maxSim = -Infinity;
                for (const dToken of docTokens) {
                    // Cosine similarity (dot product for normalized vectors)
                    let dotProduct = 0;
                    for (let i = 0; i < queryTokenDim; i++) {
                        dotProduct += qToken[i] * dToken[i];
                    }
                    maxSim = Math.max(maxSim, dotProduct);
                }
                maxSimScore += maxSim;
            }

            scores.push({
                paperId: docEmb.paperId,
                score: maxSimScore
            });
        }

        // Sort by score and get top-k
        scores.sort((a, b) => b.score - a.score);
        const topScores = scores.slice(0, topK);

        const searchTime = performance.now() - searchStart;

        // Map to paper results
        const paperResults = topScores.map((item, idx) => {
            const paper = this.papers.find(p => p.id === item.paperId);
            return {
                ...paper,
                score: item.score,
                rank: idx + 1
            };
        });

        const totalTime = performance.now() - startTime;

        return {
            query,
            results: paperResults,
            timings: {
                queryEncoding: queryTime,
                search: searchTime,
                total: totalTime,
            },
        };
    }

    /**
     * Get statistics about loaded data
     */
    getStatistics() {
        if (!this.papers || this.papers.length === 0) {
            return { loaded: false };
        }

        const avgAbstractLength = this.papers.reduce(
            (sum, p) => sum + (p.abstract?.length || 0),
            0
        ) / this.papers.length;

        const yearCounts = {};
        const venueCounts = {};

        for (const paper of this.papers) {
            yearCounts[paper.year] = (yearCounts[paper.year] || 0) + 1;
            venueCounts[paper.venue] = (venueCounts[paper.venue] || 0) + 1;
        }

        let embeddingStats = null;
        if (this.embeddings) {
            if (this.embeddings.data) {
                embeddingStats = {
                    type: 'precomputed',
                    totalSize: this.embeddings.totalSize,
                    totalSizeMB: (this.embeddings.totalSize / 1_000_000).toFixed(2),
                    embeddingDim: this.embeddings.embeddingDim,
                    avgTokens: this.embeddings.avgTokens,
                };
            } else if (this.embeddings.papers) {
                const totalTokens = this.embeddings.papers.reduce(
                    (sum, p) => sum + p.numTokens,
                    0
                );
                embeddingStats = {
                    type: 'computed',
                    computeTime: this.embeddings.computeTime,
                    avgTokens: (totalTokens / this.embeddings.papers.length).toFixed(1),
                };
            }
        }

        return {
            loaded: true,
            numPapers: this.papers.length,
            avgAbstractLength: avgAbstractLength.toFixed(0),
            years: Object.keys(yearCounts).length,
            yearRange: [
                Math.min(...this.papers.map(p => p.year)),
                Math.max(...this.papers.map(p => p.year)),
            ],
            venues: Object.keys(venueCounts).length,
            topVenues: Object.entries(venueCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([venue, count]) => ({ venue, count })),
            embeddings: embeddingStats,
        };
    }
}

/**
 * Sample paper dataset generator (for testing/demo)
 */
export function generateSamplePapers(count = 100) {
    const venues = [
        'NeurIPS', 'ICML', 'ICLR', 'CVPR', 'EMNLP', 'ACL',
        'SIGIR', 'KDD', 'WWW', 'AAAI', 'IJCAI'
    ];

    const topics = [
        'neural networks', 'transformers', 'attention mechanisms',
        'reinforcement learning', 'computer vision', 'natural language processing',
        'information retrieval', 'knowledge graphs', 'graph neural networks',
        'few-shot learning', 'meta-learning', 'transfer learning',
        'self-supervised learning', 'contrastive learning', 'multimodal learning'
    ];

    const sampleAbstracts = [
        'We propose a novel approach to {topic} that significantly improves performance on standard benchmarks. Our method achieves state-of-the-art results while being computationally efficient.',
        'This paper introduces a new framework for {topic} based on recent advances in deep learning. We demonstrate the effectiveness of our approach through extensive experiments.',
        'We present a comprehensive study of {topic} in the context of modern machine learning systems. Our analysis reveals important insights for practical applications.',
        'In this work, we address key challenges in {topic} by proposing an innovative architecture that leverages recent theoretical insights.',
        'We investigate {topic} from both theoretical and practical perspectives, providing new understanding of fundamental tradeoffs.',
    ];

    const papers = [];

    for (let i = 0; i < count; i++) {
        const topic = topics[Math.floor(Math.random() * topics.length)];
        const abstractTemplate = sampleAbstracts[Math.floor(Math.random() * sampleAbstracts.length)];
        const year = 2018 + Math.floor(Math.random() * 7); // 2018-2024

        papers.push({
            id: i,
            title: `${topic.charAt(0).toUpperCase() + topic.slice(1)}: A Novel Approach`,
            abstract: abstractTemplate.replace('{topic}', topic),
            authors: [
                `Author ${i % 50 + 1}`,
                `Author ${(i + 17) % 50 + 1}`,
                `Author ${(i + 33) % 50 + 1}`,
            ],
            venue: venues[i % venues.length],
            year,
            keywords: [topic, 'machine learning', 'deep learning'],
            arxiv_id: `${year % 100}${String(i).padStart(5, '0')}`,
            doi: `10.1234/${year}.${i}`,
        });
    }

    return papers;
}
