#!/usr/bin/env node
/**
 * Pre-compute embeddings for 1000 arXiv papers using mxbai-edge-colbert.
 * Saves the embeddings and quantized index for fast loading in the browser demo.
 */

import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const dataDir = join(rootDir, 'data');
const demoDir = join(rootDir, 'demo');

console.log('🔄 Loading mxbai-integration module...');

// Dynamic import of the mxbai integration
const { MxbaiEdgeColbertIntegration } = await import(join(demoDir, 'mxbai-integration.js'));

async function main() {
    console.log('📄 Loading 1000 papers...');
    const papersPath = join(dataDir, 'papers_1000.json');
    const papers = JSON.parse(readFileSync(papersPath, 'utf-8'));
    console.log(`✅ Loaded ${papers.length} papers`);

    // Initialize model
    console.log('🤖 Initializing mxbai-edge-colbert model...');
    const model = new MxbaiEdgeColbertIntegration();
    await model.initializeModel();
    console.log('✅ Model initialized');

    // Compute embeddings
    console.log('🧮 Computing embeddings...');
    const startTime = Date.now();
    const embeddings = [];

    for (let i = 0; i < papers.length; i++) {
        const paper = papers[i];
        const text = `${paper.title} ${paper.abstract}`;

        const embedding = await model.encode([text]);
        embeddings.push({
            id: paper.id,
            embedding: Array.from(embedding.data),
            numTokens: embedding.numTokens[0]
        });

        if ((i + 1) % 50 === 0) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            const rate = ((i + 1) / (Date.now() - startTime) * 1000).toFixed(2);
            console.log(`  Progress: ${i + 1}/${papers.length} (${rate} papers/sec, ${elapsed}s elapsed)`);
        }
    }

    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`✅ Computed embeddings in ${totalTime}s`);

    // Save embeddings
    const embeddingsPath = join(dataDir, 'embeddings_1000.json');
    console.log(`💾 Saving embeddings to ${embeddingsPath}...`);
    writeFileSync(embeddingsPath, JSON.stringify({
        papers: papers,
        embeddings: embeddings,
        metadata: {
            model: 'mixedbread-ai/mxbai-edge-colbert-v0-17m',
            embeddingDim: model.embeddingDim,
            numPapers: papers.length,
            computedAt: new Date().toISOString()
        }
    }, null, 2));
    console.log('✅ Embeddings saved');

    // TODO: Create quantized index using FastPlaid WASM
    console.log('\n⚠️  Next: Use browser to create quantized index from these embeddings');
}

main().catch(err => {
    console.error('❌ Error:', err);
    process.exit(1);
});
