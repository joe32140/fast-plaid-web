#!/usr/bin/env node
/**
 * Rebuild FastPlaid index using Rust WASM quantization with learned bucket weights
 * This replaces the Python-generated index with proper residual quantization
 */

import { readFile, writeFile } from 'fs/promises';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import init, { FastPlaidQuantized } from '../docs/node_modules/fast-plaid-rust/fast_plaid_rust.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, '..');

async function main() {
    console.log('🚀 Rebuilding FastPlaid index with Rust quantization...\n');

    // Initialize WASM
    console.log('📦 Loading WASM module...');
    const wasmPath = join(projectRoot, 'docs/node_modules/fast-plaid-rust/fast_plaid_rust_bg.wasm');
    const wasmBuffer = await readFile(wasmPath);
    await init(wasmBuffer);
    console.log('✅ WASM loaded\n');

    // Load embeddings.bin
    const embPath = join(projectRoot, 'docs/data/embeddings.bin');
    console.log(`📥 Loading embeddings from ${embPath}...`);
    const embData = await readFile(embPath);

    const view = new DataView(embData.buffer, embData.byteOffset, embData.byteLength);
    let offset = 0;

    const numPapers = view.getUint32(offset, true);
    offset += 4;
    const embeddingDim = view.getUint32(offset, true);
    offset += 4;

    console.log(`   Papers: ${numPapers.toLocaleString()}`);
    console.log(`   Embedding dim: ${embeddingDim}`);

    // Read all embeddings
    const embeddings = [];
    const docInfo = [];
    let totalTokens = 0;

    for (let i = 0; i < numPapers; i++) {
        const numTokens = view.getUint32(offset, true);
        offset += 4;

        const embedding = new Float32Array(numTokens * embeddingDim);
        for (let j = 0; j < numTokens * embeddingDim; j++) {
            embedding[j] = view.getFloat32(offset, true);
            offset += 4;
        }

        embeddings.push(embedding);
        docInfo.push(BigInt(i));
        docInfo.push(BigInt(numTokens));
        totalTokens += numTokens;
    }

    console.log(`   Total tokens: ${totalTokens.toLocaleString()}`);
    console.log(`✅ Loaded ${numPapers} papers\n`);

    // Flatten embeddings
    const totalSize = embeddings.reduce((sum, emb) => sum + emb.length, 0);
    const flatEmbeddings = new Float32Array(totalSize);
    let writeOffset = 0;
    for (const emb of embeddings) {
        flatEmbeddings.set(emb, writeOffset);
        writeOffset += emb.length;
    }

    const docInfoArray = new BigInt64Array(docInfo);

    // Build index with Rust quantization
    console.log('🗜️ Building FastPlaid index with learned bucket weights...');
    console.log('   This will:');
    console.log('   - Train 256 k-means centroids');
    console.log('   - Compute average residuals (Phase 1)');
    console.log('   - Learn per-dimension bucket weights from percentiles');
    console.log('   - Compute binary search cutoffs (Phase 2)');
    console.log('   (This may take 30-60 seconds...)\n');

    const startTime = Date.now();
    const fastplaid = new FastPlaidQuantized();
    await fastplaid.load_documents_quantized(flatEmbeddings, docInfoArray, 256);
    const buildTime = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log(`✅ Index built in ${buildTime}s\n`);

    // Save index
    console.log('💾 Saving index...');
    const indexBytes = fastplaid.save_index();

    const outputPath = join(projectRoot, 'docs/data/fastplaid_index_rust.bin');
    await writeFile(outputPath, indexBytes);

    const sizeMB = (indexBytes.length / 1_000_000).toFixed(2);
    const originalSizeMB = (embData.length / 1_000_000).toFixed(2);
    const compressionRatio = (embData.length / indexBytes.length).toFixed(1);

    console.log(`✅ Index saved to: ${outputPath}`);
    console.log(`   Original size: ${originalSizeMB} MB`);
    console.log(`   Compressed size: ${sizeMB} MB`);
    console.log(`   Compression ratio: ${compressionRatio}x\n`);

    console.log('🎉 Done! Now update demo to load this new index.');
    console.log('   The new index has LEARNED bucket weights, not uniform quantization!');
}

main().catch(err => {
    console.error('❌ Error:', err);
    process.exit(1);
});
