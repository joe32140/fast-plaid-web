#!/usr/bin/env node
/**
 * Build .fastplaid binary index using WASM (offline)
 *
 * This script uses the FastPlaidQuantized WASM module to:
 * 1. Load float32 embeddings from embeddings.bin
 * 2. Quantize to 4-bit
 * 3. Build IVF clusters
 * 4. Save to .fastplaid binary format
 *
 * Usage:
 *     node scripts/build_fastplaid_index.js demo/data/fastplaid_4bit demo/data/index.fastplaid
 */

import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import init, { FastPlaidQuantized } from '../demo/pkg/fast_plaid_rust.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

// Initialize WASM module with explicit path
const wasmPath = join(rootDir, 'demo/pkg/fast_plaid_rust_bg.wasm');
const wasmBytes = readFileSync(wasmPath);
await init({ module_or_path: wasmBytes });

async function loadEmbeddings(dataDir) {
    console.log('📥 Loading embeddings...');

    const embPath = join(dataDir, 'embeddings.bin');
    const embData = readFileSync(embPath);

    let offset = 0;
    const view = new DataView(embData.buffer, embData.byteOffset, embData.byteLength);

    // Read header
    const numPapers = view.getUint32(offset, true);
    offset += 4;
    const embeddingDim = view.getUint32(offset, true);
    offset += 4;

    console.log(`   Papers: ${numPapers}, Dim: ${embeddingDim}`);

    // Read embeddings
    const embeddings = [];
    const docInfo = [];

    for (let i = 0; i < numPapers; i++) {
        const numTokens = view.getUint32(offset, true);
        offset += 4;

        const embSize = numTokens * embeddingDim;
        const docEmb = new Float32Array(embSize);

        for (let j = 0; j < embSize; j++) {
            docEmb[j] = view.getFloat32(offset, true);
            offset += 4;
        }

        embeddings.push(docEmb);
        docInfo.push(BigInt(i));  // doc id
        docInfo.push(BigInt(numTokens));  // num tokens
    }

    // Flatten embeddings
    let totalSize = 0;
    for (const emb of embeddings) {
        totalSize += emb.length;
    }

    const flatEmbeddings = new Float32Array(totalSize);
    let writeOffset = 0;
    for (const emb of embeddings) {
        flatEmbeddings.set(emb, writeOffset);
        writeOffset += emb.length;
    }

    console.log(`✅ Loaded ${numPapers} papers, ${totalSize} total embeddings`);

    return {
        embeddings: flatEmbeddings,
        docInfo: new BigInt64Array(docInfo),
        embeddingDim
    };
}

async function main() {
    const args = process.argv.slice(2);

    if (args.length < 2) {
        console.error('Usage: node build_fastplaid_index.js <data_dir> <output.fastplaid>');
        process.exit(1);
    }

    const dataDir = args[0];
    const outputPath = args[1];

    console.log('🚀 Building .fastplaid index');
    console.log('=' .repeat(70));
    console.log(`Data dir: ${dataDir}`);
    console.log(`Output: ${outputPath}`);
    console.log();

    // Load embeddings
    const { embeddings, docInfo, embeddingDim } = await loadEmbeddings(dataDir);

    // Create FastPlaidQuantized instance
    console.log('🗜️ Quantizing and building IVF index...');
    const fastplaid = new FastPlaidQuantized();

    // Load and quantize (WASM will build IVF automatically)
    await fastplaid.load_documents_quantized(embeddings, docInfo, 256);

    // Save to binary
    console.log('💾 Saving .fastplaid index...');
    const indexBytes = fastplaid.save_index();
    writeFileSync(outputPath, Buffer.from(indexBytes));

    const sizeMB = indexBytes.length / 1_000_000;
    console.log(`✅ Saved ${outputPath} (${sizeMB.toFixed(2)} MB)`);

    console.log();
    console.log('=' .repeat(70));
    console.log('✅ .fastplaid index built!');
    console.log();
    console.log('🚀 Usage in browser:');
    console.log('   const fastplaid = new FastPlaidQuantized();');
    console.log(`   const indexBytes = await fetch('${outputPath}').then(r => r.arrayBuffer());`);
    console.log('   fastplaid.load_index(new Uint8Array(indexBytes));');
    console.log('   // Ready to search!');
}

main().catch(err => {
    console.error('❌ Error:', err);
    process.exit(1);
});
