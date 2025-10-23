#!/usr/bin/env node
/**
 * Build .fastplaid index using headless browser (Puppeteer)
 *
 * This works around Node.js WASM limitations by using a real browser engine.
 *
 * Usage:
 *   npm install puppeteer  # First time only
 *   node scripts/build_index_headless.js
 */

import puppeteer from 'puppeteer';
import { writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

async function buildIndex() {
    console.log('🚀 Building .fastplaid index using headless browser...');
    console.log('=' .repeat(70));

    // Launch browser
    console.log('📦 Launching headless Chrome...');
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    try {
        const page = await browser.newPage();

        // Listen to console logs
        page.on('console', msg => {
            const text = msg.text();
            if (text.includes('✅') || text.includes('📥') || text.includes('🗜️') ||
                text.includes('🔧') || text.includes('💾')) {
                console.log(text);
            }
        });

        // Navigate to build page
        const buildUrl = `file://${rootDir}/demo/build-index.html`;
        console.log(`📄 Loading ${buildUrl}...`);
        await page.goto(buildUrl, { waitUntil: 'networkidle0' });

        // Click build button and wait for it to complete
        console.log('🔨 Building index...');
        await page.click('#buildBtn');

        // Wait for the download to trigger (look for success message)
        await page.waitForFunction(
            () => document.getElementById('output').textContent.includes('Download started'),
            { timeout: 120000 } // 2 minutes
        );

        // Get the index bytes from the page
        console.log('💾 Extracting index bytes...');
        const indexBytes = await page.evaluate(() => {
            // Re-build to get the bytes (since download already happened)
            return new Promise(async (resolve, reject) => {
                try {
                    const { FastPlaidQuantized } = await import('./pkg/fast_plaid_rust.js');

                    // Load embeddings
                    const embResponse = await fetch('./data/fastplaid_4bit/embeddings.bin');
                    const embData = await embResponse.arrayBuffer();
                    const view = new DataView(embData);

                    let offset = 0;
                    const numPapers = view.getUint32(offset, true);
                    offset += 4;
                    const embeddingDim = view.getUint32(offset, true);
                    offset += 4;

                    const embeddings = [];
                    const docInfo = [];

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
                    }

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

                    const docInfoArray = new BigInt64Array(docInfo);

                    // Build index
                    const fastplaid = new FastPlaidQuantized();
                    await fastplaid.load_documents_quantized(flatEmbeddings, docInfoArray, 256);

                    // Get bytes
                    const indexBytes = fastplaid.save_index();
                    resolve(Array.from(indexBytes));
                } catch (error) {
                    reject(error);
                }
            });
        });

        // Save to file
        const outputPath = join(rootDir, 'demo/data/index.fastplaid');
        writeFileSync(outputPath, Buffer.from(indexBytes));

        const sizeMB = (indexBytes.length / 1_000_000).toFixed(2);
        console.log(`✅ Saved ${outputPath} (${sizeMB} MB)`);

        console.log();
        console.log('=' .repeat(70));
        console.log('✅ Index built successfully!');

    } finally {
        await browser.close();
    }
}

buildIndex().catch(err => {
    console.error('❌ Error:', err);
    process.exit(1);
});
