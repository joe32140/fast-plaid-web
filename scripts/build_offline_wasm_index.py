#!/usr/bin/env python3
"""
Build offline FastPlaid WASM index for instant loading in browser.

This script:
1. Loads papers from JSON
2. Computes ColBERT embeddings (with 2_Dense support)
3. Saves float32 embeddings + papers metadata
4. Optionally builds .fastplaid binary index using Node.js WASM

The output can be loaded instantly in the browser without re-computing embeddings.

Usage:
    python scripts/build_offline_wasm_index.py \\
        --papers data/papers_1000.json \\
        --output demo/data/precomputed \\
        --model mixedbread-ai/mxbai-edge-colbert-v0-17m
"""

import json
import struct
import sys
import time
from pathlib import Path
import argparse

def load_papers(papers_file):
    """Load papers from JSON file."""
    print(f"📂 Loading papers from {papers_file}...")
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"✅ Loaded {len(papers)} papers")
    return papers

def encode_papers_colbert(papers, model_name="mixedbread-ai/mxbai-edge-colbert-v0-17m"):
    """
    Encode papers using ColBERT model with 2_Dense support.
    Returns embeddings list and metadata.
    """
    print(f"🤖 Loading {model_name}...")

    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Load 1_Dense layer
    print("Loading 1_Dense layer...")
    dense1_path = hf_hub_download(repo_id=model_name, filename="1_Dense/model.safetensors")
    dense1_weights = load_file(dense1_path)
    dense1 = nn.Linear(
        in_features=dense1_weights['linear.weight'].shape[1],
        out_features=dense1_weights['linear.weight'].shape[0],
        bias=False
    )
    dense1.weight.data = dense1_weights['linear.weight']
    dense1.eval()

    # Load 2_Dense layer
    print("Loading 2_Dense layer...")
    dense2_path = hf_hub_download(repo_id=model_name, filename="2_Dense/model.safetensors")
    dense2_weights = load_file(dense2_path)
    dense2 = nn.Linear(
        in_features=dense2_weights['linear.weight'].shape[1],
        out_features=dense2_weights['linear.weight'].shape[0],
        bias=False
    )
    dense2.weight.data = dense2_weights['linear.weight']
    dense2.eval()

    embedding_dim = dense2.weight.shape[0]
    print(f"✅ Model loaded: 256 → {dense1.weight.shape[0]} → {embedding_dim}")

    embeddings_list = []
    metadata_list = []

    print(f"🔤 Encoding {len(papers)} papers...")
    start_time = time.time()

    with torch.no_grad():
        for i, paper in enumerate(papers):
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )

            # Forward pass through model and dense layers
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]
            token_embeddings = dense1(token_embeddings)
            token_embeddings = dense2(token_embeddings)

            # L2 normalization (CRITICAL for ColBERT)
            token_norms = torch.norm(token_embeddings, p=2, dim=1, keepdim=True)
            token_embeddings = token_embeddings / (token_norms + 1e-8)

            embedding_np = token_embeddings.cpu().numpy()

            embeddings_list.append(embedding_np)
            metadata_list.append({
                'id': paper['id'],
                'num_tokens': embedding_np.shape[0],
                'embedding_dim': embedding_np.shape[1]
            })

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i + 1}/{len(papers)} ({rate:.1f} papers/sec)")

    total_time = time.time() - start_time
    print(f"✅ Encoded {len(papers)} papers in {total_time:.1f}s")

    return embeddings_list, metadata_list, embedding_dim

def save_embeddings_binary(embeddings_list, metadata_list, embedding_dim, output_dir):
    """
    Save embeddings in binary format for fast WASM loading.
    Format matches what lib_wasm_quantized.rs expects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.bin"
    meta_path = output_dir / "embeddings_meta.json"

    print(f"💾 Saving embeddings to {embeddings_path}...")

    with open(embeddings_path, 'wb') as f:
        # Write header
        num_papers = len(embeddings_list)
        f.write(struct.pack('<I', num_papers))  # u32: number of papers
        f.write(struct.pack('<I', embedding_dim))  # u32: embedding dimension

        # Write each document's embeddings
        for emb in embeddings_list:
            num_tokens = emb.shape[0]
            f.write(struct.pack('<I', num_tokens))  # u32: number of tokens

            # Write flat float32 array
            emb_flat = emb.flatten().astype('float32')
            f.write(emb_flat.tobytes())

    file_size_mb = embeddings_path.stat().st_size / 1_000_000
    print(f"✅ Saved embeddings: {file_size_mb:.2f} MB")

    # Save metadata
    with open(meta_path, 'w') as f:
        json.dump({
            'num_papers': num_papers,
            'embedding_dim': embedding_dim,
            'format': 'float32',
            'total_tokens': sum(m['num_tokens'] for m in metadata_list),
        }, f, indent=2)

    return embeddings_path

def main():
    parser = argparse.ArgumentParser(description='Build offline WASM index')
    parser.add_argument('--papers', type=str, required=True, help='Path to papers JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--model', type=str, default='mixedbread-ai/mxbai-edge-colbert-v0-17m')

    args = parser.parse_args()

    print("🚀 Building Offline WASM Index")
    print("=" * 70)
    print(f"Papers: {args.papers}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print()

    # Load papers
    papers = load_papers(args.papers)

    # Encode papers
    embeddings_list, metadata_list, embedding_dim = encode_papers_colbert(papers, args.model)

    # Save embeddings binary
    embeddings_path = save_embeddings_binary(embeddings_list, metadata_list, embedding_dim, args.output)

    # Save papers metadata (for display in UI)
    papers_meta_path = Path(args.output) / "papers_metadata.json"
    print(f"💾 Saving papers metadata to {papers_meta_path}...")
    metadata = [{
        'id': paper['id'],
        'title': paper['title'],
        'abstract': paper['abstract'],
        'categories': paper.get('categories', []),
        'published': paper.get('published', '')
    } for paper in papers]

    with open(papers_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("📦 Offline Index Built!")
    print("=" * 70)
    print()
    print("📁 Output files:")
    print(f"   {embeddings_path} - Binary embeddings (for WASM)")
    print(f"   {papers_meta_path} - Papers metadata (for UI)")
    print()
    print("🚀 Next steps:")
    print("   1. Browser will load embeddings.bin and papers_metadata.json")
    print("   2. WASM will quantize and build IVF index on-the-fly")
    print("   3. Or use Node.js to pre-build .fastplaid index (see build_fastplaid_index.js)")
    print()
    print("✅ Ready for deployment!")

if __name__ == '__main__':
    main()
