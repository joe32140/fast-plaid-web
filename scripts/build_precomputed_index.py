#!/usr/bin/env python3
"""
Build precomputed index with both Direct MaxSim embeddings and FastPlaid index.
Outputs binary format for efficient browser loading.
"""

import json
import os
import struct
import numpy as np
from tqdm import tqdm

def load_papers(papers_file):
    """Load papers from JSON file."""
    print(f"📂 Loading papers from {papers_file}...")
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"✅ Loaded {len(papers)} papers")
    return papers

def encode_papers(papers, model_name="mixedbread-ai/mxbai-edge-colbert-v0-17m"):
    """Encode papers using ColBERT model with 2_Dense support."""
    print(f"🤖 Loading {model_name}...")

    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    print(f"Loading model components...")
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

    print(f"✅ Model loaded successfully!")
    print(f"   ModernBERT output: {model.config.hidden_size}")
    print(f"   1_Dense: {dense1.weight.shape[1]} → {dense1.weight.shape[0]}")
    print(f"   2_Dense: {dense2.weight.shape[1]} → {dense2.weight.shape[0]}")

    embeddings_list = []
    metadata_list = []

    print(f"🔤 Encoding {len(papers)} papers...")

    with torch.no_grad():
        for paper in tqdm(papers, desc="Encoding"):
            # Tokenize (title + abstract)
            # Handle both 'text' field and separate 'title'/'abstract' fields
            if 'text' in paper:
                text = paper['text']
            else:
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )

            # Get embeddings from ModernBERT
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

            # Apply 1_Dense layer
            token_embeddings = dense1(token_embeddings)  # [seq_len, 512]

            # Apply 2_Dense layer
            token_embeddings = dense2(token_embeddings)  # [seq_len, 48]

            # CRITICAL: Normalize each token vector (L2 normalization)
            # ColBERT requires normalized embeddings for correct MaxSim scoring
            token_norms = torch.norm(token_embeddings, p=2, dim=1, keepdim=True)
            token_embeddings = token_embeddings / (token_norms + 1e-8)  # [seq_len, 48]

            # Convert to numpy
            embedding_np = token_embeddings.cpu().numpy()  # [seq_len, 48]

            embeddings_list.append(embedding_np)
            metadata_list.append({
                'id': paper['id'],
                'num_tokens': embedding_np.shape[0],
                'embedding_dim': embedding_np.shape[1]
            })

    return embeddings_list, metadata_list

def save_binary_embeddings(embeddings_list, metadata_list, output_dir):
    """
    Save embeddings in binary format for efficient browser loading.
    Format: [num_papers][paper1_num_tokens][paper1_embeddings][paper2_num_tokens][paper2_embeddings]...
    All floats are float32, all ints are uint32.
    """
    print(f"💾 Saving embeddings in binary format...")

    os.makedirs(output_dir, exist_ok=True)

    # Save binary embeddings
    bin_path = os.path.join(output_dir, 'embeddings.bin')
    with open(bin_path, 'wb') as f:
        # Write header: num_papers, embedding_dim
        f.write(struct.pack('II', len(embeddings_list), embeddings_list[0].shape[1]))

        # Write each paper's embeddings
        for emb in embeddings_list:
            # Write number of tokens for this paper
            f.write(struct.pack('I', emb.shape[0]))
            # Write embeddings as float32
            f.write(emb.astype(np.float32).tobytes())

    bin_size = os.path.getsize(bin_path) / 1024 / 1024
    print(f"   ✅ Binary embeddings: {bin_size:.2f} MB")

    # Save metadata (JSON)
    meta_path = os.path.join(output_dir, 'embeddings_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'num_papers': len(embeddings_list),
            'embedding_dim': embeddings_list[0].shape[1],
            'papers': metadata_list
        }, f, indent=2)

    meta_size = os.path.getsize(meta_path) / 1024
    print(f"   ✅ Metadata: {meta_size:.2f} KB")

    return bin_path, meta_path

def build_fastplaid_index(embeddings_list, metadata_list, output_dir):
    """
    Build FastPlaid index from embeddings.
    Uses 4-bit quantization and indexing.
    """
    print(f"🏗️  Building FastPlaid index...")

    # Collect all embeddings into one array for indexing
    all_embeddings = []
    doc_boundaries = [0]  # Start positions for each document

    for emb in embeddings_list:
        all_embeddings.append(emb)
        doc_boundaries.append(doc_boundaries[-1] + emb.shape[0])

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)  # [total_tokens, 48]
    print(f"   Total tokens: {all_embeddings.shape[0]:,}")
    print(f"   Embedding dim: {all_embeddings.shape[1]}")

    # Quantize to 4-bit (simulate - in practice this would use actual quantization)
    # For now, quantize to uint8 (8-bit) as a demonstration
    # Real 4-bit would pack 2 values per byte
    min_val = all_embeddings.min(axis=0)
    max_val = all_embeddings.max(axis=0)

    # Scale to 0-255
    scaled = (all_embeddings - min_val) / (max_val - min_val + 1e-8)
    quantized = (scaled * 255).astype(np.uint8)

    print(f"   Quantized to 8-bit (demo - 4-bit would be 2x smaller)")

    # Save quantized index
    index_path = os.path.join(output_dir, 'fastplaid_index.bin')
    with open(index_path, 'wb') as f:
        # Header
        f.write(struct.pack('III',
                           all_embeddings.shape[0],  # total_tokens
                           all_embeddings.shape[1],  # embedding_dim
                           len(embeddings_list)))    # num_papers

        # Min/max for dequantization
        f.write(min_val.astype(np.float32).tobytes())
        f.write(max_val.astype(np.float32).tobytes())

        # Document boundaries
        f.write(np.array(doc_boundaries, dtype=np.uint32).tobytes())

        # Quantized embeddings
        f.write(quantized.tobytes())

    index_size = os.path.getsize(index_path) / 1024 / 1024
    print(f"   ✅ FastPlaid index: {index_size:.2f} MB")

    # Save index metadata
    index_meta_path = os.path.join(output_dir, 'fastplaid_meta.json')
    with open(index_meta_path, 'w') as f:
        json.dump({
            'total_tokens': int(all_embeddings.shape[0]),
            'embedding_dim': int(all_embeddings.shape[1]),
            'num_papers': len(embeddings_list),
            'quantization': '8-bit',  # Would be '4-bit' in production
            'compression_ratio': f'{all_embeddings.nbytes / os.path.getsize(index_path):.2f}x'
        }, f, indent=2)

    meta_size = os.path.getsize(index_meta_path) / 1024
    print(f"   ✅ Index metadata: {meta_size:.2f} KB")

    return index_path, index_meta_path

def save_papers_metadata(papers, output_dir):
    """Save paper metadata (titles, abstracts) for display."""
    print(f"📄 Saving papers metadata...")

    metadata = []
    for paper in papers:
        metadata.append({
            'id': paper['id'],
            'title': paper['title'],
            'abstract': paper['abstract'],
            'categories': paper.get('categories', []),
            'published': paper.get('published', '')
        })

    meta_path = os.path.join(output_dir, 'papers_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    meta_size = os.path.getsize(meta_path) / 1024
    print(f"   ✅ Papers metadata: {meta_size:.2f} KB")

    return meta_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build precomputed index for browser demo')
    parser.add_argument('--papers', type=str, required=True, help='Path to papers JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--model', type=str, default='mixedbread-ai/mxbai-edge-colbert-v0-17m',
                        help='Model name')

    args = parser.parse_args()

    print("🚀 Building Precomputed Index")
    print("=" * 50)
    print(f"Papers: {args.papers}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print()

    # Load papers
    papers = load_papers(args.papers)

    # Encode papers
    embeddings_list, metadata_list = encode_papers(papers, args.model)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save binary embeddings (for Direct MaxSim)
    bin_path, meta_path = save_binary_embeddings(embeddings_list, metadata_list, args.output)

    # Build FastPlaid index
    index_path, index_meta_path = build_fastplaid_index(embeddings_list, metadata_list, args.output)

    # Save papers metadata
    papers_meta_path = save_papers_metadata(papers, args.output)

    # Summary
    print()
    print("=" * 50)
    print("📊 Index Build Complete!")
    print("=" * 50)
    print()

    print("📁 Output files:")
    print(f"   {bin_path}")
    print(f"   {meta_path}")
    print(f"   {index_path}")
    print(f"   {index_meta_path}")
    print(f"   {papers_meta_path}")
    print()

    # Calculate sizes
    direct_size = os.path.getsize(bin_path) / 1024 / 1024
    fastplaid_size = os.path.getsize(index_path) / 1024 / 1024

    print("📊 Size Comparison:")
    print(f"   Direct MaxSim: {direct_size:.2f} MB (float32 embeddings)")
    print(f"   FastPlaid: {fastplaid_size:.2f} MB (8-bit quantized)")
    print(f"   Compression: {direct_size / fastplaid_size:.2f}x smaller")
    print()

    print("✅ Ready for browser demo!")
    print()
    print("🔧 Next steps:")
    print("   1. Create JavaScript loader for binary format")
    print("   2. Update papers-demo.html to use precomputed index")
    print("   3. Add size and speed comparison UI")

if __name__ == '__main__':
    main()
