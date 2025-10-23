#!/usr/bin/env python3
"""
Build optimized FastPlaid index with 4-bit quantization and clustering.
No shortcuts - production-ready implementation.
"""

import json
import os
import struct
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Load Dense layers
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

    print(f"✅ Model loaded: {model.config.hidden_size} → {dense1.weight.shape[0]} → {dense2.weight.shape[0]}")

    embeddings_list = []
    metadata_list = []

    print(f"🔤 Encoding {len(papers)} papers...")

    with torch.no_grad():
        for paper in tqdm(papers, desc="Encoding"):
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

            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]
            token_embeddings = dense1(token_embeddings)
            token_embeddings = dense2(token_embeddings)

            # CRITICAL: Normalize each token vector (L2 normalization)
            # ColBERT requires normalized embeddings for correct MaxSim scoring
            token_norms = torch.norm(token_embeddings, p=2, dim=1, keepdim=True)
            token_embeddings = token_embeddings / (token_norms + 1e-8)

            embedding_np = token_embeddings.cpu().numpy()

            embeddings_list.append(embedding_np)
            metadata_list.append({
                'id': paper['id'],
                'num_tokens': embedding_np.shape[0],
                'embedding_dim': embedding_np.shape[1]
            })

    return embeddings_list, metadata_list

def quantize_4bit(embeddings_list):
    """
    Quantize embeddings to 4-bit with optimal parameters.
    Returns packed 4-bit data with min/max per dimension.
    """
    print(f"🔢 Quantizing to 4-bit...")

    # Collect all embeddings
    all_embeddings = np.vstack(embeddings_list)
    print(f"   Total tokens: {all_embeddings.shape[0]:,}")
    print(f"   Embedding dim: {all_embeddings.shape[1]}")

    # Compute min/max per dimension for better quantization
    min_vals = all_embeddings.min(axis=0).astype(np.float32)
    max_vals = all_embeddings.max(axis=0).astype(np.float32)

    # Quantize to 4-bit (0-15)
    scaled = (all_embeddings - min_vals) / (max_vals - min_vals + 1e-8)
    quantized = np.clip(np.round(scaled * 15), 0, 15).astype(np.uint8)

    # Pack 2 values per byte
    num_values = quantized.size
    if num_values % 2 != 0:
        # Pad if odd number
        quantized_flat = np.append(quantized.flatten(), 0)
    else:
        quantized_flat = quantized.flatten()

    # Pack: value1 in lower nibble, value2 in upper nibble
    packed = np.zeros(len(quantized_flat) // 2, dtype=np.uint8)
    packed = (quantized_flat[1::2] << 4) | (quantized_flat[0::2] & 0x0F)

    print(f"   ✅ Quantized: {all_embeddings.nbytes / 1024 / 1024:.2f} MB → {packed.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Compression: {all_embeddings.nbytes / packed.nbytes:.1f}x")

    return packed, min_vals, max_vals

def build_clusters(embeddings_list, n_clusters=50):
    """
    Build k-means clusters for fast approximate search.
    """
    print(f"🎯 Building {n_clusters} clusters...")

    # Use first token of each document as representative
    doc_reps = np.array([emb[0] for emb in embeddings_list])

    # K-means clustering
    kmeans = MiniBatchKMeans(
        n_clusters=min(n_clusters, len(doc_reps)),
        random_state=42,
        batch_size=100,
        verbose=0
    )
    cluster_labels = kmeans.fit_predict(doc_reps)
    centroids = kmeans.cluster_centers_

    # Build cluster → documents mapping
    clusters = [[] for _ in range(kmeans.n_clusters)]
    for doc_id, cluster_id in enumerate(cluster_labels):
        clusters[cluster_id].append(doc_id)

    print(f"   ✅ Created {kmeans.n_clusters} clusters")
    print(f"   Avg docs per cluster: {len(doc_reps) / kmeans.n_clusters:.1f}")

    return centroids, clusters, cluster_labels

def save_fastplaid_index(packed_data, min_vals, max_vals, embeddings_list, centroids, clusters, cluster_labels, papers, output_dir):
    """
    Save optimized FastPlaid index with 4-bit quantization and clustering.
    """
    print(f"💾 Saving FastPlaid index...")

    os.makedirs(output_dir, exist_ok=True)

    # Calculate document boundaries
    doc_boundaries = [0]
    embedding_dim = embeddings_list[0].shape[1]

    for emb in embeddings_list:
        doc_boundaries.append(doc_boundaries[-1] + emb.shape[0])

    # Save binary index
    index_path = os.path.join(output_dir, 'fastplaid_4bit.bin')
    with open(index_path, 'wb') as f:
        # Header
        f.write(struct.pack('IIII',
                           doc_boundaries[-1],        # total_tokens
                           embedding_dim,             # embedding_dim
                           len(embeddings_list),      # num_papers
                           len(centroids)))           # num_clusters

        # Min/max for dequantization (per dimension)
        f.write(min_vals.tobytes())
        f.write(max_vals.tobytes())

        # Document boundaries
        f.write(np.array(doc_boundaries, dtype=np.uint32).tobytes())

        # Cluster centroids
        f.write(centroids.astype(np.float32).tobytes())

        # Cluster labels (which cluster each document belongs to)
        f.write(np.array(cluster_labels, dtype=np.uint16).tobytes())

        # Cluster sizes and document lists
        for cluster in clusters:
            f.write(struct.pack('I', len(cluster)))  # cluster size
            f.write(np.array(cluster, dtype=np.uint32).tobytes())  # document IDs

        # Packed 4-bit embeddings
        f.write(packed_data.tobytes())

    index_size = os.path.getsize(index_path) / 1024 / 1024
    print(f"   ✅ FastPlaid index: {index_size:.2f} MB")

    # Save metadata
    meta_path = os.path.join(output_dir, 'fastplaid_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'total_tokens': int(doc_boundaries[-1]),
            'embedding_dim': int(embedding_dim),
            'num_papers': len(embeddings_list),
            'num_clusters': len(centroids),
            'quantization': '4-bit',
            'compression_vs_float32': f'{(len(embeddings_list) * embedding_dim * doc_boundaries[-1] * 4) / (index_size * 1024 * 1024):.1f}x'
        }, f, indent=2)

    # Save papers metadata
    papers_meta_path = os.path.join(output_dir, 'papers_metadata.json')
    metadata = [{
        'id': paper['id'],
        'title': paper['title'],
        'abstract': paper['abstract'],
        'categories': paper.get('categories', []),
        'published': paper.get('published', '')
    } for paper in papers]

    with open(papers_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return index_path, meta_path, papers_meta_path, index_size

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build optimized FastPlaid index with 4-bit quantization')
    parser.add_argument('--papers', type=str, required=True, help='Path to papers JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--clusters', type=int, default=50, help='Number of clusters (default: 50)')
    parser.add_argument('--model', type=str, default='mixedbread-ai/mxbai-edge-colbert-v0-17m')

    args = parser.parse_args()

    print("🚀 Building Optimized FastPlaid Index")
    print("=" * 70)
    print(f"Papers: {args.papers}")
    print(f"Output: {args.output}")
    print(f"Clusters: {args.clusters}")
    print(f"Model: {args.model}")
    print()

    # Load and encode papers
    papers = load_papers(args.papers)
    embeddings_list, metadata_list = encode_papers(papers, args.model)

    # Quantize to 4-bit
    packed_data, min_vals, max_vals = quantize_4bit(embeddings_list)

    # Build clusters
    centroids, clusters, cluster_labels = build_clusters(embeddings_list, args.clusters)

    # Save index
    index_path, meta_path, papers_path, index_size = save_fastplaid_index(
        packed_data, min_vals, max_vals, embeddings_list,
        centroids, clusters, cluster_labels, papers, args.output
    )

    # Calculate baseline sizes
    embedding_dim = embeddings_list[0].shape[1]
    total_tokens = sum(emb.shape[0] for emb in embeddings_list)
    float32_size = total_tokens * embedding_dim * 4 / 1024 / 1024

    print()
    print("=" * 70)
    print("📊 FastPlaid Index Complete!")
    print("=" * 70)
    print()
    print(f"📁 Output files:")
    print(f"   {index_path}")
    print(f"   {meta_path}")
    print(f"   {papers_path}")
    print()
    print(f"📊 Size Comparison:")
    print(f"   Float32 (baseline): {float32_size:.2f} MB")
    print(f"   FastPlaid (4-bit): {index_size:.2f} MB")
    print(f"   Compression: {float32_size / index_size:.1f}x smaller")
    print()
    print(f"🎯 Index Stats:")
    print(f"   Papers: {len(papers)}")
    print(f"   Clusters: {len(centroids)}")
    print(f"   Avg papers/cluster: {len(papers) / len(centroids):.1f}")
    print(f"   Embedding dim: {embedding_dim}")
    print()
    print("✅ Ready for production use!")

if __name__ == '__main__':
    main()
