#!/usr/bin/env python3
"""
Build offline index using real mxbai-edge-colbert-v0-17m model.
This creates pre-computed embeddings that can be loaded in the browser.
"""

import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_papers(papers_file):
    """Load papers from JSON file."""
    print(f"📂 Loading papers from {papers_file}...")
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"✅ Loaded {len(papers)} papers")
    return papers

def encode_with_colbert(papers, model_name='mixedbread-ai/mxbai-edge-colbert-v0-17m'):
    """
    Encode papers using real ColBERT model.
    Requires pylate-rs Python bindings (if available) or transformers + custom code.
    """
    print(f"🤖 Loading {model_name}...")

    try:
        # Try using pylate if available
        from pylate import ColBERT
        print("✅ Using pylate ColBERT")

        model = ColBERT(model_name)

        embeddings = []
        print(f"🔤 Encoding {len(papers)} papers...")

        for paper in tqdm(papers, desc="Encoding"):
            # Encode abstract as document
            emb = model.encode([paper['text']], is_query=False)
            embeddings.append({
                'id': paper['id'],
                'embedding': emb[0].tolist(),  # Convert to list for JSON
                'num_tokens': len(emb[0]),
                'embedding_dim': len(emb[0][0]) if len(emb[0]) > 0 else 0
            })

        return embeddings

    except ImportError:
        print("⚠️ pylate not available, trying transformers...")

        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            import torch.nn as nn
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download

            print(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()

            # Load 1_Dense layer
            print("Loading 1_Dense layer...")
            dense1_path = hf_hub_download(
                repo_id=model_name,
                filename="1_Dense/model.safetensors"
            )
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
            dense2_path = hf_hub_download(
                repo_id=model_name,
                filename="2_Dense/model.safetensors"
            )
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

            embeddings = []
            print(f"🔤 Encoding {len(papers)} papers...")

            with torch.no_grad():
                for paper in tqdm(papers, desc="Encoding"):
                    # Tokenize
                    inputs = tokenizer(
                        paper['text'],
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

                    embeddings.append({
                        'id': paper['id'],
                        'embedding': token_embeddings.cpu().numpy().tolist(),
                        'num_tokens': token_embeddings.shape[0],
                        'embedding_dim': token_embeddings.shape[1]
                    })

            return embeddings

        except ImportError:
            print("❌ Neither pylate nor transformers available")
            print("Please install: pip install pylate-rs")
            print("Or: pip install transformers torch")
            return None

def save_embeddings_for_browser(embeddings, papers, output_dir='../data/offline_index'):
    """
    Save embeddings in a format that can be loaded in the browser.
    Creates both FastPlaid index and raw embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata (papers without embeddings)
    metadata = []
    for paper in papers:
        metadata.append({
            'id': paper['id'],
            'title': paper['title'],
            'abstract': paper['abstract'][:500]  # Truncate for size
        })

    metadata_path = os.path.join(output_dir, 'papers_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {metadata_path}")

    # Save embeddings info (without full embedding data for now)
    embeddings_info = []
    for emb in embeddings:
        embeddings_info.append({
            'id': emb['id'],
            'num_tokens': emb['num_tokens'],
            'embedding_dim': emb['embedding_dim']
        })

    info_path = os.path.join(output_dir, 'embeddings_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_info, f, indent=2)
    print(f"✅ Saved embeddings info to {info_path}")

    # Save embeddings in chunks (for easier loading)
    chunk_size = 1000
    for i in range(0, len(embeddings), chunk_size):
        chunk = embeddings[i:i+chunk_size]
        chunk_path = os.path.join(output_dir, f'embeddings_chunk_{i//chunk_size}.json')

        # Convert to more compact format
        compact_chunk = []
        for emb in chunk:
            # Flatten embeddings for storage
            flat_emb = []
            for token_emb in emb['embedding']:
                flat_emb.extend(token_emb)

            compact_chunk.append({
                'id': emb['id'],
                'embedding': flat_emb,
                'num_tokens': emb['num_tokens'],
                'dim': emb['embedding_dim']
            })

        with open(chunk_path, 'w') as f:
            json.dump(compact_chunk, f)

        print(f"✅ Saved chunk {i//chunk_size} ({len(chunk)} papers) to {chunk_path}")

    # Create index manifest
    manifest = {
        'num_papers': len(papers),
        'num_chunks': (len(embeddings) + chunk_size - 1) // chunk_size,
        'chunk_size': chunk_size,
        'embedding_dim': embeddings[0]['embedding_dim'] if embeddings else 0,
        'model': 'mixedbread-ai/mxbai-edge-colbert-v0-17m',
        'version': '1.0'
    }

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Saved manifest to {manifest_path}")

    print()
    print(f"📊 Index Statistics:")
    print(f"   Papers: {len(papers)}")
    print(f"   Chunks: {manifest['num_chunks']}")
    print(f"   Embedding dim: {manifest['embedding_dim']}")
    print(f"   Total size: {sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir)) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build offline index with real embeddings')
    parser.add_argument('--papers', type=str, required=True,
                        help='Input papers JSON file')
    parser.add_argument('--output', type=str, default='../data/offline_index',
                        help='Output directory for index')
    parser.add_argument('--model', type=str, default='mixedbread-ai/mxbai-edge-colbert-v0-17m',
                        help='ColBERT model to use')

    args = parser.parse_args()

    print("🚀 Building offline index...")
    print(f"   Papers: {args.papers}")
    print(f"   Model: {args.model}")
    print(f"   Output: {args.output}")
    print()

    # Load papers
    papers = load_papers(args.papers)

    # Encode with ColBERT
    embeddings = encode_with_colbert(papers, args.model)

    if embeddings is None:
        print("❌ Failed to encode papers")
        exit(1)

    # Save for browser
    save_embeddings_for_browser(embeddings, papers, args.output)

    print()
    print("✅ Offline index building complete!")
    print(f"📁 Files saved to: {args.output}")
    print()
    print("🌐 Next steps:")
    print("   1. Create HTML demo to load these embeddings")
    print("   2. Compare FastPlaid vs Direct MaxSim performance")
    print("   3. Deploy to GitHub Pages")
