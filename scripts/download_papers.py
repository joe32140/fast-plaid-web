#!/usr/bin/env python3
"""
Download real papers from HuggingFace datasets for offline index building.
Target: 5k-10k papers with titles and abstracts.
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm

def download_arxiv_papers(num_papers=10000, output_file='papers_10k.json'):
    """
    Download papers from ArXiv dataset on HuggingFace.

    Dataset: ccdv/arxiv-classification
    Contains: title, abstract, categories
    """
    print(f"📥 Downloading {num_papers} papers from ArXiv dataset...")

    # Load ArXiv dataset
    # This dataset has ~250k papers with titles and abstracts
    dataset = load_dataset("ccdv/arxiv-classification", split='train', streaming=True)

    papers = []

    print(f"📄 Processing papers...")
    for i, paper in enumerate(tqdm(dataset, total=num_papers, desc="Downloading")):
        if i >= num_papers:
            break

        papers.append({
            'id': f'arxiv_{i}',
            'title': paper['title'],
            'abstract': paper['abstract'],
            'categories': paper['categories'] if 'categories' in paper else 'Unknown',
            'text': f"{paper['title']} {paper['abstract']}"  # Combined for embedding
        })

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(papers)} papers to {output_path}")
    print(f"📊 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return papers

def download_semantic_scholar_papers(num_papers=10000, output_file='papers_semantic_scholar_10k.json'):
    """
    Alternative: Download from Semantic Scholar dataset.

    Dataset: allenai/s2orc (Semantic Scholar Open Research Corpus)
    """
    print(f"📥 Downloading {num_papers} papers from Semantic Scholar...")

    # Note: s2orc is very large, use streaming
    dataset = load_dataset("allenai/s2orc", split='train', streaming=True)

    papers = []

    print(f"📄 Processing papers...")
    for i, paper in enumerate(tqdm(dataset, total=num_papers, desc="Downloading")):
        if i >= num_papers:
            break

        # S2ORC has more detailed metadata
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        if not title or not abstract:
            continue

        papers.append({
            'id': f's2_{paper.get("paper_id", i)}',
            'title': title,
            'abstract': abstract,
            'year': paper.get('year', None),
            'venue': paper.get('venue', 'Unknown'),
            'authors': paper.get('authors', []),
            'text': f"{title} {abstract}"
        })

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(papers)} papers to {output_path}")
    print(f"📊 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return papers

def download_pubmed_papers(num_papers=10000, output_file='papers_pubmed_10k.json'):
    """
    Download biomedical papers from PubMed.

    Dataset: pubmed (medical/biomedical papers)
    """
    print(f"📥 Downloading {num_papers} papers from PubMed...")

    dataset = load_dataset("pubmed", split='train', streaming=True)

    papers = []

    print(f"📄 Processing papers...")
    for i, paper in enumerate(tqdm(dataset, total=num_papers, desc="Downloading")):
        if i >= num_papers:
            break

        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        if not title or not abstract:
            continue

        papers.append({
            'id': f'pubmed_{paper.get("pmid", i)}',
            'title': title,
            'abstract': abstract,
            'text': f"{title} {abstract}"
        })

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(papers)} papers to {output_path}")
    print(f"📊 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return papers

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download papers from HuggingFace datasets')
    parser.add_argument('--dataset', choices=['arxiv', 'semantic-scholar', 'pubmed'],
                        default='arxiv', help='Dataset to download from')
    parser.add_argument('--num-papers', type=int, default=10000,
                        help='Number of papers to download (default: 10000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: papers_{dataset}_{num}.json)')

    args = parser.parse_args()

    if args.output is None:
        args.output = f'papers_{args.dataset}_{args.num_papers}.json'

    print(f"🚀 Starting paper download...")
    print(f"   Dataset: {args.dataset}")
    print(f"   Target: {args.num_papers} papers")
    print(f"   Output: {args.output}")
    print()

    if args.dataset == 'arxiv':
        papers = download_arxiv_papers(args.num_papers, args.output)
    elif args.dataset == 'semantic-scholar':
        papers = download_semantic_scholar_papers(args.num_papers, args.output)
    elif args.dataset == 'pubmed':
        papers = download_pubmed_papers(args.num_papers, args.output)

    print()
    print("📊 Sample paper:")
    if papers:
        sample = papers[0]
        print(f"   ID: {sample['id']}")
        print(f"   Title: {sample['title'][:80]}...")
        print(f"   Abstract: {sample['abstract'][:100]}...")

    print()
    print("✅ Download complete!")
    print(f"📁 Next step: Run build_offline_index.py to create embeddings")
