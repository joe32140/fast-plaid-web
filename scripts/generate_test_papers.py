#!/usr/bin/env python3
"""
Generate test papers dataset for offline index building.
Creates realistic-looking papers with titles and abstracts.
"""

import json
import os
import random

# Sample ML/AI topics for generating realistic papers
TOPICS = [
    "transformer", "attention mechanism", "BERT", "GPT", "language model",
    "neural network", "deep learning", "reinforcement learning", "computer vision",
    "natural language processing", "machine translation", "question answering",
    "sentiment analysis", "named entity recognition", "semantic search",
    "embedding", "vector database", "information retrieval", "ranking",
    "ColBERT", "dense retrieval", "sparse retrieval", "hybrid search",
    "FAISS", "approximate nearest neighbor", "quantization", "compression",
    "knowledge distillation", "transfer learning", "fine-tuning", "pre-training",
    "multi-task learning", "meta-learning", "few-shot learning", "zero-shot",
    "contrastive learning", "self-supervised learning", "semi-supervised",
    "active learning", "curriculum learning", "adversarial training",
    "generative model", "diffusion model", "VAE", "GAN", "autoencoder",
    "graph neural network", "recurrent neural network", "convolutional neural network",
    "optimization", "gradient descent", "Adam", "learning rate scheduling",
    "regularization", "dropout", "batch normalization", "layer normalization",
    "cross-entropy loss", "contrastive loss", "triplet loss", "focal loss"
]

METHODS = [
    "Novel", "Improved", "Efficient", "Scalable", "Robust", "Adaptive",
    "Dynamic", "Hierarchical", "Multi-scale", "End-to-end", "Self-supervised",
    "Few-shot", "Zero-shot", "Multi-modal", "Cross-lingual", "Domain-adaptive"
]

TASKS = [
    "for Text Classification", "for Named Entity Recognition", "for Question Answering",
    "for Machine Translation", "for Summarization", "for Information Retrieval",
    "for Semantic Search", "for Document Ranking", "for Image Classification",
    "for Object Detection", "for Speech Recognition", "for Recommendation Systems"
]

def generate_title():
    """Generate a realistic paper title."""
    method = random.choice(METHODS)
    topic1 = random.choice(TOPICS)
    topic2 = random.choice(TOPICS)
    task = random.choice(TASKS)

    templates = [
        f"{method} Approach to {topic1.title()} {task}",
        f"{topic1.title()}: A {method} Method {task}",
        f"Learning {topic1.title()} with {topic2.title()} {task}",
        f"{method} {topic1.title()} via {topic2.title()}",
        f"Towards {method} {topic1.title()}: {topic2.title()} Perspective",
    ]

    return random.choice(templates)

def generate_abstract():
    """Generate a realistic paper abstract."""
    topic1 = random.choice(TOPICS)
    topic2 = random.choice(TOPICS)
    topic3 = random.choice(TOPICS)
    method = random.choice(METHODS).lower()

    templates = [
        f"We propose a {method} approach for {topic1} that leverages {topic2} to improve performance. "
        f"Our method addresses key limitations of existing approaches by incorporating {topic3}. "
        f"Experiments on benchmark datasets demonstrate significant improvements over baseline methods. "
        f"The proposed approach achieves state-of-the-art results while maintaining computational efficiency. "
        f"We provide comprehensive analysis and ablation studies to validate our design choices.",

        f"Recent advances in {topic1} have shown promising results, but scalability remains a challenge. "
        f"In this work, we introduce a {method} framework that combines {topic2} with {topic3}. "
        f"Our approach reduces computational costs while maintaining accuracy. "
        f"We evaluate our method on multiple datasets and show consistent improvements. "
        f"The results suggest that our approach is widely applicable across different domains.",

        f"This paper presents a novel {method} technique for {topic1} using {topic2}. "
        f"We demonstrate that incorporating {topic3} leads to better generalization. "
        f"Extensive experiments validate the effectiveness of our approach. "
        f"Our method outperforms existing baselines on standard benchmarks. "
        f"We release code and pre-trained models to facilitate future research.",
    ]

    return random.choice(templates)

def generate_papers(num_papers=10000):
    """Generate a dataset of synthetic papers."""
    papers = []

    print(f"🎲 Generating {num_papers} synthetic papers...")

    for i in range(num_papers):
        title = generate_title()
        abstract = generate_abstract()

        papers.append({
            'id': f'paper_{i:06d}',
            'title': title,
            'abstract': abstract,
            'text': f"{title} {abstract}"
        })

        if (i + 1) % 1000 == 0:
            print(f"   Generated {i+1}/{num_papers} papers...")

    return papers

def save_papers(papers, output_file):
    """Save papers to JSON file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_file) / 1024 / 1024
    print(f"✅ Saved {len(papers)} papers to {output_file}")
    print(f"📊 File size: {file_size:.2f} MB")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic papers dataset')
    parser.add_argument('--num-papers', type=int, default=10000,
                        help='Number of papers to generate (default: 10000)')
    parser.add_argument('--output', type=str, default='../data/papers_synthetic_10k.json',
                        help='Output file path')

    args = parser.parse_args()

    print(f"🚀 Generating synthetic papers dataset...")
    print(f"   Number: {args.num_papers}")
    print(f"   Output: {args.output}")
    print()

    # Generate papers
    papers = generate_papers(args.num_papers)

    # Save to file
    save_papers(papers, args.output)

    print()
    print("📊 Sample paper:")
    if papers:
        sample = papers[0]
        print(f"   ID: {sample['id']}")
        print(f"   Title: {sample['title']}")
        print(f"   Abstract: {sample['abstract'][:150]}...")

    print()
    print("✅ Dataset generation complete!")
    print(f"📁 Next step: Run build_offline_index.py to create embeddings")
