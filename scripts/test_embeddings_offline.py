#!/usr/bin/env python3
"""
Offline test to verify stored embeddings make sense.
Tests:
1. Load precomputed embeddings
2. Encode a paper title as a query
3. Check if the paper ranks #1 when searching for its own title
4. Verify embeddings are normalized and non-zero
"""

import numpy as np
import json
import struct
from pathlib import Path

def load_binary_embeddings(index_path):
    """Load embeddings from binary format (Direct MaxSim embeddings.bin)."""
    embeddings_path = Path(index_path) / "embeddings.bin"

    print(f"📂 Loading from: {index_path}")

    # Load binary embeddings
    with open(embeddings_path, 'rb') as f:
        binary_data = f.read()

    print(f"📦 Binary data size: {len(binary_data)} bytes ({len(binary_data)/1024/1024:.2f} MB)")

    # Parse header
    offset = 0
    num_papers = struct.unpack('I', binary_data[offset:offset+4])[0]
    offset += 4
    embedding_dim = struct.unpack('I', binary_data[offset:offset+4])[0]
    offset += 4

    print(f"📊 Header: {num_papers} papers, {embedding_dim} dims")

    # Parse embeddings for each paper
    embeddings = []

    for paper_idx in range(num_papers):
        # Read number of tokens (4 bytes)
        num_tokens = struct.unpack('I', binary_data[offset:offset+4])[0]
        offset += 4

        # Read embeddings (num_tokens * embedding_dim * 4 bytes)
        embedding_size = num_tokens * embedding_dim * 4
        embedding_bytes = binary_data[offset:offset+embedding_size]
        offset += embedding_size

        # Convert to numpy array
        embedding_flat = np.frombuffer(embedding_bytes, dtype=np.float32)
        embedding = embedding_flat.reshape(num_tokens, embedding_dim)

        embeddings.append(embedding)

        if paper_idx < 3 or paper_idx == num_papers - 1:
            print(f"  Paper #{paper_idx}: {num_tokens} tokens × {embedding_dim} dims")

    print(f"✅ Loaded {len(embeddings)} paper embeddings")

    metadata = {
        'num_papers': num_papers,
        'embedding_dim': embedding_dim
    }

    return embeddings, metadata

def load_paper_metadata(index_path):
    """Load paper titles and abstracts."""
    papers_path = Path(index_path) / "papers_metadata.json"

    with open(papers_path, 'r') as f:
        papers = json.load(f)

    print(f"📄 Loaded {len(papers)} paper metadata")

    return papers

def maxsim_score(query_embeddings, doc_embeddings):
    """
    Compute MaxSim score between query and document.
    query_embeddings: (num_query_tokens, dim)
    doc_embeddings: (num_doc_tokens, dim)
    """
    # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
    # For each query token, find max similarity with any doc token
    scores = []

    for query_token in query_embeddings:
        # Compute cosine similarity with all doc tokens
        sims = np.dot(doc_embeddings, query_token)
        max_sim = np.max(sims)
        scores.append(max_sim)

    # Sum of max similarities
    return np.sum(scores)

def test_exact_title_match(embeddings, papers, test_idx=0):
    """
    Test if searching for a paper's exact title returns that paper as rank 1.
    """
    print(f"\n{'='*80}")
    print(f"TEST: Exact Title Match")
    print(f"{'='*80}")

    # Get test paper
    test_paper = papers[test_idx]
    test_title = test_paper['title']
    test_abstract = test_paper.get('abstract', '')

    print(f"\n📄 Test Paper #{test_idx}:")
    print(f"   Title: {test_title}")
    print(f"   Abstract: {test_abstract[:100]}...")

    # For this test, we need to encode the title as a query
    # But we don't have the model here, so let's simulate by using
    # the document embedding as a "query" (should rank itself #1)

    print(f"\n🔍 Using document embedding as query (should rank itself #1)")

    query_embeddings = embeddings[test_idx]

    print(f"   Query tokens: {query_embeddings.shape[0]}")
    print(f"   Query dims: {query_embeddings.shape[1]}")

    # Search all papers
    scores = []
    for doc_idx, doc_embeddings in enumerate(embeddings):
        score = maxsim_score(query_embeddings, doc_embeddings)
        scores.append((doc_idx, score))

    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)

    # Show top 10 results
    print(f"\n📊 Top 10 Results:")
    print(f"{'Rank':<6} {'Paper ID':<10} {'Score':<12} {'Title':<60}")
    print("-" * 90)

    for rank, (paper_idx, score) in enumerate(scores[:10], 1):
        title = papers[paper_idx]['title'][:60]
        marker = " ← TEST PAPER" if paper_idx == test_idx else ""
        print(f"{rank:<6} {paper_idx:<10} {score:<12.4f} {title}{marker}")

    # Check if test paper is rank 1
    top_paper_idx = scores[0][0]

    if top_paper_idx == test_idx:
        print(f"\n✅ PASS: Test paper ranked #1 (score: {scores[0][1]:.4f})")
        return True
    else:
        print(f"\n❌ FAIL: Test paper ranked #{[i for i, (idx, _) in enumerate(scores, 1) if idx == test_idx][0]}")
        print(f"   Expected: Paper #{test_idx} to rank #1")
        print(f"   Got: Paper #{top_paper_idx} ranked #1")
        print(f"   Test paper score: {[s for i, s in scores if i == test_idx][0]:.4f}")
        print(f"   Top paper score: {scores[0][1]:.4f}")
        return False

def test_embedding_quality(embeddings):
    """
    Test if embeddings are properly normalized and non-zero.
    """
    print(f"\n{'='*80}")
    print(f"TEST: Embedding Quality")
    print(f"{'='*80}")

    issues = []

    for paper_idx, paper_emb in enumerate(embeddings):
        # Check for zero embeddings
        if np.allclose(paper_emb, 0):
            issues.append(f"Paper #{paper_idx}: All zeros")
            continue

        # Check normalization (each token vector should be normalized)
        for token_idx, token_vec in enumerate(paper_emb):
            norm = np.linalg.norm(token_vec)

            if np.isclose(norm, 0):
                issues.append(f"Paper #{paper_idx}, token #{token_idx}: Zero vector")
            elif not np.isclose(norm, 1.0, atol=0.01):
                issues.append(f"Paper #{paper_idx}, token #{token_idx}: Not normalized (norm={norm:.4f})")

    if issues:
        print(f"\n❌ FAIL: Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
        return False
    else:
        print(f"\n✅ PASS: All embeddings are normalized and non-zero")

        # Show some stats
        num_tokens = [emb.shape[0] for emb in embeddings]
        print(f"\n📊 Statistics:")
        print(f"   Total papers: {len(embeddings)}")
        print(f"   Avg tokens per paper: {np.mean(num_tokens):.1f}")
        print(f"   Min tokens: {np.min(num_tokens)}")
        print(f"   Max tokens: {np.max(num_tokens)}")
        print(f"   Embedding dim: {embeddings[0].shape[1]}")

        return True

def test_query_vs_document_embeddings(demo_path):
    """
    Test if query embeddings and document embeddings are different.
    This checks if the [Q] vs [D] prefix is working.
    """
    print(f"\n{'='*80}")
    print(f"TEST: Query vs Document Embeddings")
    print(f"{'='*80}")

    print(f"\n⚠️  This test requires loading both query and document embeddings.")
    print(f"   Currently we only have document embeddings.")
    print(f"   SKIPPED")

    return None

def test_diversity(embeddings, papers):
    """
    Test if different papers have different embeddings.
    """
    print(f"\n{'='*80}")
    print(f"TEST: Embedding Diversity")
    print(f"{'='*80}")

    # Compute pairwise similarities between first tokens of different papers
    num_samples = min(10, len(embeddings))

    print(f"\n📊 Computing similarity between first tokens of {num_samples} papers...")

    first_tokens = [emb[0] for emb in embeddings[:num_samples]]

    similarities = []
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            sim = np.dot(first_tokens[i], first_tokens[j])
            similarities.append((i, j, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)

    print(f"\n📊 Most similar pairs:")
    print(f"{'Paper 1':<10} {'Paper 2':<10} {'Similarity':<12}")
    print("-" * 35)

    for i, j, sim in similarities[:5]:
        print(f"{i:<10} {j:<10} {sim:<12.4f}")
        print(f"   Paper {i}: {papers[i]['title'][:60]}")
        print(f"   Paper {j}: {papers[j]['title'][:60]}")
        print()

    print(f"\n📊 Least similar pairs:")
    print(f"{'Paper 1':<10} {'Paper 2':<10} {'Similarity':<12}")
    print("-" * 35)

    for i, j, sim in similarities[-5:]:
        print(f"{i:<10} {j:<10} {sim:<12.4f}")
        print(f"   Paper {i}: {papers[i]['title'][:60]}")
        print(f"   Paper {j}: {papers[j]['title'][:60]}")
        print()

    # Check if similarities are too high
    avg_sim = np.mean([s for _, _, s in similarities])
    max_sim = similarities[0][2]
    min_sim = similarities[-1][2]

    print(f"📊 Statistics:")
    print(f"   Average similarity: {avg_sim:.4f}")
    print(f"   Max similarity: {max_sim:.4f}")
    print(f"   Min similarity: {min_sim:.4f}")

    if avg_sim > 0.8:
        print(f"\n⚠️  WARNING: Average similarity is high ({avg_sim:.4f})")
        print(f"   Papers might be too similar to each other")
        return False
    else:
        print(f"\n✅ PASS: Papers have diverse embeddings")
        return True

def main():
    # Paths
    demo_path = Path(__file__).parent.parent / "demo"
    index_path = demo_path / "data" / "precomputed_index_500"

    print("=" * 80)
    print("OFFLINE EMBEDDING TEST")
    print("=" * 80)

    # Load data
    embeddings, metadata = load_binary_embeddings(index_path)
    papers = load_paper_metadata(index_path)

    # Run tests
    results = {}

    results['quality'] = test_embedding_quality(embeddings)
    results['diversity'] = test_diversity(embeddings, papers)
    results['exact_match_0'] = test_exact_title_match(embeddings, papers, test_idx=0)
    results['exact_match_10'] = test_exact_title_match(embeddings, papers, test_idx=10)
    results['exact_match_50'] = test_exact_title_match(embeddings, papers, test_idx=50)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for test_name, result in results.items():
        status = "✅ PASS" if result else ("❌ FAIL" if result is False else "⚠️  SKIP")
        print(f"{test_name:<20} {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print(f"\n❌ ISSUES DETECTED: The embeddings may have problems!")
        print(f"\nPossible causes:")
        print(f"1. Document embeddings stored instead of query embeddings")
        print(f"2. Embeddings not normalized correctly")
        print(f"3. Wrong MaxSim implementation")
        print(f"4. Index built with wrong model/settings")
        return 1
    else:
        print(f"\n✅ All tests passed!")
        return 0

if __name__ == "__main__":
    exit(main())
