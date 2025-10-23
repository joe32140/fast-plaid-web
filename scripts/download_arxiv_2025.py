#!/usr/bin/env python3
"""
Download real papers from ArXiv 2025 using ArXiv API.
Only fetches title and abstract.
"""

import json
import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm import tqdm

def query_arxiv(category='cs.AI', max_results=10000, year=2025):
    """
    Query ArXiv API for recent papers.

    Popular categories:
    - cs.AI: Artificial Intelligence
    - cs.CL: Computation and Language (NLP)
    - cs.LG: Machine Learning
    - cs.CV: Computer Vision
    - cs.IR: Information Retrieval
    """

    print(f"📥 Downloading {max_results} papers from ArXiv...")
    print(f"   Category: {category}")
    print(f"   Year: {year}")
    print()

    papers = []
    batch_size = 100  # ArXiv API limit per request

    for start in tqdm(range(0, max_results, batch_size), desc="Fetching batches"):
        # Construct query - just use category, get most recent papers
        query = f'cat:{category}'

        params = {
            'search_query': query,
            'start': start,
            'max_results': min(batch_size, max_results - start),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode(params)

        try:
            # Fetch from ArXiv API
            with urllib.request.urlopen(url) as response:
                data = response.read().decode('utf-8')

            # Parse XML
            root = ET.fromstring(data)

            # Extract namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            entries = root.findall('atom:entry', ns)

            if not entries:
                print(f"\n⚠️ No more papers found at offset {start}")
                break

            for entry in entries:
                # Extract ID
                arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]

                # Extract title
                title = entry.find('atom:title', ns).text
                title = ' '.join(title.split())  # Clean whitespace

                # Extract abstract
                abstract = entry.find('atom:summary', ns).text
                abstract = ' '.join(abstract.split())  # Clean whitespace

                # Extract published date
                published = entry.find('atom:published', ns).text

                # Extract categories
                categories = [cat.attrib['term'] for cat in entry.findall('atom:category', ns)]

                papers.append({
                    'id': arxiv_id,
                    'title': title,
                    'abstract': abstract,
                    'published': published,
                    'categories': categories,
                    'text': f"{title} {abstract}"
                })

            # Be nice to ArXiv API - rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"\n❌ Error fetching batch at {start}: {e}")
            break

    return papers

def download_multi_category(categories, papers_per_category=2000, year=2025):
    """
    Download papers from multiple categories.
    """
    all_papers = []

    for category in categories:
        print(f"\n📚 Fetching from category: {category}")
        papers = query_arxiv(category, papers_per_category, year)

        # Add category prefix to IDs to avoid duplicates
        for paper in papers:
            paper['id'] = f"{category.replace('.', '_')}_{paper['id']}"

        all_papers.extend(papers)
        print(f"   ✅ Got {len(papers)} papers from {category}")

    return all_papers

def save_papers(papers, output_file):
    """Save papers to JSON file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_file) / 1024 / 1024
    print(f"\n✅ Saved {len(papers)} papers to {output_file}")
    print(f"📊 File size: {file_size:.2f} MB")

    return output_file

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download 2025 papers from ArXiv')
    parser.add_argument('--category', type=str, default='cs.CL',
                        help='ArXiv category (default: cs.CL for NLP papers)')
    parser.add_argument('--multi-category', action='store_true',
                        help='Download from multiple categories')
    parser.add_argument('--num-papers', type=int, default=10000,
                        help='Total number of papers to download (default: 10000)')
    parser.add_argument('--year', type=int, default=2025,
                        help='Year to download papers from (default: 2025)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')

    args = parser.parse_args()

    if args.output is None:
        args.output = f'../data/papers_arxiv_{args.year}_{args.num_papers}.json'

    print("🚀 ArXiv Paper Downloader")
    print("=" * 50)
    print(f"Target: {args.num_papers} papers from {args.year}")
    print()

    if args.multi_category:
        # Download from multiple categories for diversity
        categories = ['cs.CL', 'cs.AI', 'cs.LG', 'cs.IR', 'cs.CV']
        papers_per_cat = args.num_papers // len(categories)

        print(f"📚 Downloading from {len(categories)} categories:")
        print(f"   Categories: {', '.join(categories)}")
        print(f"   Papers per category: {papers_per_cat}")
        print()

        papers = download_multi_category(categories, papers_per_cat, args.year)
    else:
        # Download from single category
        papers = query_arxiv(args.category, args.num_papers, args.year)

    # Save papers
    output_path = save_papers(papers, args.output)

    # Show sample
    print()
    print("📄 Sample papers:")
    for i, paper in enumerate(papers[:3]):
        print(f"\n{i+1}. {paper['title']}")
        print(f"   ID: {paper['id']}")
        print(f"   Published: {paper['published']}")
        print(f"   Abstract: {paper['abstract'][:120]}...")

    print()
    print("✅ Download complete!")
    print(f"📁 Output: {output_path}")
    print()
    print("🔧 Next steps:")
    print(f"   python build_offline_index.py --papers {output_path}")
