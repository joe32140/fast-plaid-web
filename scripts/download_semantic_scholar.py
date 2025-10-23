#!/usr/bin/env python3
"""
Download recent papers from Semantic Scholar API.
Gets papers with titles and abstracts.
"""

import json
import os
import time
import urllib.request
import urllib.parse
from tqdm import tqdm

def download_papers_semantic_scholar(num_papers=5000, fields_of_study=['Computer Science']):
    """
    Download papers from Semantic Scholar bulk API.
    API: https://api.semanticscholar.org/
    """
    print(f"📥 Downloading {num_papers} papers from Semantic Scholar...")
    print(f"   Fields of study: {', '.join(fields_of_study)}")
    print()

    papers = []
    offset = 0
    limit = 100  # API limit per request

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; AcademicResearch/1.0)'
    }

    with tqdm(total=num_papers, desc="Downloading") as pbar:
        while len(papers) < num_papers:
            # Construct query for recent papers
            params = {
                'query': ' OR '.join([f'fieldsOfStudy:{field}' for field in fields_of_study]),
                'offset': offset,
                'limit': min(limit, num_papers - len(papers)),
                'fields': 'paperId,title,abstract,year,venue,authors',
                'sort': 'publicationDate:desc'
            }

            url = 'https://api.semanticscholar.org/graph/v1/paper/search?' + urllib.parse.urlencode(params)

            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode('utf-8'))

                if 'data' not in data or not data['data']:
                    print(f"\n⚠️ No more papers found at offset {offset}")
                    break

                for paper in data['data']:
                    title = paper.get('title', '')
                    abstract = paper.get('abstract', '')

                    # Skip papers without abstract
                    if not title or not abstract:
                        continue

                    papers.append({
                        'id': paper.get('paperId', f'ss_{len(papers)}'),
                        'title': title,
                        'abstract': abstract,
                        'year': paper.get('year'),
                        'venue': paper.get('venue', 'Unknown'),
                        'text': f"{title} {abstract}"
                    })

                    pbar.update(1)

                    if len(papers) >= num_papers:
                        break

                offset += limit

                # Rate limiting - be nice to the API
                time.sleep(1)

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"\n⚠️ Rate limited, waiting 10 seconds...")
                    time.sleep(10)
                else:
                    print(f"\n❌ HTTP Error {e.code}: {e.reason}")
                    break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                break

    return papers

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

    parser = argparse.ArgumentParser(description='Download papers from Semantic Scholar')
    parser.add_argument('--num-papers', type=int, default=5000,
                        help='Number of papers to download (default: 5000)')
    parser.add_argument('--fields', nargs='+', default=['Computer Science', 'Artificial Intelligence'],
                        help='Fields of study to filter by')
    parser.add_argument('--output', type=str, default='../data/papers_semantic_scholar_5k.json',
                        help='Output file path')

    args = parser.parse_args()

    print("🚀 Semantic Scholar Paper Downloader")
    print("=" * 50)
    print(f"Target: {args.num_papers} papers")
    print(f"Fields: {', '.join(args.fields)}")
    print()

    # Download papers
    papers = download_papers_semantic_scholar(args.num_papers, args.fields)

    if not papers:
        print("\n❌ Failed to download papers")
        exit(1)

    # Save papers
    output_path = save_papers(papers, args.output)

    # Show samples
    print()
    print("📄 Sample papers:")
    for i, paper in enumerate(papers[:3]):
        print(f"\n{i+1}. {paper['title']}")
        print(f"   ID: {paper['id']}")
        print(f"   Year: {paper.get('year', 'Unknown')}")
        print(f"   Abstract: {paper['abstract'][:120]}...")

    print()
    print("✅ Download complete!")
    print(f"📁 Output: {output_path}")
    print()
    print("🔧 Next steps:")
    print(f"   python build_offline_index.py --papers {output_path}")
