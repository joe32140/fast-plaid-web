#!/usr/bin/env python3
"""
Download NeurIPS 2025 accepted papers.
Sources:
1. NeurIPS 2025 OpenReview (if available)
2. Papers with Code NeurIPS 2025 page
3. Official NeurIPS website
"""

import json
import os
import re
import urllib.request
from bs4 import BeautifulSoup
from tqdm import tqdm

def download_from_openreview(year=2025):
    """
    Download NeurIPS papers from OpenReview.
    OpenReview URL: https://openreview.net/group?id=NeurIPS.cc/{year}/Conference
    """
    print(f"📥 Downloading NeurIPS {year} papers from OpenReview...")

    url = f"https://api.openreview.net/notes?invitation=NeurIPS.cc/{year}/Conference/-/Submission&details=directReplies"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())

        papers = []

        print(f"📄 Processing {len(data.get('notes', []))} papers...")

        for note in tqdm(data.get('notes', []), desc="Processing"):
            # Check if paper was accepted
            if 'decision' in note.get('content', {}):
                decision = note['content']['decision']
                if 'Accept' not in decision:
                    continue

            paper_id = note['id']
            title = note['content'].get('title', '')
            abstract = note['content'].get('abstract', '')

            if not title or not abstract:
                continue

            papers.append({
                'id': f'neurips2025_{paper_id}',
                'title': title,
                'abstract': abstract,
                'openreview_id': paper_id,
                'text': f"{title} {abstract}"
            })

        return papers

    except Exception as e:
        print(f"❌ Error accessing OpenReview: {e}")
        return None

def download_from_papers_with_code():
    """
    Download from Papers with Code NeurIPS 2025 page.
    URL: https://paperswithcode.com/conference/neurips-2025-12
    """
    print(f"📥 Downloading NeurIPS 2025 papers from Papers with Code...")

    url = "https://paperswithcode.com/conference/neurips-2025-12"

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')

        soup = BeautifulSoup(html, 'html.parser')

        # Find paper entries
        paper_items = soup.find_all('div', class_='paper-card')

        papers = []

        print(f"📄 Processing {len(paper_items)} papers...")

        for i, item in enumerate(tqdm(paper_items, desc="Processing")):
            title_elem = item.find('h1')
            title = title_elem.text.strip() if title_elem else f"Paper {i}"

            abstract_elem = item.find('p', class_='paper-abstract')
            abstract = abstract_elem.text.strip() if abstract_elem else ""

            if not abstract:
                continue

            papers.append({
                'id': f'neurips2025_{i:04d}',
                'title': title,
                'abstract': abstract,
                'text': f"{title} {abstract}"
            })

        return papers

    except Exception as e:
        print(f"❌ Error accessing Papers with Code: {e}")
        return None

def download_from_neurips_website():
    """
    Download from official NeurIPS website.
    URL: https://neurips.cc/virtual/2025/papers.html
    """
    print(f"📥 Downloading NeurIPS 2025 papers from official website...")

    url = "https://neurips.cc/virtual/2025/papers.html"

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')

        soup = BeautifulSoup(html, 'html.parser')

        # Parse paper list (structure may vary)
        papers = []

        # Try to find paper entries
        # Structure varies - may need adjustment based on actual HTML
        paper_divs = soup.find_all('div', class_='paper')

        print(f"📄 Processing {len(paper_divs)} papers...")

        for i, div in enumerate(tqdm(paper_divs, desc="Processing")):
            title = div.find('span', class_='title')
            abstract = div.find('span', class_='abstract')

            if title and abstract:
                papers.append({
                    'id': f'neurips2025_{i:04d}',
                    'title': title.text.strip(),
                    'abstract': abstract.text.strip(),
                    'text': f"{title.text.strip()} {abstract.text.strip()}"
                })

        return papers

    except Exception as e:
        print(f"❌ Error accessing NeurIPS website: {e}")
        return None

def download_neurips_2025():
    """
    Try multiple sources to download NeurIPS 2025 papers.
    """
    print("🔍 Searching for NeurIPS 2025 papers from multiple sources...")
    print()

    # Try OpenReview first (most reliable)
    papers = download_from_openreview(2025)

    if papers and len(papers) > 0:
        print(f"✅ Found {len(papers)} papers from OpenReview")
        return papers

    print("⚠️ OpenReview not available, trying Papers with Code...")
    papers = download_from_papers_with_code()

    if papers and len(papers) > 0:
        print(f"✅ Found {len(papers)} papers from Papers with Code")
        return papers

    print("⚠️ Papers with Code not available, trying NeurIPS website...")
    papers = download_from_neurips_website()

    if papers and len(papers) > 0:
        print(f"✅ Found {len(papers)} papers from NeurIPS website")
        return papers

    print("❌ Could not download from any source")
    return None

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

    parser = argparse.ArgumentParser(description='Download NeurIPS 2025 accepted papers')
    parser.add_argument('--output', type=str, default='../data/papers_neurips_2025.json',
                        help='Output file path')

    args = parser.parse_args()

    print("🚀 NeurIPS 2025 Paper Downloader")
    print("=" * 50)
    print()

    # Download papers
    papers = download_neurips_2025()

    if papers is None or len(papers) == 0:
        print("\n❌ Failed to download papers")
        print("\n💡 Alternative:")
        print("   Try downloading recent ArXiv papers instead:")
        print("   python download_arxiv_2025.py --multi-category --num-papers 10000")
        exit(1)

    # Save papers
    output_path = save_papers(papers, args.output)

    # Show samples
    print()
    print("📄 Sample papers:")
    for i, paper in enumerate(papers[:3]):
        print(f"\n{i+1}. {paper['title']}")
        print(f"   ID: {paper['id']}")
        print(f"   Abstract: {paper['abstract'][:120]}...")

    print()
    print("✅ Download complete!")
    print(f"📁 Output: {output_path}")
    print()
    print("🔧 Next steps:")
    print(f"   python build_offline_index.py --papers {output_path}")
