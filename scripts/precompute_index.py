#!/usr/bin/env python3
"""
Pre-compute embeddings and index for 1000 arXiv papers for the GitHub Pages demo.
This generates a pre-computed index that can be loaded directly in the browser.
"""

import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("🔄 Pre-computing index for 1000 arXiv papers...")

    # Load papers
    data_dir = Path(__file__).parent.parent / "data"
    papers_file = data_dir / "papers_arxiv_recent_10k.json"

    print(f"📄 Loading papers from {papers_file}...")
    with open(papers_file) as f:
        all_papers = json.load(f)

    # Take first 1000 papers
    papers = all_papers[:1000]
    print(f"✅ Loaded {len(papers)} papers")

    # Save the 1000 papers subset
    output_papers = data_dir / "papers_1000.json"
    with open(output_papers, 'w') as f:
        json.dump(papers, f, indent=2)
    print(f"💾 Saved 1000 papers to {output_papers}")

    # TODO: Compute embeddings using Node.js + mxbai-integration.js
    # For now, we'll create a Node.js script to do this since mxbai runs in WASM
    print("\n⚠️  Next step: Run the Node.js script to compute embeddings")
    print("    The embeddings must be computed in Node.js using the mxbai WASM module")

    return output_papers

if __name__ == "__main__":
    main()
