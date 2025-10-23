#!/usr/bin/env python3
"""
Build .fastplaid binary index for WASM loading

This script:
1. Loads embeddings.bin
2. Saves them in WASM-compatible .fastplaid format
3. Output can be loaded instantly by WASM load_index()
"""

import struct
import sys
from pathlib import Path

def read_embeddings(embeddings_path):
    """Read embeddings.bin file"""
    print(f"📥 Reading {embeddings_path}...")

    with open(embeddings_path, 'rb') as f:
        # Read header
        num_papers = struct.unpack('<I', f.read(4))[0]
        embedding_dim = struct.unpack('<I', f.read(4))[0]

        print(f"   Papers: {num_papers}, Dim: {embedding_dim}")

        # Read all embeddings
        embeddings = []
        doc_info = []

        for i in range(num_papers):
            num_tokens = struct.unpack('<I', f.read(4))[0]

            # Read embedding data
            emb_size = num_tokens * embedding_dim
            emb_data = struct.unpack(f'<{emb_size}f', f.read(emb_size * 4))

            embeddings.append((num_tokens, emb_data))
            doc_info.append((i, num_tokens))

        print(f"✅ Loaded {num_papers} papers")

        return embeddings, doc_info, embedding_dim

def build_fastplaid_index(embeddings, doc_info, embedding_dim, output_path):
    """
    Build .fastplaid index in WASM-compatible format

    Format (compatible with WASM load_index()):
    - Magic: "FPQZ" (4 bytes)
    - Version: u32
    - embedding_dim: u32
    - num_docs: u32
    - For each doc:
      - doc_id: u64
      - num_tokens: u32
      - embeddings: f32[num_tokens * embedding_dim]
    """
    print(f"💾 Building .fastplaid index...")

    num_docs = len(embeddings)

    with open(output_path, 'wb') as f:
        # Write header
        f.write(b'FPQZ')  # Magic number
        f.write(struct.pack('<I', 1))  # Version
        f.write(struct.pack('<I', embedding_dim))
        f.write(struct.pack('<I', num_docs))

        print(f"   Writing {num_docs} documents...")

        # Write each document
        for (doc_id, num_tokens), (nt, emb_data) in zip(doc_info, embeddings):
            assert num_tokens == nt, f"Token count mismatch for doc {doc_id}"

            # Write doc metadata
            f.write(struct.pack('<Q', doc_id))  # doc_id as u64
            f.write(struct.pack('<I', num_tokens))

            # Write embeddings
            f.write(struct.pack(f'<{len(emb_data)}f', *emb_data))

    file_size = Path(output_path).stat().st_size
    size_mb = file_size / 1_000_000

    print(f"✅ Saved {output_path} ({size_mb:.2f} MB)")

    return file_size

def main():
    if len(sys.argv) < 3:
        print("Usage: python build_fastplaid_wasm_index.py <embeddings.bin> <output.fastplaid>")
        print()
        print("Example:")
        print("  python scripts/build_fastplaid_wasm_index.py \\")
        print("    demo/data/fastplaid_4bit/embeddings.bin \\")
        print("    demo/data/index.fastplaid")
        sys.exit(1)

    embeddings_path = sys.argv[1]
    output_path = sys.argv[2]

    print("🚀 Building .fastplaid index for WASM")
    print("=" * 70)
    print(f"Input: {embeddings_path}")
    print(f"Output: {output_path}")
    print()

    # Read embeddings
    embeddings, doc_info, embedding_dim = read_embeddings(embeddings_path)

    # Build index
    file_size = build_fastplaid_index(embeddings, doc_info, embedding_dim, output_path)

    print()
    print("=" * 70)
    print("✅ .fastplaid index built successfully!")
    print()
    print("🚀 Usage in browser:")
    print("   const response = await fetch('./data/index.fastplaid');")
    print("   const indexBytes = await response.arrayBuffer();")
    print("   fastPlaidWasm.load_index(new Uint8Array(indexBytes));")
    print("   // Instant loading!")

if __name__ == '__main__':
    main()
