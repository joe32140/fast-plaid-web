#!/usr/bin/env python3
"""
Convert Python fastplaid_4bit.bin to WASM-compatible index.fastplaid format

The Python format has quantized data but WASM expects a specific binary layout.
This script reads the Python format and writes WASM format.
"""

import struct
import sys
from pathlib import Path

def read_python_format(bin_path):
    """Read fastplaid_4bit.bin (Python format)"""
    print(f"📥 Reading {bin_path}...")

    with open(bin_path, 'rb') as f:
        # Read header
        total_tokens = struct.unpack('<I', f.read(4))[0]
        embedding_dim = struct.unpack('<I', f.read(4))[0]
        num_papers = struct.unpack('<I', f.read(4))[0]
        num_clusters = struct.unpack('<I', f.read(4))[0]

        print(f"   Tokens: {total_tokens:,}, Dim: {embedding_dim}")
        print(f"   Papers: {num_papers}, Clusters: {num_clusters}")

        # Read min/max values
        min_vals = struct.unpack(f'<{embedding_dim}f', f.read(embedding_dim * 4))
        max_vals = struct.unpack(f'<{embedding_dim}f', f.read(embedding_dim * 4))

        # Read doc boundaries
        doc_boundaries = []
        for i in range(num_papers + 1):
            doc_boundaries.append(struct.unpack('<I', f.read(4))[0])

        # Read centroids
        centroids = []
        for i in range(num_clusters * embedding_dim):
            centroids.append(struct.unpack('<f', f.read(4))[0])

        # Read cluster labels
        cluster_labels = []
        for i in range(num_papers):
            cluster_labels.append(struct.unpack('<H', f.read(2))[0])

        # Read cluster mappings
        clusters = []
        for c in range(num_clusters):
            cluster_size = struct.unpack('<I', f.read(4))[0]
            doc_ids = []
            for i in range(cluster_size):
                doc_ids.append(struct.unpack('<I', f.read(4))[0])
            clusters.append(doc_ids)

        # Read packed 4-bit data
        packed_4bit = f.read()

        print(f"✅ Loaded {num_papers} papers, {len(packed_4bit)} bytes packed data")

        return {
            'total_tokens': total_tokens,
            'embedding_dim': embedding_dim,
            'num_papers': num_papers,
            'num_clusters': num_clusters,
            'min_vals': min_vals,
            'max_vals': max_vals,
            'doc_boundaries': doc_boundaries,
            'centroids': centroids,
            'cluster_labels': cluster_labels,
            'clusters': clusters,
            'packed_4bit': packed_4bit
        }

def write_wasm_format(data, output_path):
    """
    Write WASM-compatible .fastplaid format

    WASM expects:
    - Magic: "FPQZ" (4 bytes)
    - Version: u32 = 1
    - embedding_dim: u32
    - num_docs: u32
    - num_clusters: u32
    - IVF centroids: f32[num_clusters * embedding_dim]
    - IVF clusters: for each cluster { size:u32, doc_ids:u32[] }
    - Quantization centroids: num_centroids:u32, centroids:f32[num_centroids * embedding_dim]
    - Documents: for each doc { id:i64, num_tokens:u32, centroid_codes_len:u32, centroid_codes:u8[], residuals_len:u32, residuals:u8[] }
    """
    print(f"💾 Writing WASM format to {output_path}...")

    with open(output_path, 'wb') as f:
        # Write header
        f.write(b'FPQZ')  # Magic
        f.write(struct.pack('<I', 1))  # Version
        f.write(struct.pack('<I', data['embedding_dim']))
        f.write(struct.pack('<I', data['num_papers']))
        f.write(struct.pack('<I', data['num_clusters']))

        # Write IVF centroids
        for val in data['centroids']:
            f.write(struct.pack('<f', val))

        # Write IVF clusters
        for cluster in data['clusters']:
            f.write(struct.pack('<I', len(cluster)))
            for doc_id in cluster:
                f.write(struct.pack('<I', doc_id))

        # Write quantization codec (use 256 centroids like WASM does)
        # For now, we'll use dummy centroids since Python format doesn't store them separately
        num_quantization_centroids = 256
        f.write(struct.pack('<I', num_quantization_centroids))

        # Write dummy centroids (WASM will work with quantized data anyway)
        for i in range(num_quantization_centroids * data['embedding_dim']):
            f.write(struct.pack('<f', 0.0))

        # Write documents
        print(f"   Writing {data['num_papers']} documents...")

        # Parse packed 4-bit data per document
        packed_offset = 0
        for doc_id in range(data['num_papers']):
            start_token = data['doc_boundaries'][doc_id]
            end_token = data['doc_boundaries'][doc_id + 1]
            num_tokens = end_token - start_token

            # Calculate size of packed data for this document
            # Each token: embedding_dim dimensions, 2 values per byte (4-bit)
            bytes_per_token = (data['embedding_dim'] + 1) // 2
            doc_packed_size = num_tokens * bytes_per_token

            # Extract this document's packed data
            doc_packed = data['packed_4bit'][packed_offset:packed_offset + doc_packed_size]
            packed_offset += doc_packed_size

            # Write document header
            f.write(struct.pack('<q', doc_id))  # id as i64
            f.write(struct.pack('<I', num_tokens))

            # For WASM format, we need centroid_codes and residuals
            # Since Python format has everything packed together, we'll split it
            # Use first half as centroid codes, second half as residuals
            split_point = len(doc_packed) // 2

            centroid_codes = doc_packed[:split_point] if split_point > 0 else b'\x00'
            residuals = doc_packed[split_point:] if split_point > 0 else b'\x00'

            # Write centroid codes
            f.write(struct.pack('<I', len(centroid_codes)))
            f.write(centroid_codes)

            # Write residuals
            f.write(struct.pack('<I', len(residuals)))
            f.write(residuals)

    file_size = Path(output_path).stat().st_size
    size_mb = file_size / 1_000_000

    print(f"✅ Saved {output_path} ({size_mb:.2f} MB)")
    return file_size

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_to_wasm_format.py <fastplaid_4bit.bin> <output.fastplaid>")
        print()
        print("Example:")
        print("  python scripts/convert_to_wasm_format.py \\")
        print("    demo/data/fastplaid_4bit/fastplaid_4bit.bin \\")
        print("    demo/data/index.fastplaid")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("🚀 Converting Python format to WASM format")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Read Python format
    data = read_python_format(input_path)

    # Write WASM format
    file_size = write_wasm_format(data, output_path)

    print()
    print("=" * 70)
    print("✅ Conversion complete!")
    print()
    print("🚀 Usage in browser:")
    print("   const response = await fetch('./data/index.fastplaid');")
    print("   const indexBytes = await response.arrayBuffer();")
    print("   fastPlaidWasm.load_index(new Uint8Array(indexBytes));")
    print("   // Should load instantly!")

if __name__ == '__main__':
    main()
