#!/usr/bin/env python3
"""
Fix WASM table size limits to allow growth
Modifies the funcref table from max=64 to max=256
"""

import sys
import struct

def fix_wasm_table(input_file, output_file):
    with open(input_file, 'rb') as f:
        data = bytearray(f.read())

    # WASM magic number: \0asm
    if data[:4] != b'\x00asm':
        print("Error: Not a valid WASM file")
        return False

    # WASM version
    version = struct.unpack('<I', data[4:8])[0]
    print(f"WASM version: {version}")

    pos = 8
    while pos < len(data):
        section_id = data[pos]
        pos += 1

        # Read LEB128 section size
        section_size, bytes_read = read_leb128(data, pos)
        pos += bytes_read
        section_start = pos
        section_end = pos + section_size

        # Section 4 is the Table section
        if section_id == 4:
            print(f"Found Table section at offset {section_start}, size {section_size}")

            # Read number of tables
            num_tables, bytes_read = read_leb128(data, pos)
            pos += bytes_read
            print(f"Number of tables: {num_tables}")

            for i in range(num_tables):
                table_type = data[pos]
                pos += 1
                print(f"Table {i}: type={table_type} ({'funcref' if table_type == 0x70 else 'externref' if table_type == 0x6f else 'unknown'})")

                # Read limits
                limits_type = data[pos]
                pos += 1

                initial, bytes_read = read_leb128(data, pos)
                initial_pos = pos
                pos += bytes_read

                if limits_type == 0x01:  # Has max
                    maximum, bytes_read = read_leb128(data, pos)
                    max_pos = pos
                    pos += bytes_read

                    print(f"  Initial: {initial}, Max: {maximum}, Limits type: {limits_type}")

                    # If this is the funcref table with max=64, change it to 256
                    if table_type == 0x70 and maximum == 64:
                        print(f"  → Changing max from 64 to 256")
                        # Write new LEB128 value for 256
                        new_max = encode_leb128(256)
                        # Replace the old max value
                        data[max_pos:max_pos+bytes_read] = new_max
                        print(f"  ✅ Fixed funcref table max size")
                else:
                    print(f"  Initial: {initial}, No max, Limits type: {limits_type}")

        pos = section_end

    # Write modified WASM
    with open(output_file, 'wb') as f:
        f.write(data)

    print(f"\n✅ Modified WASM written to {output_file}")
    return True

def read_leb128(data, pos):
    """Read unsigned LEB128"""
    result = 0
    shift = 0
    bytes_read = 0
    while True:
        byte = data[pos + bytes_read]
        bytes_read += 1
        result |= (byte & 0x7f) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result, bytes_read

def encode_leb128(value):
    """Encode unsigned LEB128"""
    result = bytearray()
    while True:
        byte = value & 0x7f
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)

if __name__ == '__main__':
    files_to_fix = [
        'docs/pkg/fast_plaid_rust_bg.wasm',
        'docs/pkg/pylate_rs_bg.wasm',
    ]

    success = True
    for wasm_file in files_to_fix:
        print(f"\n{'='*60}")
        print(f"Processing: {wasm_file}")
        print('='*60)
        if fix_wasm_table(wasm_file, wasm_file):
            print(f"✅ {wasm_file} fixed successfully")
        else:
            print(f"❌ Failed to fix {wasm_file}")
            success = False

    if success:
        print("\n" + "="*60)
        print("✅ All WASM files fixed successfully")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some WASM files failed to fix")
        print("="*60)
        sys.exit(1)
