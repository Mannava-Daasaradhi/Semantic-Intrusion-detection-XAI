#!/usr/bin/env python3
"""
IoT2023 IP Extraction Script
Extracts src_ip and dst_ip from every PCAP in CIC-IoT2023 using tshark.
Reads ONLY IP headers (not payloads) — fast even on 549GB of PCAPs.
Output: one CSV per attack folder saved alongside the flow CSVs.
"""

import subprocess
import csv
import os
import sys
import time
from pathlib import Path

PCAP_ROOT = Path("/media/mannava/D/S-XG-NID/data/01_raw/CIC-IoT2023/PCAP")
CSV_ROOT  = Path("/media/mannava/D/S-XG-NID/data/01_raw/CIC-IoT2023/CSV")
OUT_ROOT  = Path("/media/mannava/D/S-XG-NID/data/01_raw/CIC-IoT2023/IP_Mappings")

OUT_ROOT.mkdir(parents=True, exist_ok=True)


def extract_ips_from_pcap(pcap_path: Path) -> list[tuple[str, str]]:
    """
    Uses tshark to extract (src_ip, dst_ip) for every packet in a PCAP.
    Only reads IP layer headers — does not decode payload at all.
    Returns a list of (src_ip, dst_ip) tuples, one per packet.
    """
    cmd = [
        "tshark",
        "-r", str(pcap_path),
        "-T", "fields",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-E", "separator=,",
        "-E", "empty=NONE",
        # Only process IP packets, skip ARP/other layer 2 frames
        "-Y", "ip",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max per file
        )
        pairs = []
        for line in result.stdout.splitlines():
            parts = line.strip().split(",")
            if len(parts) == 2:
                src, dst = parts[0].strip(), parts[1].strip()
                if src and dst and src != "NONE" and dst != "NONE":
                    pairs.append((src, dst))
        return pairs
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT on {pcap_path.name} — skipping")
        return []
    except Exception as e:
        print(f"  ERROR on {pcap_path.name}: {e}")
        return []


def get_flow_count(csv_folder: Path) -> int:
    """Count total rows across all CSVs in a folder (excluding header)."""
    total = 0
    for csv_file in sorted(csv_folder.glob("*.csv")):
        try:
            with open(csv_file, "r") as f:
                total += sum(1 for _ in f) - 1  # subtract header
        except Exception:
            pass
    return total


def process_attack_folder(attack_name: str):
    """
    Process one attack folder:
    - Find all PCAPs in PCAP/attack_name/
    - Extract IPs from each PCAP
    - Save combined IP mapping CSV to IP_Mappings/attack_name_ip_map.csv
    """
    pcap_folder = PCAP_ROOT / attack_name
    csv_folder  = CSV_ROOT  / attack_name
    out_file    = OUT_ROOT  / f"{attack_name}_ip_map.csv"

    if not pcap_folder.exists():
        print(f"  [SKIP] No PCAP folder: {attack_name}")
        return

    if not csv_folder.exists():
        print(f"  [SKIP] No CSV folder: {attack_name}")
        return

    if out_file.exists():
        print(f"  [DONE] Already extracted: {attack_name}")
        return

    pcap_files = sorted(pcap_folder.glob("*.pcap")) + sorted(pcap_folder.glob("*.pcapng"))
    if not pcap_files:
        print(f"  [SKIP] No PCAP files in: {attack_name}")
        return

    flow_count = get_flow_count(csv_folder)
    print(f"\n[{attack_name}]")
    print(f"  PCAPs: {len(pcap_files)}  |  CSV flows: {flow_count}")

    all_pairs = []
    for i, pcap_file in enumerate(pcap_files):
        size_gb = pcap_file.stat().st_size / (1024**3)
        print(f"  Processing [{i+1}/{len(pcap_files)}]: {pcap_file.name} ({size_gb:.1f} GB)...",
              end="", flush=True)
        t0 = time.time()
        pairs = extract_ips_from_pcap(pcap_file)
        elapsed = time.time() - t0
        print(f" {len(pairs):,} packets in {elapsed:.0f}s")
        all_pairs.extend(pairs)

    print(f"  Total packets extracted: {len(all_pairs):,}")

    # Save the IP mapping
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["packet_idx", "src_ip", "dst_ip"])
        for idx, (src, dst) in enumerate(all_pairs):
            writer.writerow([idx, src, dst])

    print(f"  Saved: {out_file.name} ({out_file.stat().st_size / 1024:.0f} KB)")


def main():
    # Get all attack folders that have both PCAP and CSV
    attack_folders = sorted([
        d.name for d in PCAP_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print("=" * 60)
    print("S-XG-NID IoT2023 IP Extraction")
    print(f"Found {len(attack_folders)} attack folders")
    print(f"Output directory: {OUT_ROOT}")
    print("=" * 60)

    # Allow filtering via command line argument
    # e.g. python extract_iot_ips.py Benign_Final
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target in attack_folders:
            print(f"Running single folder: {target}")
            process_attack_folder(target)
        else:
            print(f"Folder '{target}' not found. Available: {attack_folders}")
        return

    # Process all folders
    t_start = time.time()
    for attack_name in attack_folders:
        process_attack_folder(attack_name)

    elapsed = (time.time() - t_start) / 60
    print("\n" + "=" * 60)
    print(f"Complete. Total time: {elapsed:.1f} minutes")
    print(f"IP mappings saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
