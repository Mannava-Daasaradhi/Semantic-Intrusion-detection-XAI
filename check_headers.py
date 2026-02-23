import pandas as pd
import glob
import os

# Define paths
paths = [
    "/media/mannava/D/S-XG-NID/data/01_raw/CIC-IoT2023",
    "/media/mannava/D/S-XG-NID/data/01_raw/CIC-IDS2017",
    "/media/mannava/D/S-XG-NID/data/01_raw/UNSW-NB15"
]

for p in paths:
    print(f"\n--- Checking {os.path.basename(p)} ---")
    files = glob.glob(os.path.join(p, '**', '*.csv'), recursive=True)
    if files:
        # Read just the header
        try:
            df = pd.read_csv(files[0], nrows=1)
            cols = df.columns.tolist()
            # Check for IP-like columns
            ip_cols = [c for c in cols if 'IP' in c or 'ip' in c or 'Src' in c or 'Dst' in c]
            print(f"File: {os.path.basename(files[0])}")
            print(f"Found IP Columns: {ip_cols}")
        except Exception as e:
            print(f"Error reading: {e}")
    else:
        print("No CSVs found.")
