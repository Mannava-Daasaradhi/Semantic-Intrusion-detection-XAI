import pandas as pd
import glob
import os

# Define paths
paths = [
    # 1. UNSW (Fixing the crash)
    "/media/mannava/D/S-XG-NID/data/01_raw/UNSW-NB15",
    
    # 2. IDS2017 (Targeting the folder that usually has IPs)
    "/media/mannava/D/S-XG-NID/data/01_raw/CIC-IDS2017/CSV/TrafficLabelling",
    
    # 3. IoT2023 (Just double checking)
    "/media/mannava/D/S-XG-NID/data/01_raw/CIC-IoT2023"
]

def check_file(filepath):
    try:
        # Try different encodings to handle UNSW
        df = pd.read_csv(filepath, nrows=1, encoding='cp1252') # Windows encoding often fixes UNSW
    except:
        try:
            df = pd.read_csv(filepath, nrows=1, encoding='latin1')
        except:
            return "Read Error"
            
    cols = df.columns.tolist()
    # Look for IP columns (case insensitive)
    ip_cols = [c for c in cols if 'src' in c.lower() or 'dst' in c.lower() or 'ip' in c.lower()]
    return ip_cols

print("--- DEEP HEADER INSPECTION ---")
for p in paths:
    print(f"\n📂 Checking: {p}")
    if not os.path.exists(p):
        print("   ❌ Path does not exist!")
        continue
        
    files = glob.glob(os.path.join(p, '**', '*.csv'), recursive=True)
    if files:
        # Check the first file found
        target = files[0]
        result = check_file(target)
        print(f"   📄 File: {os.path.basename(target)}")
        print(f"   ✅ Found Potential IPs: {result}")
    else:
        print("   ⚠️ No CSVs found in this specific folder.")