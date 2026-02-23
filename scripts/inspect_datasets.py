import os

# Define the path to your data (on the SSD)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', '01_raw')

def print_file_content(filepath, lines=15):
    """Reads and prints the first few lines of text files."""
    try:
        print(f"\n   📄 PREVIEW ({os.path.basename(filepath)}):")
        print("   " + "-"*40)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= lines:
                    print("   ... [Truncated] ...")
                    break
                print(f"   | {line.strip()}")
        print("   " + "-"*40)
    except Exception as e:
        print(f"   [Error reading file: {e}]")

def scan_directory(path, level=0):
    """Recursively lists files and reads READMEs."""
    if not os.path.exists(path):
        print(f"❌ Path not found: {path}")
        return

    indent = "    " * level
    entries = sorted(os.listdir(path))
    
    for entry in entries:
        full_path = os.path.join(path, entry)
        
        # If Directory: Recurse
        if os.path.isdir(full_path):
            print(f"{indent}📁 {entry}/")
            scan_directory(full_path, level + 1)
        
        # If File: Print Name
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"{indent}---------- {entry} ({size_mb:.2f} MB)")
            
            # If it's a documentation file, read it
            if entry.lower().endswith(('.md', '.txt', '.info', '.json')):
                print_file_content(full_path)

if __name__ == "__main__":
    print(f"🔍 INSPECTING DATASETS AT: {DATA_DIR}\n")
    scan_directory(DATA_DIR)