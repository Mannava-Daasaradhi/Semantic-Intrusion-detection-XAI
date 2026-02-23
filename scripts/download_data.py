import os
import requests
import sys

# Define where we want the data
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '01_raw')

# Official download links for UNSW-NB15 (CloudStor/Research mirrors)
# These are the specific "Training" and "Testing" splits used in research papers
FILES = {
    "UNSW_NB15_training-set.csv": "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv": "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW_NB15_testing-set.csv"
}

def download_file(url, filename):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"✅ {filename} already exists. Skipping.")
        return

    print(f"⬇️ Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ {filename} downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        # Clean up partial file
        if os.path.exists(filepath):
            os.remove(filepath)

def main():
    # Ensure directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"📂 Target Directory: {RAW_DATA_DIR}")
    
    # Check for requests library
    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' library is missing.")
        print("   Run: pip install requests")
        return

    for filename, url in FILES.items():
        download_file(url, filename)

    print("\n🎉 Download Checkpoint Complete.")
    print(f"   Verify files exist in: {RAW_DATA_DIR}")

if __name__ == "__main__":
    main()