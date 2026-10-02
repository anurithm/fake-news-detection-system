import urllib.request
import os
import pandas as pd
from pathlib import Path

def download_csv():
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
    data_dir = Path(__file__).parent
    output_path = data_dir / "dataset.csv"
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as e:
        print(f"Failed to download with urllib: {e}")
        print("Attempting with curl...")
        os.system(f"curl -L {url} -o {output_path}")

    print("Download complete. Verifying schema...")
    df = pd.read_csv(output_path)
    
    print("\nOriginal columns:", df.columns.tolist())
    
    # The dataset has Unnamed: 0, title, text, label.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    print("\nVerified columns:", df.columns.tolist())
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print(f"\nTotal samples: {len(df)}")
    
    # Save a clean version
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved successfully at {output_path}")

if __name__ == "__main__":
    download_csv()
