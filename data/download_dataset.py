import os
from pathlib import Path
import pandas as pd

def download_csv():
    data_dir = Path(__file__).parent
    output_path = data_dir / "dataset.csv"
    sample_path = data_dir / "sample_data.csv"
    
    print("Initiating WELFake Dataset extraction...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not found. Installing...")
        os.system("pip install datasets")
        from datasets import load_dataset

    # Load WELFake from HuggingFace
    print("Loading davanstrien/WELFake from HuggingFace...")
    ds = load_dataset("davanstrien/WELFake", split="train")
    df = ds.to_pandas()
    
    # Map classes explicitly 
    # WELFake has label: 0=REAL, 1=FAKE. 
    df["label"] = df["label"].map({0: "REAL", 1: "FAKE"})
    df = df[["title", "text", "label"]]
    
    dfs_to_concat = [df]
    
    # Combine with local sample_data to inject niche domains perfectly
    if sample_path.exists():
        print(f"Loading local augmented domains from {sample_path}...")
        df_sample = pd.read_csv(sample_path)
        if all(c in df_sample.columns for c in ["title", "text", "label"]):
            df_sample["label"] = df_sample["label"].str.upper()
            dfs_to_concat.append(df_sample[["title", "text", "label"]])
            
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Final data cleaning
    combined_df = combined_df.dropna(subset=["text", "label"])
    empty_mask = combined_df["text"].str.strip() == ""
    combined_df = combined_df[~empty_mask]
    
    n_dups = combined_df.duplicated(subset=["text"]).sum()
    if n_dups > 0:
        print(f"Dropping {n_dups} duplicate texts...")
        combined_df = combined_df.drop_duplicates(subset=["text"])
        
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal Dataset schema verified:", combined_df.columns.tolist())
    print("Class distribution:")
    print(combined_df["label"].value_counts())
    print(f"Total clean samples: {len(combined_df)}")
    
    combined_df.to_csv(output_path, index=False)
    print(f"\nDataset saved successfully at {output_path}")

if __name__ == "__main__":
    download_csv()
