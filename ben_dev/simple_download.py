#!/usr/bin/env python3
"""
Simple script to download TLDR dataset and save as parquet files.
Parquet is faster and more efficient than JSON for training.
"""

import pandas as pd
from pathlib import Path
import requests

def download_tldr_parquet():
    """Download and save TLDR dataset as parquet files."""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Downloading TLDR dataset as parquet files...")
    
    # Direct URLs to the parquet files on HuggingFace (using resolve for raw download)
    urls = {
        "train": "https://huggingface.co/datasets/CarperAI/openai_summarize_tldr/resolve/main/data/train-00000-of-00001-e8c59e5cf7bce1c0.parquet",
        "valid": "https://huggingface.co/datasets/CarperAI/openai_summarize_tldr/resolve/main/data/valid-00000-of-00001-0e33e6bd86e3edc9.parquet", 
        "test": "https://huggingface.co/datasets/CarperAI/openai_summarize_tldr/resolve/main/data/test-00000-of-00001-59ffb27399371eac.parquet"
    }
    
    for split_name, url in urls.items():
        print(f"Downloading {split_name} split...")
        
        try:
            # Download the parquet file
            response = requests.get(url)
            response.raise_for_status()
            
            # Save parquet file directly
            output_file = data_dir / f"tldr_{split_name}.parquet"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Check the file by reading it
            df = pd.read_parquet(output_file)
            print(f"Saved {len(df)} examples to {output_file}")
            print(f"  Columns: {list(df.columns)}")
            
        except Exception as e:
            print(f"Error downloading {split_name}: {e}")
    
    print("\nDone! Parquet files saved in data/ directory")
    print("These are faster to load than JSON for training!")

if __name__ == "__main__":
    download_tldr_parquet()
