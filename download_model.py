#!/usr/bin/env python3
"""Download Epona model from Hugging Face with progress bar"""

from huggingface_hub import hf_hub_download
import os

print("Downloading Epona NuPlan model...")
print("This may take a while depending on your internet connection.")

try:
    file_path = hf_hub_download(
        repo_id="Kevin-thu/Epona",
        filename="epona_nuplan.pkl",
        local_dir="pretrained",
        resume_download=True,  # Enable resume
    )
    print(f"\n✅ Download completed successfully!")
    print(f"Model saved to: {file_path}")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("You may need to:")
    print("1. Check your internet connection")
    print("2. Try downloading manually from https://huggingface.co/Kevin-thu/Epona")
