#!/bin/bash
set -e  # Stop the script if any command fails

# Install Git LFS manually
apt-get update && apt-get install -y git-lfs

# Initialize and fetch LFS files
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py --server.port $PORT
