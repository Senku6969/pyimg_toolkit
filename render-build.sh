#!/bin/bash
set -e

# Git LFS workaround (since Render doesn't support apt installs)
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS is not available, skipping LFS pull..."
else
    git lfs install
    git lfs pull
fi

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py --server.port $PORT
