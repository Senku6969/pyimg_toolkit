#!/bin/bash
set -e  # Stop the script if any command fails

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py --server.port $PORT
