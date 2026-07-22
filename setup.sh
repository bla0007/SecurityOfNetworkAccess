#!/bin/bash
# setup.sh — SONA (Security of Network Access) — One-time environment setup
# Run from inside the sona/ folder

echo "======================================"
echo " SONA — Security of Network Access"
echo " Environment Setup"
echo "======================================"

# 1. Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Done."

# 3. Create output directories
echo ""
echo "[3/4] Creating directories..."
mkdir -p data models plots

# 4. Remind user to download data
echo ""
echo "[4/4] Next steps:"
echo ""
echo "  a) Download the NSL-KDD dataset from:"
echo "     https://www.unb.ca/cic/datasets/nsl.html"
echo ""
echo "  b) Place these two files inside data/:"
echo "     data/KDDTrain+.txt"
echo "     data/KDDTest+.txt"
echo ""
echo "  c) Run EDA:"
echo "     cd notebooks && python 01_eda.py"
echo ""
echo "  d) Train models:"
echo "     python src/train.py"
echo ""
echo "  e) Launch SONA dashboard:"
echo "     streamlit run dashboard/app.py"
echo ""
echo "======================================"
echo " Setup complete. Activate with:"
echo "   source venv/bin/activate"
echo "======================================"
