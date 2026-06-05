#!/bin/bash

# ==============================================================================
# DATA DOWNLOAD AUTOMATION SCRIPT FOR QKANs-ML4SCI
# ==============================================================================

# Get the absolute path of the directory where this script is located
REPO_ROOT="$(dirname "$(readlink -f "$0")")"

# Define the relative path to the data folder within the repo structure
DATA_PATH="$REPO_ROOT/data"

# Ensure the target directory exists before proceeding
mkdir -p "$DATA_PATH"
cd "$DATA_PATH" || exit 1

echo "🚀 Starting accelerated data download using aria2..."
echo "--------------------------------------------------"

# Initialize an array for URLs
URLS=(
    "https://zenodo.org/records/2603256/files/test.h5?download=1"
    "https://zenodo.org/records/2603256/files/train.h5?download=1"
    "https://zenodo.org/records/2603256/files/val.h5?download=1"
    "https://archive.ics.uci.edu/static/public/280/higgs.zip"
)

# Dynamically generate and append Quark-Gluon jet dataset URLs (_0 to _19)
for i in {0..19}; do
    URLS+=("https://zenodo.org/records/19362155/files/QG_jets_fp32_${i}.npz?download=1")
done

# Create a temporary file to feed URLs into aria2
F_TEMPORAL=$(mktemp)
for url in "${URLS[@]}"; do
    echo "$url" >> "$F_TEMPORAL"
done

# Execute aria2c with cautious settings to prevent Zenodo from triggering a 429 error
# -j2  : Downloads up to 2 files concurrently
# -x4  : Opens a maximum of 4 connections per file (safe threshold for Zenodo)
# -s4  : Allocates 4 splits per file
aria2c -j2 -x4 -s4 --no-netrc -i "$F_TEMPORAL"

# Capture execution status before removing the temporary file
ESTADO=$?
rm "$F_TEMPORAL"

# Post-processing: Clean up Zenodo's '?download=1' suffix from filenames
echo "🧹 Cleaning up downloaded filenames..."
for file in *\?download=1; do
    if [ -f "$file" ]; then
        mv "$file" "${file%\?download=1}"
    fi
done

echo "--------------------------------------------------"
if [ $ESTADO -eq 0 ]; then
    echo "✅ All datasets successfully downloaded!"
else
    echo "❌ Some downloads failed or were interrupted."
fi
echo "--------------------------------------------------"

# ==============================================================================
# Automatic Extraction of HIGGS Dataset
# ==============================================================================
ZIP_FILE="higgs.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "📦 Extracting $ZIP_FILE..."
    # -o overwrites existing files without prompting, -q runs quietly
    unzip -oq "$ZIP_FILE"
    if [ $? -eq 0 ]; then
        echo "✅ HIGGS dataset unzipped successfully."
        rm "$ZIP_FILE"
        echo "🗑️ Removed $ZIP_FILE to optimize disk space."
    else
        echo "❌ Failed to extract $ZIP_FILE"
    fi
fi

echo "🚀 Setup complete. Ready for training!"
