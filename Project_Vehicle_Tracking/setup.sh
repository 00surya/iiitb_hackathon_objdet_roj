#!/bin/bash

# Setup script for YOLOv8 hackathon project using Conda

# Function to create and activate the Conda environment
setup_environment() {
    echo "Creating the Conda environment..."
    
    # Create Conda environment with necessary packages
    conda create -y -n hackathon_env python=3.9

    # Activate the environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate hackathon_env
}

# Function to install dependencies using Conda
install_dependencies() {
    echo "Installing dependencies with Conda..."

    # Install dependencies (replace these with exact package names in your requirements)
    conda install -y -c conda-forge opencv matplotlib pandas seaborn scikit-learn tqdm
    conda install -y pytorch torchvision torchaudio -c pytorch
    conda install -y -c conda-forge ultralytics  # YOLOv8 package
    conda install -y gdown  # Install gdown for downloading from Google Drive

    # If additional dependencies are required, add them here
    echo "Installed all necessary dependencies."
}

# Function to download YOLOv8 model from Google Drive folder
download_model() {
    echo "Downloading YOLOv8 model..."

    # Create 'models' directory if it does not exist
    mkdir -p models

    # URL for the model file (replace 'YOUR_FILE_ID' with the actual file ID from your Google Drive)
    FILE_ID="YOUR_FILE_ID"  # Replace this with the actual file ID from your folder

    # Download the model using gdown
    gdown "https://drive.google.com/uc?id=$FILE_ID" -O models/model.pt

    echo "YOLOv8 model downloaded and saved to 'models/' directory."
}

# Main setup script
main() {
    echo "Starting setup..."

    # Step 1: Set up the environment
    setup_environment

    # Step 2: Install dependencies
    install_dependencies

    # Step 3: Download YOLOv8 model
    download_model

    echo "Setup complete!"
    echo "Activate the environment with 'conda activate hackathon_env'."
}

# Run the main setup
main
