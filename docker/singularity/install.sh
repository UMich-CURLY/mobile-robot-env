#!/bin/bash
# Script to download and install Singularity-CE 4.3.0 for Ubuntu Focal

echo "Starting Singularity-CE installation..."

# Download the Singularity-CE deb package
echo "Downloading Singularity-CE 4.3.0 deb package..."
cd /tmp
wget -q https://github.com/sylabs/singularity/releases/download/v4.3.0/singularity-ce_4.3.0-focal_amd64.deb

# Check if download was successful
if [ ! -f "/tmp/singularity-ce_4.3.0-focal_amd64.deb" ]; then
    echo "Failed to download Singularity-CE deb package."
    exit 1
fi
echo "Download complete."

# Install dependencies and fix potential broken packages
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y fuse2fs || true
sudo apt --fix-broken install -y

# Install Singularity-CE
echo "Installing Singularity-CE..."
sudo dpkg -i /tmp/singularity-ce_4.3.0-focal_amd64.deb
if [ $? -ne 0 ]; then
    echo "Trying to fix dependencies and install again..."
    sudo apt --fix-broken install -y
    sudo dpkg -i /tmp/singularity-ce_4.3.0-focal_amd64.deb
fi

# Verify installation
if command -v singularity &> /dev/null; then
    echo "Singularity-CE $(singularity --version) installed successfully!"
else
    echo "Singularity installation failed. Please check the error messages above."
    exit 1
fi

# Clean up
echo "Cleaning up..."
rm -f /tmp/singularity-ce_4.3.0-focal_amd64.deb

echo "Singularity installation complete!"