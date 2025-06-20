#!/bin/bash

# Check if Singularity is installed
if ! command -v singularity &> /dev/null; then
    echo "Singularity is not installed. Installing it now..."
    
    # Make install script executable and run it
    bash install_singularity.sh
    
    # Exit if installation failed
    if ! command -v singularity &> /dev/null; then
        echo "Failed to install Singularity. Exiting."
        exit 1
    fi
fi

# Build the Singularity container if needed
SINGULARITY_IMAGE="vln_sg_nav.sif"
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "Singularity image not found. Building it now..."
    # Set the path for the Singularity image
    SINGULARITY_IMAGE="vln_sg_nav.sif"

    # Create a temporary directory with more space
    TEMP_DIR=~/singularity_tmp
    mkdir -p $TEMP_DIR

    # Set environment variable to use this directory for temporary files
    export SINGULARITY_TMPDIR=$TEMP_DIR

    echo "Building Singularity container for VLN..."
    echo "Using temporary directory: $SINGULARITY_TMPDIR"
    echo "Building from local Docker image"

    # Build from the local Docker image
    singularity build $SINGULARITY_IMAGE docker-daemon://curly/sg_vln:latest

    if [ $? -eq 0 ]; then
        echo "Singularity container built successfully at $SINGULARITY_IMAGE"
    else
        echo "Failed to build Singularity container"
        exit 1
    fi
fi