# GR Scene USD Model Decimation Pipeline

## Overview

This repository contains a complete pipeline for processing USD (Universal Scene Description) models in GR scene to reduce their face count while preserving materials and hierarchy. The pipeline is designed to handle large datasets of 3D models, making them more suitable for real-time applications like robotics simulation.

## Pipeline Overview

The pipeline consists of three main steps:

1. **Prim Name Preprocessing** - Make USD prim names unique and limit their length
2. **Face Decimation** - Reduce mesh face count using Blender
3. **Mesh Replacement** - Replace original meshes with decimated versions while preserving materials

## Files Description

### Core Scripts

| File | Description | Dependencies |
|------|-------------|--------------|
| `make_usd_prim_unique.py` | Makes prim names unique and limits length to 60 characters | Python USD API |
| `decimate_usd_models_face_blender.py` | Batch decimates USD models using Blender | Blender |
| `replace_usd_models_isaac.py` | Replaces original meshes with decimated versions | Python USD API |
| `decimate_usd_models_face_autorestart.sh` | Main pipeline orchestrator with auto-restart capability | Bash, systemd |

### Testing/Debugging Scripts

| File | Description | Purpose |
|------|-------------|---------|
| `tests/decimate_usd_face_blender.py` | Single USD file decimation | Testing/debugging |
| `tests/replace_mesh_in_usd.py` | Single USD mesh replacement with debugging | Testing/debugging |
| `tests/count_faces_in_usd.py` | Count total mesh faces in usd file | Testing/debugging |
## Prerequisites

- **Python 3.8+** with USD Python bindings (`pxr`)
- **Blender 3.0+** with USD support
- **systemd** (for memory management)
- **Linux environment** (tested on Ubuntu)

## Installation

1. Install Blender with USD support:
```bash
$ sudo apt install blender
# Or download from https://www.blender.org/
```

2. Install USD Python bindings:
```bash
$ pip install usd-core
# Or use conda: conda install -c conda-forge usd-py
```

## Quick Start

### Basic Usage

Run the complete pipeline on a folder containing USD models:

```bash
# Make the script executable
$ chmod +x decimate_usd_models_face_autorestart.sh

# Run the pipeline
$ ./decimate_usd_models_face_autorestart.sh /path/to/your/usd/models
```

### Example

```bash
$ ./decimate_usd_models_face_autorestart.sh "/home/user/data/grscenes_commercial/models/"
```

## Pipeline Details

### Step 1: Prim Name Processing (`make_usd_prim_unique.py`)

**Purpose**: Ensures all prim names are unique and under 60 characters (Blender limitation)

**Input**: Folders containing `instance.usd` files  
**Output**: `instance_renamed.usd` files with processed names

**Features**:
- Limits prim names to 60 characters
- Adds unique suffixes to prevent naming conflicts
- Preserves USD hierarchy and relationships

**Manual Usage**:
```bash
$ python make_usd_prim_unique.py --input_folder /path/to/models
```

### Step 2: Face Decimation (`decimate_usd_models_face_blender.py`)

**Purpose**: Reduces mesh face count using Blender's decimation algorithms

**Input**: `instance_renamed.usd` files  
**Output**: `instance_renamed_decimated.usd` files

**Features**:
- Batch processing of multiple USD files
- Configurable decimation ratio (default: 0.1 = 10% of original faces)
- Skips meshes with < 500 faces (already optimized)
- Memory management and auto-restart capability

**Manual Usage**:
```bash
$ blender --background --python decimate_usd_models_face_blender.py -- \
  --input_folder /path/to/models --ratio 0.1
```

### Step 3: Mesh Replacement (`replace_usd_models_isaac.py`)

**Purpose**: Combines decimated geometry with original materials/hierarchy

**Input**: Both `instance_renamed.usd` and `instance_renamed_decimated.usd`  
**Output**: Final `instance.usd` with decimated geometry

**Features**:
- Preserves original USD structure and materials
- Copies geometry attributes (points, normals, UVs)
- Maintains parent-child relationships

**Manual Usage**:
```bash
$ python replace_usd_models_isaac.py --input_folder /path/to/models
```

## Configuration Options

### Auto-Restart Script Parameters

Edit `decimate_usd_models_face_autorestart.sh` to customize:

```bash
MAX_IDLE=10        # Seconds before killing hung Blender process
MEM_LIMIT="7G"     # Memory limit for Blender
RATIO=0.1          # Default decimation ratio (10%)
```

## File Structure Requirements

The pipeline expects this folder structure:

```
input_folder/
├── category1/
│   └── subcategory/
│       └── model_id/
│           └── instance.usd          # Original model
├── category2/
│   └── subcategory/
│       └── model_id/
│           ├── instance.usd          # Original
│           ├── instance_renamed.usd  # After step 1
│           ├── instance_renamed_decimated.usd  # After step 2
│           └── instance.usd          # Final output (overwrites original)
```

## Error Handling

### Auto-Restart Mechanism

The pipeline includes robust error handling:

- **Memory limits**: Prevents Blender from consuming too much RAM
- **Timeout detection**: Kills hung processes after no output for `MAX_IDLE` seconds
- **Automatic restart**: Restarts Blender if it crashes or hangs
- **Skip processed files**: Avoids reprocessing already completed models

### Common Issues

1. **"No module named 'pxr'"**: Install USD Python bindings
2. **Blender not found**: Install Blender or add to PATH
3. **Memory errors**: Reduce `MEM_LIMIT` or increase system RAM, or kill process then restart:
   ```bash
   $ pkill -9 -f "python.*decimate_usd_models_face_blender.py"
   ```

## Troubleshooting

### Debug Single Model

Test individual scripts on a single model:

```bash
# Test prim renaming
$ python make_usd_prim_unique.py --input_folder /path/to/single/model/

# Test decimation
$ blender --background --python decimate_usd_face_blender.py

# Test mesh replacement  
$ python replace_mesh_in_usd.py
```

## License

This project is developed by Po-Hsun Chang (pohsun@umich.edu). Please cite appropriately if used in research.