# Decimated file: /home/junzhewu/pohsun/data_decimated/grscenes_commercial/scenes/MWHLEPQKTIFZIAABAAAAAAA8_usd/start_result_navigation.usd
# Undecimated file: /home/junzhewu/data/isaac_scenes_v1/grscenes_commercial/scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation.usd
# python tests/count_faces_in_usd.py --input_file "/home/junzhewu/pohsun/data_decimated/grscenes_commercial/scenes/MWHLEPQKTIFZIAABAAAAAAA8_usd/start_result_navigation.usd"

## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu

import os
import sys
import argparse
# Ensure the package root (model_decimation) is on sys.path so `utils` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf
from utils.mesh_utils import count_total_faces_in_mesh_prim

def count_total_faces_in_usd(usd_path):
    """Counts the total number of mesh faces in a USD file."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[ERROR] Failed to open USD stage: {usd_path}")
        return 0

    total_faces = 0
    for prim in stage.Traverse():
        total_faces += count_total_faces_in_mesh_prim(prim)
    
    return total_faces

# --- SCRIPT USAGE EXAMPLE ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Count total mesh faces in USD files")
    parser.add_argument("--input_file", 
                        type=str, 
                        required=True,
                        help="Path to the input USD file to process")

    args = parser.parse_args()
    
    # Use parsed arguments
    input_file = args.input_file

    print(f"[INFO] Input file: {input_file}")
    print(f"[INFO] Starting mesh face counting process...")

    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file does not exist: {input_file}")
        sys.exit(1)

    # Count total faces in the input USD file
    total_faces = count_total_faces_in_usd(input_file)
    print(f"[INFO] Total faces in USD file '{input_file}': {total_faces}")
