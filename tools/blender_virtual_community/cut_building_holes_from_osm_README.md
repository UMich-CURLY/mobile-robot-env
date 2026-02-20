# cut_building_holes_from_osm.py

A Blender script that cuts rectangular holes in building façades at OSM amenity (POI) locations. It uses only building, road, and OSM map data—no storefront files required. Buildings are imported from `{city}/buildings`, and the entire scene is exported to `generated/{city}/`.

---

## Features

- Imports all buildings from `{city}/buildings` folder (.usd, .usdc, .glb, .gltf)
- Loads roads and amenity points from OSM files
- Cuts rectangular holes on building façades at each amenity location
- Fixed hole dimensions (configurable, no storefront USD needed)
- Hole orientation determined automatically from nearby road direction
- Exports the full scene: all buildings included (both with holes and unchanged)

---

## Requirements

- **Blender 4.x** (with USD import/export support)
- **Python package**: `tqdm` (optional; falls back to plain iteration if not installed)

---

## Directory Structure & Inputs

```
ViCo/
├── cut_building_holes_from_osm.py
├── generated/
│   └── {city}/
│       └── {city}_cutholes.usd     # Output
└── {city}/
    ├── buildings/                   # Input: all building files
    │   ├── *.usd
    │   ├── *.glb
    │   └── ...
    └── road_data/
        └── road_data_with_height.osm  # Input: roads and amenity points
```

**Input details:**
- `{city}/buildings/`: Folder containing building files (.usd, .usdc, .glb, .gltf). All files in this folder are imported.
- `road_data_with_height.osm`: Must contain `highway` roads and `amenity` POI nodes with `utm_x`, `utm_y`, and `height` tags.

---

## Configuration

Edit the `CONFIG` dictionary at the top of the script:

| Parameter | Description | Default |
|------------|-------------|---------|
| `CITIES` | List of cities to process | `["MADRID"]` |
| `BASE_DIR` | Project root directory | `"D:/Desktop/ViCo/"` |
| `BUILDING_FOLDER` | Subfolder name under `{city}/` for buildings | `"buildings"` |
| `HOLE_WIDTH` | Hole width (meters) | `4.0` |
| `HOLE_HEIGHT` | Hole height (meters) | `3.5` |
| `HOLE_DEPTH_MARGIN` | Depth margin for cutting (meters) | `0.1` |
| `OUTPUT_SUFFIX` | Suffix for output filename | `"_cutholes"` |
| `BUILDING_NAME_KEYWORDS` | Filter buildings by name (empty = no filter) | `[]` |
| `TEST_MAX_POIS` | Limit to first N amenity points for testing; set to `None` or `0` to process all | `5` |

---

## Output

- **Path**: `generated/{city}/{city}_cutholes.usd`
- **Content**: All buildings from the input folder. Buildings near amenity points have rectangular holes cut in their façades; all other buildings are included unchanged. No buildings are removed.

---

## Running the Script

### Terminal (Blender on D: drive)

```powershell
cd d:\Desktop\ViCo
D:\Blender\blender.exe --background --python cut_building_holes_from_osm.py
```

If Blender is elsewhere (e.g. `D:\Program Files\Blender Foundation\Blender 4.5\`):

```powershell
"D:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python cut_building_holes_from_osm.py
```

### With Blender GUI (for debugging)

```powershell
D:\Blender\blender.exe --python cut_building_holes_from_osm.py
```

### Inside Blender

1. Open Blender
2. Switch to the Scripting workspace, open `cut_building_holes_from_osm.py`
3. Click Run Script

---

## Amenity Types

The script processes POI points with these amenity types: `restaurant`, `cafe`, `fast_food`, `parking_entrance`, `bar`, `bank`, `pub`, `atm`, `bicycle_rental`, `parking`, `pharmacy`, `toilets`, `theatre`, `library`, `dentist`, `school`, `post_office`, `bureau_de_change`, `doctors`, `bicycle_repair_station`, `clinic`, `community_centre`, `car_rental`, `police`, `arts_centre`, `cinema`, `kindergarten`, `university`, `coworking_space`, `college`.

Edit the `AMENITY_TYPES` list in the script to change this.
