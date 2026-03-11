# 3D Synthetic Dataset Generator

A Blender-based pipeline for generating synthetic 3D scene datasets with multiple views. This tool creates realistic indoor scenes by placing 3D objects from Objaverse in rooms with random materials, lighting, and camera positions.

## Features

- **Multi-view scene generation**: Generate multiple camera views of the same scene
- **Objaverse integration**: Automatic downloading and filtering of 3D objects from Objaverse dataset
- **Smart object selection**: Filters objects based on file size, vertex count, texture quality, and visual properties
- **Realistic materials**: Random PBR materials for floors and walls
- **Automated rendering**: Generates RGB images, depth maps, and object masks

## Requirements

- **Blender** 4.5.4
- Python packages (see `requirements.txt`)

### Python Dependencies

```
pillow
matplotlib
opencv-python
objaverse
scipy
pygltflib
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/imricc4rdo/dataset.git
cd dataset
```

### 2. Setup Blender Python environment

Edit the `BLENDER_PATH` in `blender_env_setup.sh` to match your Blender installation:

```bash
# Linux/Mac
bash blender_env_setup.sh
```

Or manually install dependencies into Blender's Python:

```bash
/path/to/blender/python -m pip install -r requirements.txt
```

## Usage

*Note: To run Blender from the terminal, you need to add the blender command to your PATH. A command can only be executed from the terminal if its executable is located in a directory included in the PATH environment variable.*

### Basic usage

```bash
blender --background --python create_dataset.py
```

### With custom parameters

```bash
blender --background --python create_dataset.py -- \
    --n_scenes 100 \
    --n_views 3 \
    --min_objects 2 \
    --max_objects 5 \
    --common_objects 2 \
    --out_folder ./output
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_scenes` | 500 | Number of scenes to generate |
| `--n_views` | 3 | Number of views per scene |
| `--max_view_attempts` | 5 | Maximum number of attempts for a view |
| `--min_objects` | 3 | Minimum objects per view |
| `--max_objects` | 5 | Maximum objects per view |
| `--common_objects` | 2 | Objects shared across all views |
| `--area` | 2000.0 | Placement area (X × Y) |
| `--walls_height` | 25.0 | Height of room walls |
| `--ground_z` | 0.0 | Ground Z coordinate |
| `--obj_folder` | ./objects | Folder containing downloaded objects |
| `--ann_folder` | ./annotations | Folder for Objaverse annotations |
| `--out_folder` | ./output | Output directory |
| `--background_images_folder` | ./background_images | Folder with background images |
| `--size_mb_threshold` | 40.0 | Max object file size (MB) |
| `--vertices_threshold` | 50000 | Max vertices per object |
| `--texture_variance_threshold` | 450.0 | Min texture color variance |
| `--texture_entropy_threshold` | 3.5 | Min texture entropy |
| `--overlap_coverage_threshold` | 0.3 | Min surface overlap between views |
| `--padding` | 0.10 | Extra spacing between objects |
| `--processes` | None | Number of processes to use for downloading objects |

## Project Structure

```
.
├── render_examples/            # Folder with some render examples 
├── utils/                      # Utility modules
    ├── common_utils.py
    ├── mapping_utils.py
    ├── objaverse_utils.py
    ├── room_utils.py
    └── validation_utils.py
├── blender_env_setup.sh        # Blender Python environment setup
├── create_dataset.py           # Main dataset generation script
├── mapping_example.ipynb       # Example mapping notebook
├── requirements.txt            # Python dependencies
└── viz_images.ipynb            # Visualization notebook 
```

## Output Format

The generated dataset is organized as follows:

```
output/
├── n_objects1/
│   ├── 0000/
│   │   ├── camera.npz           # Camera parameters
│   │   ├── depth.exr            # Depth map
│   │   ├── obj_mask.exr         # Object mask
│   │   ├── objs_per_view.npz    # Objects data
│   │   └── render.png           # Rendered image
│   ├── 0001/
│   └── 0002/
├── n_objects2/
└── ...
```
