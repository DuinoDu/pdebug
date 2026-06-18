# Trajectory Analysis Script

`trajectory_analysis.py` is a tool for analyzing trajectory data, classifying movement states based on geometric properties, and visualizing the results.

## Usage

```bash
python3 scripts/trajectory_analysis.py [OPTIONS]
```

## Inputs & Outputs

### Input (`--input`)
The script supports three input formats:
1. **TUM Text (`.txt`)**: Standard space-separated format (`timestamp x y z qx qy qz qw`).
2. **JSON (`.json`)**: Dictionary where keys are timestamps and values include odometry data.
3. **Lance Dataset (`.lance`)**: A Lance dataset URI. Must contain `timestamp` and `odo` columns.

### Output
1. **States Data (`--states`)**:
   - If output is `.txt`: Writes a space-separated file with full analysis details (yaw, velocity, state, etc.).
   - If output is `.lance` (and input is `.lance`): Appends/Overwrites the `action_tag` column in the dataset.
2. **Visualization (`--output`)**:
   - Generates a PNG file (default `trajectory_analysis.png`) showing:
     - 2D Trajectory colored by state.
     - State distribution over time.
     - Histogram of future yaw differences.

## Lance Dataset Integation

When working with Lance datasets, the script adds a specific column to the schema:

| Column Name | Type | Description |
|-------------|------|-------------|
| **`action_tag`** | `string` | The classified movement state. |

**Streaming Updates**: The script reads the input dataset in batches (default size 100), appends the `action_tag` column, and writes to the new/updated dataset path using `lance.write_dataset` in streaming mode. This preserves all existing columns (except `action_tag` which is updated found).

### State Categories
The `action_tag` will contain one of the following string values:
- `stop`: Velocity below threshold.
- `forward`: Moving generally in the direction of orientation (within ±90°).
- `backward`: Moving generally opposite to orientation.
- `*_left`: Turning left (Future yaw - Current yaw > 15°).
- `*_right`: Turning right (Future yaw - Current yaw < -15°).

*Note: Turn detection is based on a "future" point lookahead relative to the current point.*

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `tum.txt` | Path to input file or Lance dataset URI. |
| `--output` | `trajectory_analysis.png` | Path where the visualization image will be saved. |
| `--states` | `trajectory_states.txt` | Path output data. Use `.lance` extension for dataset updates. |
| `--override` | `False` | Reforce classification even if `action_tag` exists in the input Lance dataset. |
| `--no-viz` | `False` | Skip generating the visualization image. |
| `--skip` | `100` | Number of points to skip between direction arrow indicators in plots. |
| `--threshold`| `0.01` | Velocity threshold for detecting `stop` state. |
| `--future` | `3` | Number of steps ahead to compare orientation for turn detection. |
| `--separate-triggers` | `False` | Save separate visualization files for each unique trigger segment. |

---

# Lance Dataset Visualization Script

`visualize_lance.py` creates comprehensive visualization collages from an annotated Lance dataset. It's designed to inspect the outputs of various AI models (captioning, detection, segmentation, depth) stored within the dataset.

## Usage

```bash
# Basic usage
python scripts/visualize_lance.py <input_dataset> --output <output_dir>

# With stride and batch sizing
python scripts/visualize_lance.py <input_dataset> --output <output_dir> --stride 10 --batch-size 50
```

## Features

- **Batch Processing**: Processes images in configurable batches to manage memory usage (OOM prevention).
- **Selective Visualization**: Use `--stride` to visualize a subset of the data (e.g., every 10th frame).
- **Multi-Model Collage**: Each output image combines:
    - Original RGB Image (with Action Tag overlay if present)
    - Caption (e.g., Moondream) & Tags (e.g., Qwen)
    - Object Detection boxes (e.g., GroundingDINO)
    - Semantic/Instance Segmentation masks (e.g., InternImage)
    - Depth Maps (e.g., ML-Depth-Pro)

## Configuration Options

| Option | Argument | Default | Description |
|--------|----------|---------|-------------|
| **Input** | `<input_dataset>` | *Required* | Path to the Lance dataset folder. |
| **Output** | `--output`, `-o` | *Required* | Directory to save the generated PNG collages. |
| **Image Col** | `--image-col` | `camera_left` | Column name containing the image binary/bytes. |
| **Stride** | `--stride` | `1` | Interval for visualization (1 = all frames, 10 = every 10th, etc.). |
| **Batch Size** | `--batch-size` | `100` | Number of records to load into memory at once. |
| **Preview** | `--preview/--no-preview` | `True` | Whether to print the first 3 rows metadata to console before running. |
| **Filters** | `--row-limit` | `None` | Stop after processing N rows total. |
| **Filters** | `--trigger` | `None` | Filter dataset by a specific trigger event string. |
| **Metadata** | `--timestamp-col` | `None` | Optional column for timestamp display. |

## Output Format
The script generates `.png` files in the specified output directory following the naming convention:
`{index:04d}_{image_name}.png`

