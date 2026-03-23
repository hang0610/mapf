# MAPF (Multi-Agent Path Finding) Environment

A minimal, grid-based Multi-Agent Path Finding (MAPF) environment built on MovingAI benchmark maps. This package provides tools for loading maps, sampling instances, validating paths, and visualizing multi-agent trajectories.

## Visualization Examples

Here are example visualizations of multi-agent path finding on different maps with 50 agents using random movement:

### Warehouse Map
![Warehouse Map Demo](results/warehouse-20-40-10-2-2_demo.gif)
*50 agents moving randomly on warehouse-20-40-10-2-2 map*

### Room Map
![Room Map Demo](results/room-64-64-8_demo.gif)
*50 agents moving randomly on room-64-64-8 map*

### Random Map
![Random Map Demo](results/random-64-64-10_demo.gif)
*50 agents moving randomly on random-64-64-10 map*

### Maze Map
![Maze Map Demo](results/maze-128-128-2_demo.gif)
*50 agents moving randomly on maze-128-128-2 map*

### Mansion Map
![Mansion Map Demo](results/ht_mansion_n_demo.gif)
*50 agents moving randomly on ht_mansion_n map*

## Features

- **Grid-based MAPF Environment**: Discrete-time, grid-based environment for multi-agent path finding
- **MovingAI Map Support**: Load and work with standard MovingAI MAPF benchmark maps
- **Instance Sampling**: Sample MAPF instances (start/goal pairs) from scenario files
- **Path Validation**: Validate paths for collisions, illegal moves, and goal completion
- **Visualization**: Animate and playback paths as GIFs with collision highlighting
- **4-connected and 8-connected Motion**: Support for both 4-directional and 8-directional (including diagonals) movement
- **Comprehensive Testing**: Full test suite for all core functionality

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate mapf
```

### Using pip

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- ImageIO >= 2.9.0

## Quick Start

### Preview a Map

Visualize a map file to see its structure:

```bash
python -m scripts.preview_map data/mapf-map/empty-32-32.map
```

### Sample a MAPF Instance

Sample a MAPF instance with `k` agents from a scenario file:

```bash
python -m scripts.sample_instance --map empty-32-32 --k 10
```

This will:
- Load the map `empty-32-32.map`
- Sample 10 agents from the scenario file
- Display start and goal positions for each agent

### Run CBS (Conflict-Based Search)

Plan with the included **sum-of-costs CBS** solver (4-connected) and write `paths.npy`:

```bash
python -m scripts.run_cbs --map random-32-32-10 --k 10 --out paths.npy
```

Optional: `--max_time 256` increases the space–time search horizon if no solution is found; `--validate` runs `validate_paths` after planning.

Then validate and visualize as usual:

```bash
python -m scripts.validate_paths paths.npy --map random-32-32-10 --k 10
python -m scripts.playback_paths --map random-32-32-10 --paths paths.npy --k 10 --out results/cbs_demo.gif --fps 6
```

#### CBS run statistics and conflict-tree log

To record a **single JSON summary** of the run and a **JSONL trace** of each high-level CBS node expansion, pass `--stats_json` and `--log_ct`:

```bash
python -m scripts.run_cbs --map random-32-32-10 --k 10 --out paths.npy \
  --stats_json cbs_stats.json --log_ct data/ct_run.jsonl
```

- **`cbs_stats.json`** (from `--stats_json`): Run-level metrics—wall time, success or timeout, conflict-tree nodes popped, children enqueued, max open-list size, sum of costs, plus the map/scenario paths and planner options used for that run.
- **`ct_run.jsonl`** (from `--log_ct`): One JSON object per line, each describing a conflict-tree expansion (instrumentation for analysis or learning). Optional `--log_ct_features` adds per-conflict feature vectors; rollout labeling flags (`--rollout_max_pops`, etc.) attach extra fields when enabled—see `python -m scripts.run_cbs --help`.

### Validate Paths

Validate a solution path file:

```bash
python -m scripts.validate_paths paths.npy --map empty-32-32 --k 10
```

The validator checks for:
- Out-of-bounds positions
- Positions on obstacles
- Illegal moves (non-adjacent transitions)
- Vertex collisions (multiple agents on same cell)
- Edge collisions (agents swapping positions)
- Goal completion (if goals are provided)

### Playback Paths as GIF

Create an animated GIF from a path solution:

```bash
python -m scripts.playback_paths --map empty-32-32 --paths paths.npy --k 10 --out demo.gif --fps 6
```

### Random Rollout

Generate a random rollout visualization:

```bash
python -m scripts.random_rollout --map empty-32-32 --k 10 --steps 20 --motion 4
```

## Data

### Downloading Maps and Scenarios

Download MovingAI MAPF benchmark maps and scenarios from:
**https://www.movingai.com/benchmarks/mapf/index.html**

### Directory Structure

Place files in the following directories:
- **Map files** (`.map`): `data/mapf-map/`
- **Scenario files** (`.scen`): `data/scens/`

The scenario files should follow the naming convention: `<map-name>-random-*.scen`

## API Usage

### Basic Environment Usage

```python
from core.env import MAPFEnv
from core.instance import MAPFInstance
import numpy as np

# Create a MAPF instance (see instance sampling below)
instance = MAPFInstance(
    grid=grid,           # 2D numpy array (0=free, 1=obstacle)
    starts=starts,       # (N, 2) array of (row, col) start positions
    goals=goals,         # (N, 2) array of (row, col) goal positions
    num_agents=N
)

# Create environment (4-connected or 8-connected motion)
env = MAPFEnv(instance, motion="4")  # or motion="8"

# Reset to initial state
state = env.reset()

# Step with joint actions (one action per agent)
# Actions: 0=WAIT, 1=RIGHT, 2=DOWN, 3=UP, 4=LEFT
# For 8-connected: 5=DOWN-RIGHT, 6=DOWN-LEFT, 7=UP-RIGHT, 8=UP-LEFT
joint_action = np.array([1, 2, 0, 3, 4])  # Actions for 5 agents
state, info = env.step(joint_action)

# Access state information
print(f"Time step: {state.t}")
print(f"Agent positions: {state.pos}")  # (N, 2) array
print(f"Goals: {state.goals}")           # (N, 2) array

# Check for collisions
print(f"Vertex collisions: {info['vertex_collisions']}")
print(f"Edge collisions: {info['edge_collisions']}")
print(f"Invalid moves: {info['invalid_moves']}")
```

### Sampling Instances from Scenarios

```python
from mapf_env.io.movingai_map import load_map
from mapf_env.io.movingai_scene import load_scen
from core.instance import instance_from_scen

# Load map and scenario
grid = load_map("data/mapf-map/empty-32-32.map")
scen_starts, scen_goals = load_scen("data/scens/empty-32-32-random-1.scen")

# Create instance with k agents, starting from offset
instance = instance_from_scen(
    grid=grid,
    scen_starts=scen_starts,
    scen_goals=scen_goals,
    k=10,        # Number of agents
    offset=0     # Starting row in scenario file
)
```

### Path Validation

```python
from core.validate import validate_paths
import numpy as np

# Load paths (shape: T x N x 2)
paths = np.load("paths.npy")

# Validate paths
result = validate_paths(
    grid=grid,
    paths=paths,
    starts=starts,      # Optional: verify start positions
    goals=goals,        # Optional: verify goal completion
    connectivity="4"    # "4" or "8" for motion type
)

if result["ok"]:
    print("Paths are valid!")
else:
    print(f"Validation failed: {result['first_error']}")
    print(f"Vertex collisions: {result['num_vertex_collisions']}")
    print(f"Edge collisions: {result['num_edge_collisions']}")
    print(f"Illegal moves: {result['num_illegal_moves']}")
```

## Path Format Specification

All paths must follow this canonical format:

- **Shape**: `(T, N, 2)` where:
  - `T` = number of time steps
  - `N` = number of agents
  - `2` = (row, col) coordinates

- **Conventions**:
  - `paths[0, i]` = start position of agent `i`
  - `paths[T-1, i]` = final position of agent `i` (should be goal)
  - Coordinates are 0-based integers: `(row, col)`
  - Origin `(0, 0)` is top-left
  - If an agent reaches its goal early, it should wait (repeat goal position)

Example:
```python
paths = np.array([
    [[0, 0], [5, 5]],    # t=0: agent 0 at (0,0), agent 1 at (5,5)
    [[0, 1], [5, 4]],    # t=1: agent 0 at (0,1), agent 1 at (5,4)
    [[0, 2], [5, 3]],    # t=2: agent 0 at (0,2), agent 1 at (5,3)
    # ... more timesteps
])
```

## Action Space

### 4-Connected Motion

- `0`: WAIT → `(0, 0)`
- `1`: RIGHT → `(0, +1)`
- `2`: DOWN → `(+1, 0)`
- `3`: UP → `(-1, 0)`
- `4`: LEFT → `(0, -1)`

### 8-Connected Motion

Includes all 4-connected actions plus:
- `5`: DOWN-RIGHT → `(+1, +1)`
- `6`: DOWN-LEFT → `(+1, -1)`
- `7`: UP-RIGHT → `(-1, +1)`
- `8`: UP-LEFT → `(-1, -1)`

## Coordinate System

- Grid cells are indexed as **(row, col)**
- Both `row` and `col` are **0-based integers**
- Origin `(0, 0)` = **top-left** cell
- Increasing `row` → moves **down**
- Increasing `col` → moves **right**
- Access pattern: `grid[row, col]` or `grid[row][col]`

## Project Structure

```
mapf/
├── core/                    # Core MAPF functionality
│   ├── env.py              # MAPF environment implementation
│   ├── instance.py         # MAPF instance representation
│   └── validate.py         # Path validation logic
├── planners/               # MAPF planners
│   └── cbs.py              # Conflict-Based Search (sum-of-costs, 4-connected)
├── mapf_env/               # MAPF environment package
│   ├── io/                 # Input/output utilities
│   │   ├── movingai_map.py # Map file loading
│   │   └── movingai_scene.py # Scenario file loading
│   └── viz/                # Visualization
│       ├── render.py       # State rendering
│       └── animate.py      # Path animation
├── scripts/                 # Command-line tools
│   ├── sample_instance.py  # Sample MAPF instances
│   ├── run_cbs.py          # Run CBS planner → paths.npy
│   ├── validate_paths.py   # Validate path solutions
│   ├── playback_paths.py   # Create animated GIFs
│   ├── preview_map.py      # Preview map files
│   └── random_rollout.py   # Random rollout visualization
├── tests/                   # Test files
│   ├── test_sample_instance.py
│   └── test_validate_paths.py
├── data/                    # Data directory
│   ├── mapf-map/           # Map files (.map)
│   └── scens/              # Scenario files (.scen)
├── environment.yml          # Conda environment
├── requirements.txt         # pip requirements
├── SPEC.md                  # Detailed specification
└── TESTING.md              # Testing guide
```

## Testing

### Run All Tests

```bash
./test_all.sh
```

### Quick Tests

```bash
./test_quick.sh
```

### Individual Test Suites

```bash
# Test instance sampling
./test_sample_instance.sh
python tests/test_sample_instance.py

# Test path validation
./test_validate_paths.sh
python tests/test_validate_paths.py

# Test error cases
./test_error_cases.sh

# Test with different maps
./test_different_maps.sh
```

For more details, see [TESTING.md](TESTING.md).

## Command-Line Tools

### `sample_instance.py`

Sample a MAPF instance from a scenario file.

```bash
python -m scripts.sample_instance --map <map_name> --k <num_agents> [--offset <offset>]
```

**Options**:
- `--map`: Map basename (without `.map` extension)
- `--k`: Number of agents to sample
- `--offset`: Starting row index in scenario file (default: 0)
- `--maps_dir`: Directory containing map files (default: `data/mapf-map`)
- `--scen_dir`: Directory containing scenario files (default: `data/scens`)

### `validate_paths.py`

Validate a path solution file.

```bash
python -m scripts.validate_paths <paths.npy> --map <map_name> --k <num_agents> [options]
```

**Options**:
- `--map`: Map basename
- `--k`: Number of agents
- `--connectivity`: Motion type, "4" or "8" (default: "4")
- `--check_goals`: Verify all agents reach their goals

### `playback_paths.py`

Create an animated GIF from paths.

```bash
python -m scripts.playback_paths --map <map_name> --paths <paths.npy> [options]
```

**Options**:
- `--map`: Map basename
- `--paths`: Path to `.npy` file with shape `(T, N, 2)`
- `--out`: Output GIF filename (default: `paths.gif`)
- `--k`: Number of agents (for loading starts/goals from scenario)
- `--fps`: Frames per second (default: 5)
- `--stride`: Temporal downsampling factor (default: 1)
- `--no_collision_highlight`: Disable collision highlighting

### `preview_map.py`

Preview a map file.

```bash
python -m scripts.preview_map <map_file>
```

### `random_rollout.py`

Generate a random rollout visualization.

```bash
python -m scripts.random_rollout --map <map_name> --k <num_agents> --steps <num_steps> [--motion 4|8]
```

## Specification

For detailed specifications on coordinate systems, path formats, action spaces, and conventions, see [SPEC.md](SPEC.md).

## Examples

### Complete Workflow

```bash
# 1. Sample an instance
python -m scripts.sample_instance --map empty-32-32 --k 10

# 2. Run your planner (produces paths.npy)
# ... your planning code ...

# 3. Validate the solution
python -m scripts.validate_paths paths.npy --map empty-32-32 --k 10 --check_goals

# 4. Visualize as GIF
python -m scripts.playback_paths --map empty-32-32 --paths paths.npy --k 10 --out results/demo.gif --fps 6
```

### Running MAPF Demos on Multiple Maps

Generate visualization GIFs for multiple maps with many agents using random movement:

```bash
# Activate conda environment
conda activate mapf

# Run demos on multiple maps
bash run_mapf_demos.sh
```

Or run directly:
```bash
PYTHONPATH=. python scripts/run_mapf_demos.py \
    --maps warehouse-20-40-10-2-2 room-64-64-8 random-64-64-10 maze-128-128-2 ht_mansion_n \
    --k 50 \
    --fps 5 \
    --results_dir results
```

## License

See LICENSE file for details.

## Citation

If you use this MAPF environment in your research, please cite the MovingAI benchmark:

> MovingAI Lab. "MAPF Benchmarks." https://www.movingai.com/benchmarks/mapf/index.html

## Contributing

When contributing to this project:

1. Follow the conventions specified in `SPEC.md`
2. Ensure all tests pass: `./test_all.sh`
3. Maintain the path format: `(T, N, 2)` with `(row, col)` coordinates
4. Use the standard action space (0-4 for 4-connected, 0-8 for 8-connected)
