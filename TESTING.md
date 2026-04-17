# Testing Guide

This directory contains comprehensive test files and scripts for the core MAPF utilities, plus smoke tests for learned CBS conflict selection.

## Test Files

### Python Test Files (Strict Tests)

1. **`tests/test_sample_instance.py`** - Comprehensive Python tests for `sample_instance.py`
   - Tests basic usage, different maps, k values, offsets, error cases
   - Run with: `python tests/test_sample_instance.py`

2. **`tests/test_validate_paths.py`** - Comprehensive Python tests for `validate_paths.py`
   - Tests validation, error handling, different connectivity, edge cases
   - Run with: `python tests/test_validate_paths.py`

3. **`tests/test_cbs_learning.py`** - Smoke tests for learned CBS conflict selection
   - Tests rollout-label supervision, model save/load, and an end-to-end `collect -> train -> run_cbs --conflict_policy learned` loop
   - Run with: `python tests/test_cbs_learning.py`

### Shell Script Test Files

1. **`test_sample_instance.sh`** - Shell script tests for `sample_instance.py`
   - Tests various scenarios and configurations
   - Run with: `./test_sample_instance.sh`

2. **`test_validate_paths.sh`** - Shell script tests for `validate_paths.py`
   - Tests validation with different parameters
   - Run with: `./test_validate_paths.sh`

3. **`test_all.sh`** - Run all tests
   - Executes all test scripts
   - Run with: `./test_all.sh`

4. **`test_quick.sh`** - Quick essential tests
   - Runs only the most important tests
   - Run with: `./test_quick.sh`

5. **`test_error_cases.sh`** - Test error handling
   - Tests various error scenarios and edge cases
   - Run with: `./test_error_cases.sh`

6. **`test_different_maps.sh`** - Test with different maps
   - Tests both scripts with various map files
   - Run with: `./test_different_maps.sh`

## Usage

### Quick Start

```bash
# Run quick tests
./test_quick.sh

# Run all tests
./test_all.sh

# Run specific test suite
./test_sample_instance.sh
./test_validate_paths.sh
```

### Running Python Tests

```bash
# Run strict Python tests
python tests/test_sample_instance.py
python tests/test_validate_paths.py
```

### Running Individual Test Scripts

```bash
# Test error cases
./test_error_cases.sh

# Test with different maps
./test_different_maps.sh
```

## Test Coverage

### sample_instance.py Tests

- ✅ Basic usage with different maps
- ✅ Different k values (1, 5, 10, 20, 50, 100)
- ✅ Offset parameter
- ✅ Explicit map and scenario paths
- ✅ Error cases (invalid maps, missing arguments)
- ✅ Large k values
- ✅ Different map types (empty, maze, room, etc.)

### validate_paths.py Tests

- ✅ Basic validation
- ✅ Validation with goals check
- ✅ Invalid paths detection
- ✅ Missing file error handling
- ✅ Example path detection
- ✅ Different connectivity (4 vs 8)
- ✅ Wrong array shape detection
- ✅ Different maps
- ✅ Offset parameter

### Learned CBS Tests

- ✅ Rollout-label to effort-target conversion (`E_sum`, `E_min`, censoring)
- ✅ Two-hidden-layer MLP export/import consistency and parameter shape checks
- ✅ End-to-end dataset collection, training, and learned-policy CBS smoke test

## Expected Output

All tests should complete successfully. If a test fails:

1. Check the error message
2. Verify that required data files exist (`data/mapf-map/`, `data/scens/`)
3. Ensure the conda environment is activated: `conda activate mapf`
4. Check that all dependencies are installed

## Troubleshooting

### Common Issues

1. **Permission denied**: Make scripts executable with `chmod +x test_*.sh`
2. **Module not found**: Ensure you're in the mapf directory and conda environment is activated
3. **File not found**: Check that map and scenario files are downloaded in `data/` directories
4. **Test failures**: Some tests may fail if scenario files are missing for certain maps

## Adding New Tests

To add new tests:

1. **For shell scripts**: Add test cases to the appropriate `.sh` file
2. **For Python tests**: Add test functions to `tests/test_*.py` files
3. Follow the existing test patterns and naming conventions
