# RP1 Antibody Pipeline - Test Suite

Test suite for the RP1 antibody discovery pipeline.

## Overview

This directory contains automated tests for pipeline components, with focus on the checkpoint system.

## Test Files

### test_checkpoints.py

Tests checkpoint manager functionality:
- ✅ CheckpointManager initialization
- ✅ Save operations (multiple data types)
- ✅ Load operations with integrity verification
- ✅ Stage listing and management
- ✅ Data format handling (numpy, lists, dicts, pickles)

## Running Tests

### Quick Start

```bash
# From project root
python test_checkpoints.py
```

### As Python Module

```bash
# Run from anywhere
python -m RP1_antibody_pipeline.tests.test_checkpoints
```

### Within Virtual Environment

```bash
# Activate venv first (Windows)
.venv\Scripts\activate

# Then run tests
python test_checkpoints.py
```

## Expected Output

```
Testing CheckpointManager...
[OK] Initialized CheckpointManager: RP1_antibody_pipeline\experiments\checkpoints\test\20260226_092738
[OK] Saved Stage 0 checkpoint: ...
[OK] Saved Stage 2 checkpoint: ...
[OK] Saved Stage 2b checkpoint: ...
[OK] Saved stages: ['test_stage_0', 'test_stage_2', 'test_stage_2b']
[OK] Loaded Stage 2 checkpoint
[OK] Data integrity verified
  - Sequences: 3
  - Scores: [0.85 0.92 0.78]
[OK] Loaded Stage 2b checkpoint
  - Binding matrix shape: (3, 2)
[OK] All runs: [...]

[SUCCESS] All tests passed!
```

## Test Data Location

Test checkpoints are saved to:
```
RP1_antibody_pipeline/experiments/checkpoints/test/
```

### Cleanup Test Data

```bash
# Remove all test checkpoints
rm -rf RP1_antibody_pipeline/experiments/checkpoints/test/

# Or manually delete specific test runs
rm -rf RP1_antibody_pipeline/experiments/checkpoints/test/20260226_*
```

## Test Coverage

### Current Coverage

- ✅ CheckpointManager initialization with default/custom directories
- ✅ Stage data saving (Stage 0, 2, 2b)
- ✅ Multiple data format saving (numpy arrays, lists, dicts)
- ✅ Stage data loading
- ✅ Data integrity verification (round-trip)
- ✅ Run listing functionality
- ✅ Stage listing per run

### Future Coverage (Planned)

- ⏳ Full pipeline integration tests
- ⏳ Performance benchmarks
- ⏳ Error handling tests
- ⏳ Concurrent access tests
- ⏳ Large dataset tests
- ⏳ Resume-from-checkpoint tests

## Writing New Tests

### Test Template

```python
"""
test_new_feature.py - Description of what this tests
"""

import numpy as np
from RP1_antibody_pipeline.utils.checkpoint_manager import CheckpointManager


def test_new_feature():
    """Test new feature functionality."""
    # Setup
    manager = CheckpointManager(checkpoint_dir="RP1_antibody_pipeline/experiments/checkpoints/test")

    # Test logic
    test_data = {"key": "value"}
    manager.save_stage("test_stage", test_data)
    loaded = manager.load_stage("test_stage")

    # Assertions
    assert loaded["key"] == "value", "Data mismatch"
    print("[OK] New feature test passed")


if __name__ == "__main__":
    test_new_feature()
```

### Running New Tests

```bash
# As script
python RP1_antibody_pipeline/tests/test_new_feature.py

# As module
python -m RP1_antibody_pipeline.tests.test_new_feature
```

## Test Best Practices

1. **Use test-specific directories**: Save test data to `checkpoints/test/`
2. **Clean up after tests**: Remove test data or document cleanup steps
3. **Verify data integrity**: Check that loaded data matches saved data
4. **Test edge cases**: Empty data, large arrays, special characters
5. **Document expected output**: Show what success looks like

## Continuous Integration

### Local Testing

```bash
# Run all tests before committing
python test_checkpoints.py
```

### Automated Testing

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    python test_checkpoints.py
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'RP1_antibody_pipeline'`

**Solution**: Run from project root or use module syntax:
```bash
python -m RP1_antibody_pipeline.tests.test_checkpoints
```

### Permission Errors

**Problem**: Cannot write to checkpoint directory

**Solution**: Check directory permissions or specify writable path:
```python
manager = CheckpointManager(checkpoint_dir="/writable/path")
```

### Test Data Accumulation

**Problem**: Test directory growing too large

**Solution**: Regular cleanup:
```bash
rm -rf RP1_antibody_pipeline/experiments/checkpoints/test/
```

## Documentation Links

### Checkpoint System
- **[../Docs/CHECKPOINTS.md](../Docs/CHECKPOINTS.md)** - Complete checkpoint documentation
- **[../Docs/CHECKPOINTS_DIRECTORY.md](../Docs/CHECKPOINTS_DIRECTORY.md)** - Directory structure
- **[../README.md](../README.md)** - Project overview

### API Reference
- **[../utils/checkpoint_manager.py](../utils/checkpoint_manager.py)** - CheckpointManager source
- **[../utils/analyze_checkpoints.py](../utils/analyze_checkpoints.py)** - Analysis tools

## Test Statistics

- **Total Tests**: 1 file (test_checkpoints.py)
- **Test Functions**: 1 (test_checkpoint_manager)
- **Assertions**: 3 integrity checks
- **Coverage**: Checkpoint system core functionality
- **Runtime**: ~2-5 seconds

## Contributing Tests

When adding new tests:

1. Create test file in this directory
2. Follow naming convention: `test_*.py`
3. Include docstrings
4. Add cleanup procedures
5. Update this README
6. Link to relevant documentation

## Support

- **Run tests**: `python test_checkpoints.py`
- **Check status**: All tests should pass with `[SUCCESS]`
- **Report issues**: See [../README.md](../README.md)

---

**Last Updated**: February 26, 2026
**Test Location**: `RP1_antibody_pipeline/tests/`
**Python Version**: 3.14.3
