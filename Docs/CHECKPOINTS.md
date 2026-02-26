# Checkpoint System Documentation

## Overview

The RP1 antibody pipeline includes an integrated checkpoint system that automatically saves intermediate data at 16 process milestones, enabling pipeline resumption, debugging, and analysis.

## File Structure

```
RP1_antibody_pipeline/
├── utils/
│   ├── checkpoint_manager.py            # Core checkpoint functionality
│   └── analyze_checkpoints.py           # Analysis and comparison tool
├── tests/
│   ├── __init__.py
│   └── test_checkpoints.py              # Test suite
├── test_checkpoints.py                  # Wrapper: run tests from project root
├── analyze_checkpoints.py               # Wrapper: run analyzer from project root
└── main.py                              # Pipeline with checkpoint integration

└── experiments/
    └── checkpoints/                     # Default checkpoint storage
        ├── README.md
        └── <run_id>/                    # Timestamped run directories
            ├── stage_0_escape_panel/
            ├── stage_1_bcr_repertoire/
            └── ...
```

## Quick Start

### Run Pipeline with Checkpoints

```bash
# From project root, checkpoints enabled by default
python -m RP1_antibody_pipeline.main --mock

# Disable checkpoints
python -m RP1_antibody_pipeline.main --mock --no-checkpoints

# Custom checkpoint directory
python -m RP1_antibody_pipeline.main --checkpoint-dir /path/to/checkpoints
```

### Test Checkpoint System

```bash
# From project root (wrapper script)
python RP1_antibody_pipeline/test_checkpoints.py

# Or as a module
python -m RP1_antibody_pipeline.tests.test_checkpoints
```

### Analyze Checkpoints

```bash
# From project root (wrapper script)
python RP1_antibody_pipeline/analyze_checkpoints.py list

# Or as a module
python -m RP1_antibody_pipeline.utils.analyze_checkpoints list

# Show stage details
python RP1_antibody_pipeline/analyze_checkpoints.py summary <run_id> stage_2_lm_scoring

# Compare runs
python RP1_antibody_pipeline/analyze_checkpoints.py compare stage_5_evolution <run_id_1> <run_id_2>

# Export data
python RP1_antibody_pipeline/analyze_checkpoints.py export <run_id> stage_2b_md_binding --output ./data
```

## Programmatic Usage

```python
from RP1_antibody_pipeline.utils.checkpoint_manager import CheckpointManager

# Initialize manager
manager = CheckpointManager()

# List available runs
runs = manager.list_runs()

# Load specific stage
stage_data = manager.load_stage("stage_2_lm_scoring", run_id=runs[0])
sequences = stage_data['sequences']
scores = stage_data['lm_scores']
```

## Process Milestones

The pipeline saves checkpoints at these 16 stages:

| Stage | Name | Key Outputs |
|-------|------|-------------|
| 0 | escape_panel | Viral escape mutant panel |
| 1 | bcr_repertoire | BCR sequences, atlas |
| 2 | lm_scoring | LM scores, sequences |
| 2a | antigen_profile | Affinity matrix |
| 2b | md_binding | Binding matrix |
| 2c | alm_finetune | Fine-tuned model |
| 2d | blind_spots | Coverage analysis |
| 2.5 | pathways | Pathway MSM, timescales |
| 3 | structure | Latent embeddings |
| 4 | msm | MSM model, timescales |
| 5 | evolution | Evolved sequences |
| 6 | screening | Top candidates |
| 7 | cross_reactivity | Coverage matrix |
| 8 | vaccine_design | Vaccine candidates |
| 9 | validation | Predictions vs experiments |
| 10 | lab_loop | Lab feedback |

## Data Formats

Each checkpoint contains:
- **metadata.json**: Stage info, timestamp, configuration
- **summary.json**: Statistics (shapes, means, ranges)
- **.npy**: NumPy arrays (matrices, vectors)
- **.txt**: Text files (sequences, one per line)
- **.json**: JSON dictionaries
- **.pkl**: Python objects (models)

## Commands Reference

### Test Commands
```bash
# Run checkpoint tests
python RP1_antibody_pipeline/test_checkpoints.py
python -m RP1_antibody_pipeline.tests.test_checkpoints
```

### Analysis Commands
```bash
# List all checkpoint runs
python RP1_antibody_pipeline/analyze_checkpoints.py list

# Show stage summary with statistics
python RP1_antibody_pipeline/analyze_checkpoints.py summary <run_id> <stage_name>

# Compare same stage across multiple runs
python RP1_antibody_pipeline/analyze_checkpoints.py compare <stage_name> <run_id_1> <run_id_2> ...

# Export stage data for external analysis
python RP1_antibody_pipeline/analyze_checkpoints.py export <run_id> <stage_name> --output <directory>

# Custom checkpoint directory
python RP1_antibody_pipeline/analyze_checkpoints.py --checkpoint-dir /path/to/checkpoints list
```

### Pipeline Commands
```bash
# Run with checkpoints (default)
python -m RP1_antibody_pipeline.main --mock

# Run without checkpoints
python -m RP1_antibody_pipeline.main --mock --no-checkpoints

# Custom checkpoint location
python -m RP1_antibody_pipeline.main --checkpoint-dir /custom/path
```

## Examples

### Load and Analyze Binding Matrix

```python
from RP1_antibody_pipeline.utils.checkpoint_manager import CheckpointManager
import numpy as np

manager = CheckpointManager()
run_id = manager.list_runs()[0]

# Load Stage 2b
stage_2b = manager.load_stage("stage_2b_md_binding", run_id=run_id)
binding_matrix = stage_2b['binding_matrix']

# Analyze
print(f"Binding matrix shape: {binding_matrix.shape}")
print(f"Mean binding: {binding_matrix.mean():.4f}")
print(f"Max binding: {binding_matrix.max():.4f}")

# Find top binders
top_idx = np.unravel_index(binding_matrix.argmax(), binding_matrix.shape)
print(f"Best binder: Ab[{top_idx[0]}] x Ag[{top_idx[1]}]")
```

### Compare Evolution Results

```python
from RP1_antibody_pipeline.utils.checkpoint_manager import CheckpointManager
import numpy as np

manager = CheckpointManager()

for run_id in manager.list_runs()[-3:]:  # Last 3 runs
    try:
        stage_5 = manager.load_stage("stage_5_evolution", run_id=run_id)
        scores = stage_5['scores']
        print(f"{run_id[:8]}: mean={np.mean(scores):.3f}, max={np.max(scores):.3f}")
    except FileNotFoundError:
        print(f"{run_id[:8]}: Stage 5 not found")
```

## Maintenance

### Clean Up Old Runs

```bash
# Remove test checkpoints
rm -rf RP1_antibody_pipeline/experiments/checkpoints/test/

# Remove a specific run
rm -rf RP1_antibody_pipeline/experiments/checkpoints/<run_id>/

# Archive important runs
tar -czf run_backup.tar.gz RP1_antibody_pipeline/experiments/checkpoints/<run_id>/
```

### Check Disk Usage

```bash
# Total checkpoint size
du -sh RP1_antibody_pipeline/experiments/checkpoints/

# Size per run
du -sh RP1_antibody_pipeline/experiments/checkpoints/*/
```

## Troubleshooting

**Issue**: "FileNotFoundError: Checkpoint not found"
- **Solution**: Verify stage exists with `python RP1_antibody_pipeline/analyze_checkpoints.py list`

**Issue**: Out of disk space
- **Solution**: Use `--no-checkpoints` flag or clean up old runs

**Issue**: Import errors
- **Solution**: Ensure you're running from the project root (`MISM/`) and that the virtual environment is activated

## Benefits

1. **Resume from failures**: Continue pipeline from any stage
2. **Debug efficiently**: Inspect intermediate outputs
3. **Compare experiments**: Analyze results across hyperparameters
4. **Reproducibility**: Complete execution records with configs
5. **Collaboration**: Share checkpoints with team members

## Additional Documentation

- `Docs/CHECKPOINTS_DIRECTORY.md` - Detailed checkpoint directory structure
- `utils/checkpoint_manager.py` - Source code documentation
- `utils/analyze_checkpoints.py` - Analysis tool source
- `tests/test_checkpoints.py` - Test suite source

---

**Last Updated**: February 26, 2026
