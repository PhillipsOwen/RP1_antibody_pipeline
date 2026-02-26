# RP1 Antibody Pipeline - Checkpoint System Quick Reference

## Files

### Core Checkpoint System
- `RP1_antibody_pipeline/utils/checkpoint_manager.py` - Main checkpoint functionality
- `RP1_antibody_pipeline/main.py` - Pipeline with checkpoint integration

### Testing & Analysis Tools
- `RP1_antibody_pipeline/tests/test_checkpoints.py` - Test suite
- `RP1_antibody_pipeline/utils/analyze_checkpoints.py` - Analysis CLI tool

### Convenience Wrappers (in `RP1_antibody_pipeline/`)
- `RP1_antibody_pipeline/test_checkpoints.py` - Run tests from project root
- `RP1_antibody_pipeline/analyze_checkpoints.py` - Run analyzer from project root

### Checkpoint Storage
- `experiments/checkpoints/` - Default checkpoint directory

## Usage

### Running Tests

```bash
# From project root (MISM/)
python RP1_antibody_pipeline/test_checkpoints.py

# Or as module
python -m RP1_antibody_pipeline.tests.test_checkpoints
```

### Analyzing Checkpoints

```bash
# List all checkpoint runs
python RP1_antibody_pipeline/analyze_checkpoints.py list

# Show stage details
python RP1_antibody_pipeline/analyze_checkpoints.py summary <run_id> <stage_name>

# Compare runs
python RP1_antibody_pipeline/analyze_checkpoints.py compare <stage_name> <run_id_1> <run_id_2>

# Export data
python RP1_antibody_pipeline/analyze_checkpoints.py export <run_id> <stage_name>

# As module (from anywhere)
python -m RP1_antibody_pipeline.utils.analyze_checkpoints list
```

### Running Pipeline

```bash
# With checkpoints (default)
python -m RP1_antibody_pipeline.main --mock

# Without checkpoints
python -m RP1_antibody_pipeline.main --mock --no-checkpoints
```

## Directory Structure

```
MISM/
└── RP1_antibody_pipeline/
    ├── main.py                            # Pipeline entry point
    ├── test_checkpoints.py                # Wrapper: run tests
    ├── analyze_checkpoints.py             # Wrapper: run analyzer
    │
    ├── utils/
    │   ├── checkpoint_manager.py          # Core checkpoint system
    │   └── analyze_checkpoints.py         # Analysis CLI tool
    │
    ├── tests/
    │   ├── __init__.py
    │   └── test_checkpoints.py            # Test suite
    │
    ├── experiments/
    │   └── checkpoints/                   # Checkpoint storage (default)
    │       ├── README.md
    │       └── <timestamp_runs>/
    │
    └── Docs/
        └── CHECKPOINTS.md                 # Full documentation
```

## What Was Implemented

### 16 Process Milestones

Checkpoints are automatically saved after:
- Stage 0: Viral escape panel
- Stage 1: BCR repertoire & atlas
- Stage 2: LM scoring
- Stage 2a: Antigen-ALM binding profile
- Stage 2b: MD binding prediction
- Stage 2c: ALM fine-tuning
- Stage 2d: Immune blind spots
- Stage 2.5: Ag-Ab structural pathways
- Stage 3: Structural modeling
- Stage 4: MD + MSM
- Stage 5: Synthetic evolution
- Stage 6: Repertoire screening
- Stage 7: Cross-reactivity
- Stage 8: Vaccine design
- Stage 9: Validation
- Stage 10: Lab-in-the-loop

### Key Features

- **Automatic saving**: Checkpoints saved by default
- **Multiple formats**: NumPy, JSON, pickle, text
- **Metadata tracking**: Timestamps, configs, statistics
- **Easy loading**: Simple API for checkpoint access
- **Analysis tools**: CLI for comparing and exporting
- **Flexible storage**: Custom checkpoint directories

## Python API Examples

### Load Checkpoint

```python
from RP1_antibody_pipeline.utils.checkpoint_manager import CheckpointManager

manager = CheckpointManager()
runs = manager.list_runs()
data = manager.load_stage("stage_2_lm_scoring", run_id=runs[0])
```

### Analyze Binding Data

```python
manager = CheckpointManager()
stage_2b = manager.load_stage("stage_2b_md_binding", run_id="20260226_143052")
binding_matrix = stage_2b['binding_matrix']
print(f"Binding shape: {binding_matrix.shape}")
```

### Compare Runs

```python
for run_id in manager.list_runs():
    stage_5 = manager.load_stage("stage_5_evolution", run_id=run_id)
    print(f"{run_id}: {len(stage_5['evolved_sequences'])} sequences")
```

## For More Information

See `Docs/CHECKPOINTS.md` for comprehensive documentation including:
- Detailed command reference
- Advanced usage examples
- Troubleshooting guide
- Maintenance procedures

---

**Implementation Date**: February 26, 2026
**Status**: Complete and integrated into codebase
