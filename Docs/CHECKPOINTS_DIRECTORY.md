# Pipeline Checkpoints

This directory contains intermediate data files saved at each process milestone during pipeline execution.

## Directory Structure

Each pipeline run creates a unique subdirectory with timestamp:
```
experiments/checkpoints/
└── 20260226_143052/          # Run ID: YYYYMMDD_HHMMSS
    ├── stage_0_escape_panel/
    │   ├── metadata.json      # Stage metadata and config
    │   ├── summary.json       # Summary statistics
    │   ├── escape_panel.pkl   # Escape mutant objects
    │   └── epitope_residues.json
    ├── stage_1_bcr_repertoire/
    │   ├── metadata.json
    │   ├── summary.json
    │   ├── sequences.txt      # One sequence per line
    │   ├── atlas_centroid.npy
    │   └── atlas_covariance.npy
    ├── stage_2_lm_scoring/
    ├── stage_2a_antigen_profile/
    ├── stage_2b_md_binding/
    ├── stage_2c_alm_finetune/
    ├── stage_2d_blind_spots/
    ├── stage_2_5_pathways/
    ├── stage_3_structure/
    ├── stage_4_msm/
    ├── stage_5_evolution/
    ├── stage_6_screening/
    ├── stage_7_cross_reactivity/
    ├── stage_8_vaccine_design/
    ├── stage_9_validation/
    └── stage_10_lab_loop/
```

## File Formats

- **metadata.json**: Stage information, timestamp, configuration snapshot
- **summary.json**: Quick statistics about the data (shapes, counts, ranges)
- **.npy**: NumPy arrays (matrices, vectors)
- **.txt**: Plain text files (one item per line, typically sequences)
- **.json**: JSON-serializable dictionaries
- **.pkl**: Python pickle files (complex objects, models)

## Usage

### Running with Checkpoints (default)
```bash
python -m RP1_antibody_pipeline.main --mock
```

Checkpoints will be saved to: `experiments/checkpoints/<timestamp>/`

### Running without Checkpoints
```bash
python -m RP1_antibody_pipeline.main --mock --no-checkpoints
```

### Custom Checkpoint Directory
```bash
python -m RP1_antibody_pipeline.main --checkpoint-dir /path/to/checkpoints
```

### Loading Checkpoints Programmatically

```python
from RP1_antibody_pipeline.utils.checkpoint_manager import CheckpointManager

# Initialize manager (uses RP1_antibody_pipeline/experiments by default)
manager = CheckpointManager()

# List available runs
runs = manager.list_runs()
print(f"Available runs: {runs}")

# Load specific stage from a run
run_id = runs[0]  # Most recent run
stage_data = manager.load_stage("stage_2_lm_scoring", run_id=run_id)

# Access loaded data
sequences = stage_data['sequences']
scores = stage_data['lm_scores']

# List all stages in a run
stages = manager.list_stages(run_id)
print(f"Stages in run {run_id}: {stages}")
```

### Using the Analysis Tool

```bash
# List all runs
python RP1_antibody_pipeline/analyze_checkpoints.py list

# Show stage details
python RP1_antibody_pipeline/analyze_checkpoints.py summary <run_id> stage_2_lm_scoring

# Compare runs
python RP1_antibody_pipeline/analyze_checkpoints.py compare stage_5_evolution <run_id_1> <run_id_2>

# Export data
python RP1_antibody_pipeline/analyze_checkpoints.py export <run_id> stage_2b_md_binding --output ./exported
```

## Process Milestones (16 Stages)

| Stage | Name | Key Outputs |
|-------|------|-------------|
| 0 | escape_panel | Viral escape mutant panel |
| 1 | bcr_repertoire | BCR sequences, atlas |
| 2 | lm_scoring | LM scores |
| 2a | antigen_profile | Affinity matrix |
| 2b | md_binding | Binding matrix |
| 2c | alm_finetune | Fine-tuned model |
| 2d | blind_spots | Coverage gaps |
| 2.5 | pathways | Pathway MSM |
| 3 | structure | Latent embeddings |
| 4 | msm | MSM model |
| 5 | evolution | Evolved sequences |
| 6 | screening | Top candidates |
| 7 | cross_reactivity | Coverage matrix |
| 8 | vaccine_design | Vaccine candidates |
| 9 | validation | Predictions vs experiments |
| 10 | lab_loop | Lab feedback |

## Benefits

1. **Resume from failures**: Restart pipeline from any stage without re-running earlier stages
2. **Debugging**: Inspect intermediate results to diagnose issues
3. **Analysis**: Compare outputs across different runs and hyperparameters
4. **Reproducibility**: Full record of pipeline execution with timestamps and configs
5. **Efficiency**: Skip expensive stages during development/testing

## Data Retention

Checkpoints can consume significant disk space. Consider:
- Cleaning up old runs periodically
- Using `--no-checkpoints` flag for production runs where intermediates aren't needed
- Archiving important runs to external storage

### Clean Up Commands

```bash
# Remove test checkpoints
rm -rf experiments/checkpoints/test/

# Remove a specific run
rm -rf experiments/checkpoints/20260226_143052/

# Archive a run
tar -czf run_20260226_143052.tar.gz experiments/checkpoints/20260226_143052/
```

---

For more information, see:
- `RP1_antibody_pipeline/Docs/CHECKPOINTS.md` - Comprehensive documentation
- `RP1_antibody_pipeline/utils/checkpoint_manager.py` - Source code
