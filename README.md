# RP1 Antibody Discovery Pipeline

A comprehensive computational pipeline for predicting antibody responses to viral escape mutants, combining molecular dynamics (MD) simulations, antibody language models (ALMs), and experimental validation.

## Overview

The RP1 pipeline integrates:
- **Viral escape prediction**: Generate and analyze escape mutant panels
- **BCR repertoire analysis**: Atlas construction and immune response profiling
- **Antibody language models**: Score and generate candidate sequences
- **MD simulations**: Predict binding affinities and structural pathways
- **Machine learning**: VAE/GAN for structural modeling, MSM for dynamics
- **Synthetic evolution**: Optimize candidates for broad neutralization
- **Experimental validation**: Lab-in-the-loop refinement

## Features

### Pipeline Capabilities
- 16-stage pipeline with automatic checkpointing
- Cross-reactivity analysis against escape panels
- Vaccine candidate design for broad coverage
- Integration with experimental validation data
- Lab-in-the-loop iterative optimization

### Checkpoint System
- **Automatic data saving** at each process milestone
- **Resume from failures** without re-running completed stages
- **16 checkpoint stages** capturing all intermediate results
- **Analysis tools** for comparing runs and exploring data

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MISM/RP1_antibody_pipeline

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run with mock models (fast, no GPU required)
python -m RP1_antibody_pipeline.main --mock

# Run full pipeline
python -m RP1_antibody_pipeline.main

# Run without checkpoints (for production)
python -m RP1_antibody_pipeline.main --no-checkpoints

# Custom checkpoint location
python -m RP1_antibody_pipeline.main --checkpoint-dir /path/to/checkpoints
```

### Analyzing Checkpoints

```bash
# List all checkpoint runs
python RP1_antibody_pipeline/analyze_checkpoints.py list

# Show stage details
python RP1_antibody_pipeline/analyze_checkpoints.py summary <run_id> stage_2_lm_scoring

# Compare multiple runs
python RP1_antibody_pipeline/analyze_checkpoints.py compare stage_5_evolution <run_id_1> <run_id_2>

# Export data for analysis
python RP1_antibody_pipeline/analyze_checkpoints.py export <run_id> stage_2b_md_binding --output ./data
```

### Running Tests

```bash
# Run checkpoint tests
python RP1_antibody_pipeline/test_checkpoints.py

# Or as module
python -m RP1_antibody_pipeline.tests.test_checkpoints
```

## Project Structure

```
RP1_antibody_pipeline/
├── main.py                      # Main pipeline entry point
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
│
├── data/                        # Input data files
│   └── SARS-CoV-2_sequences.fasta
│
├── experiments/                 # Experimental components
│   ├── checkpoints/             # Pipeline checkpoints (default location)
│   ├── output/                  # Experiment outputs
│   ├── lab_loop.py              # Lab-in-the-loop integration
│   └── validation.py            # Experimental validation
│
├── models/                      # Machine learning models
│   ├── antibody_lm.py           # Antibody language model
│   ├── alm_finetuner.py         # ALM fine-tuning
│   ├── vae.py                   # Variational autoencoder
│   └── gan.py                   # Generative adversarial network
│
├── md_simulations/              # Molecular dynamics
│   ├── md_runner.py             # MD simulation runner
│   ├── binding_md.py            # Binding affinity prediction
│   └── structural_pathways.py   # Pathway analysis
│
├── msm_analysis/                # Markov state models
│   └── msm_builder.py           # MSM construction
│
├── synthetic_evolution/         # Evolutionary optimization
│   └── evolution.py             # Genetic algorithms
│
├── viral_escape/                # Escape mutant analysis
│   └── escape_panel.py          # Panel generation
│
├── utils/                       # Utility modules
│   ├── checkpoint_manager.py    # Checkpoint system
│   ├── analyze_checkpoints.py   # Analysis tools
│   └── helpers.py               # Helper functions
│
├── tests/                       # Test suite
│   ├── README.md
│   └── test_checkpoints.py
│
└── Docs/                        # Documentation
    ├── README.md                # Documentation index
    ├── DOCUMENTATION_INDEX.md   # Complete guide
    ├── CHECKPOINTS.md           # Checkpoint system
    ├── CHECKPOINTS_DIRECTORY.md # Directory details
    └── RP1_SUMMARY.md           # Pipeline architecture
```

## Pipeline Stages

The pipeline consists of 16 major stages with automatic checkpointing:

| Stage | Name | Description |
|-------|------|-------------|
| 0 | Viral Escape Panel | Generate escape mutant variants |
| 1 | BCR Repertoire | Load repertoire and build immune atlas |
| 2 | LM Scoring | Score candidates with antibody language model |
| 2a | Antigen-ALM Profile | Compute binding site profiles |
| 2b | MD Binding | Predict binding with MD simulations |
| 2c | ALM Fine-tuning | Fine-tune model with MD guidance |
| 2d | Blind Spot Analysis | Identify immune coverage gaps |
| 2.5 | Structural Pathways | Analyze Ag-Ab binding pathways |
| 3 | Structural Modeling | VAE/GAN latent space embedding |
| 4 | MD + MSM | Build Markov state models from MD |
| 5 | Synthetic Evolution | Evolve candidates for optimization |
| 6 | Repertoire Screening | Screen at repertoire scale |
| 7 | Cross-reactivity | Test against escape panel |
| 8 | Vaccine Design | Select broadly neutralizing candidates |
| 9 | Experimental Validation | Compare with lab data |
| 10 | Lab-in-the-Loop | Incorporate experimental feedback |

## Configuration

Pipeline settings are defined in `config.py`. Key configuration sections:

- **ViralEscapeConfig**: Escape panel generation parameters
- **BCRConfig**: B-cell repertoire loading settings
- **AntibodyLMConfig**: Language model configuration
- **VAEConfig**: Variational autoencoder settings
- **MSMConfig**: Markov state model parameters
- **EvolutionConfig**: Genetic algorithm settings

## Checkpoint System

The pipeline automatically saves intermediate results at each stage.

### Default Location
```
experiments/checkpoints/
```

### Features
- **Automatic saving**: All stages checkpointed by default
- **Multiple formats**: NumPy arrays, JSON, pickle, text
- **Metadata tracking**: Timestamps, configs, statistics
- **Resume capability**: Restart from any stage
- **Analysis tools**: Compare runs and export data

### Programmatic Access

```python
from RP1_antibody_pipeline.utils.checkpoint_manager import CheckpointManager

# Load checkpoint data
manager = CheckpointManager()
runs = manager.list_runs()
data = manager.load_stage("stage_2_lm_scoring", run_id=runs[0])

# Access results
sequences = data['sequences']
scores = data['lm_scores']
```

## Documentation

Complete documentation is available in the `Docs/` directory:

- **[Docs/README.md](Docs/README.md)** - Documentation index
- **[Docs/CHECKPOINTS.md](Docs/CHECKPOINTS.md)** - Checkpoint system guide
- **[Docs/RP1_SUMMARY.md](Docs/RP1_SUMMARY.md)** - Pipeline architecture details
- **[Docs/DOCUMENTATION_INDEX.md](Docs/DOCUMENTATION_INDEX.md)** - Complete navigation

## Requirements

### Core Dependencies
- Python 3.14+
- NumPy, SciPy
- PyTorch (for language models)
- Biopython (for sequence analysis)
- MDTraj (for MD analysis)
- PyEMMA (for MSM)

See `requirements.txt` for complete list.

### Optional
- CUDA-compatible GPU (for full language models)
- OpenMM (for MD simulations)

## Testing

```bash
# Run all tests
python RP1_antibody_pipeline/test_checkpoints.py

# Test individual components
python -m RP1_antibody_pipeline.tests.test_checkpoints
```

## Development

### Virtual Environment

Located at: `../.venv/` (project root)
Python version: 3.14.3

```bash
# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### Adding New Stages

1. Implement stage function in appropriate module
2. Add checkpoint save function in `utils/checkpoint_manager.py`
3. Integrate into `main.py` pipeline
4. Add tests in `tests/`
5. Update documentation in `Docs/`

## Performance

- **Mock mode**: ~2-5 minutes (no GPU required)
- **Full pipeline**: 1-4 hours (depends on GPU and dataset size)
- **Checkpoint overhead**: ~1-2 seconds per stage
- **Disk usage**: 100-500 MB per run (with checkpoints)

## Citation

If you use this pipeline in your research, please cite:

```
[Citation to be added]
```

## License

[License information to be added]

## Support

- **Issues**: Report bugs and request features via GitHub issues
- **Documentation**: See `Docs/` directory
- **Tests**: Run `python RP1_antibody_pipeline/test_checkpoints.py` to verify installation

## Version

- **Pipeline Version**: 1.0.0
- **Last Updated**: February 26, 2026
- **Python**: 3.14.3

## Contact

[Contact information to be added]

---

**Quick Links**:
- [Documentation](Docs/README.md)
- [Checkpoint Guide](Docs/CHECKPOINTS.md)
- [Pipeline Architecture](Docs/RP1_SUMMARY.md)
- [Test Suite](tests/README.md)
