# RP1 Antibody Pipeline Documentation

Complete documentation for the RP1 antibody discovery pipeline.

## Quick Navigation

- **[Main README](../README.md)** - Start here for project overview
- **[CHECKPOINTS.md](CHECKPOINTS.md)** - Complete checkpoint system guide
- **[RP1_summary.md](RP1_summary.md)** - Pipeline architecture details

## Documentation Index

### Getting Started

1. **[../README.md](../README.md)** - Project overview, quick start, installation
2. **[CHECKPOINTS.md](CHECKPOINTS.md)** - How to use the checkpoint system
3. **[RP1_summary.md](RP1_summary.md)** - Understand the pipeline architecture

### Checkpoint System

- **[CHECKPOINTS.md](CHECKPOINTS.md)** - Complete checkpoint documentation
  - Overview and benefits
  - Quick start guide
  - Command reference
  - Programmatic API
  - All 16 process milestones

- **[CHECKPOINTS_DIRECTORY.md](CHECKPOINTS_DIRECTORY.md)** - Directory structure details
  - File formats and layout
  - Loading and analyzing data
  - Data retention strategies
  - Cleanup procedures

- **[README_CHECKPOINTS.md](README_CHECKPOINTS.md)** - Quick reference
  - File locations
  - Basic commands
  - Common tasks

### Pipeline Architecture

- **[RP1_summary.md](RP1_summary.md)** - Pipeline design and architecture
  - Component descriptions
  - Stage-by-stage breakdown
  - Technical specifications

### Testing

- **[../tests/README.md](../tests/README.md)** - Test suite documentation
  - Running tests
  - Test coverage
  - Adding new tests

## By Use Case

### I want to run the pipeline

1. Read [../README.md - Quick Start](../README.md#quick-start)
2. Check [CHECKPOINTS.md - Running Pipeline](CHECKPOINTS.md#run-pipeline-with-checkpoints)

### I want to analyze checkpoint data

1. Read [CHECKPOINTS_DIRECTORY.md](CHECKPOINTS_DIRECTORY.md)
2. See [CHECKPOINTS.md - Programmatic Usage](CHECKPOINTS.md#programmatic-usage)

### I want to understand the architecture

1. Start with [RP1_summary.md](RP1_summary.md)

### I want to write tests

1. See [../tests/README.md](../tests/README.md)
2. Reference [CHECKPOINTS.md](CHECKPOINTS.md) for API details

### I want to extend the pipeline

1. Understand architecture: [RP1_summary.md](RP1_summary.md)
2. Add checkpoints: [CHECKPOINTS.md](CHECKPOINTS.md)

## Pipeline Stages Overview

The pipeline has 16 checkpoint stages:

| Stage | Milestone | Documentation |
|-------|-----------|---------------|
| 0 | Viral escape panel | [RP1_summary.md](RP1_summary.md) |
| 1 | BCR repertoire | [RP1_summary.md](RP1_summary.md) |
| 2 | LM scoring | [RP1_summary.md](RP1_summary.md) |
| 2a | Antigen-ALM profile | [RP1_summary.md](RP1_summary.md) |
| 2b | MD binding | [RP1_summary.md](RP1_summary.md) |
| 2c | ALM fine-tuning | [RP1_summary.md](RP1_summary.md) |
| 2d | Blind spots | [RP1_summary.md](RP1_summary.md) |
| 2.5 | Structural pathways | [RP1_summary.md](RP1_summary.md) |
| 3 | Structure (VAE/GAN) | [RP1_summary.md](RP1_summary.md) |
| 4 | MD + MSM | [RP1_summary.md](RP1_summary.md) |
| 5 | Synthetic evolution | [RP1_summary.md](RP1_summary.md) |
| 6 | Repertoire screening | [RP1_summary.md](RP1_summary.md) |
| 7 | Cross-reactivity | [RP1_summary.md](RP1_summary.md) |
| 8 | Vaccine design | [RP1_summary.md](RP1_summary.md) |
| 9 | Validation | [RP1_summary.md](RP1_summary.md) |
| 10 | Lab-in-the-loop | [RP1_summary.md](RP1_summary.md) |

## File Organization

```
Docs/
├── README.md                          # This file - documentation index
├── DOCUMENTATION_INDEX.md             # Complete navigation guide
│
├── CHECKPOINTS.md                     # Main checkpoint documentation
├── CHECKPOINTS_DIRECTORY.md           # Directory structure guide
├── README_CHECKPOINTS.md              # Quick reference
│
├── RP1_summary.md                     # Pipeline architecture
```

## Quick Commands

### Run Pipeline
```bash
# With checkpoints (default)
python -m RP1_antibody_pipeline.main --mock

# Without checkpoints
python -m RP1_antibody_pipeline.main --mock --no-checkpoints
```

### Analyze Checkpoints
```bash
# List runs
python RP1_antibody_pipeline/analyze_checkpoints.py list

# View stage
python RP1_antibody_pipeline/analyze_checkpoints.py summary <run_id> <stage_name>

# Compare runs
python RP1_antibody_pipeline/analyze_checkpoints.py compare <stage_name> <run_id_1> <run_id_2>
```

### Run Tests
```bash
python RP1_antibody_pipeline/test_checkpoints.py
```

## Key Features

### Checkpoint System
- Automatic saving at 16 milestones
- Resume from any stage
- Multiple data formats
- Analysis and comparison tools
- Full metadata tracking

### Pipeline
- 16-stage processing pipeline
- MD simulation integration
- Antibody language models
- Machine learning (VAE/GAN, MSM)
- Experimental validation
- Lab-in-the-loop optimization

## Support

- **Issues**: Check [../README.md](../README.md) for contact info
- **Tests**: Run `python RP1_antibody_pipeline/test_checkpoints.py` to verify
- **Documentation**: You're in the right place!

---

**Last Updated**: February 26, 2026
**Version**: 1.0.0
**Location**: `RP1_antibody_pipeline/Docs/`

For the most current information, always check the main [README.md](../README.md).
