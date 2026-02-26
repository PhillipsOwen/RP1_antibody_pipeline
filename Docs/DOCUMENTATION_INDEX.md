# RP1 Antibody Pipeline - Documentation Index

Complete documentation index for the RP1 antibody discovery pipeline.

## Documentation Location

All documentation is centralized in the **Docs/** directory:
```
RP1_antibody_pipeline/Docs/
```

## Checkpoint System Documentation

### Primary Documents

1. **[CHECKPOINTS.md](CHECKPOINTS.md)**
   - Complete checkpoint system guide
   - Quick start and usage examples
   - Programmatic API reference
   - Command reference
   - 16 process milestones detailed

2. **[CHECKPOINTS_DIRECTORY.md](CHECKPOINTS_DIRECTORY.md)**
   - Detailed directory structure
   - File format specifications
   - Loading and analyzing checkpoints
   - Data retention and cleanup strategies
   - All 16 stages with outputs

3. **[README_CHECKPOINTS.md](README_CHECKPOINTS.md)**
   - Quick reference guide
   - File structure overview
   - Basic commands
   - Testing procedures

### Supporting Documents

- **[../tests/README.md](../tests/README.md)** - Test suite documentation
- **[../experiments/checkpoints/README.md](../experiments/checkpoints/README.md)** - Checkpoint directory info

## Pipeline Documentation

### Architecture and Design

1. **[RP1_summary.md](RP1_summary.md)**
   - Pipeline architecture overview
   - Component descriptions
   - Stage-by-stage breakdown
   - Technical specifications

## Documentation Structure

```
RP1_antibody_pipeline/
├── README.md                                # Main project readme
│
├── Docs/                                    # All documentation here
│   ├── README.md                            # Documentation index
│   ├── DOCUMENTATION_INDEX.md               # This file
│   ├── CHECKPOINTS.md                       # Checkpoint system guide
│   ├── CHECKPOINTS_DIRECTORY.md             # Directory details
│   ├── README_CHECKPOINTS.md                # Quick reference
│   ├── RP1_summary.md                       # Pipeline summary
│
├── tests/
│   └── README.md                            # Test suite docs
│
└── experiments/
    └── checkpoints/
        └── README.md                        # Checkpoint directory info
```

## Finding Documentation

### By Topic

**Checkpoints**:
- Overview: [CHECKPOINTS.md](CHECKPOINTS.md)
- Directory structure: [CHECKPOINTS_DIRECTORY.md](CHECKPOINTS_DIRECTORY.md)
- Quick reference: [README_CHECKPOINTS.md](README_CHECKPOINTS.md)

**Pipeline Architecture**:
- Summary: [RP1_summary.md](RP1_summary.md)

**Testing**:
- Test suite: [../tests/README.md](../tests/README.md)

### By Task

**Running the Pipeline**:
1. Read [CHECKPOINTS.md - Quick Start](CHECKPOINTS.md#quick-start)
2. Check [README_CHECKPOINTS.md](README_CHECKPOINTS.md) for commands

**Analyzing Checkpoints**:
1. See [CHECKPOINTS_DIRECTORY.md](CHECKPOINTS_DIRECTORY.md)
2. Reference [CHECKPOINTS.md - Commands Reference](CHECKPOINTS.md#commands-reference)

**Understanding the Pipeline**:
1. Start with [RP1_summary.md](RP1_summary.md)

**Writing Tests**:
1. See [../tests/README.md](../tests/README.md)
2. Reference [CHECKPOINTS.md - Programmatic Usage](CHECKPOINTS.md#programmatic-usage)

## Documentation Standards

All documentation follows these conventions:
- Markdown format (GitHub-flavored)
- Code blocks with language specification
- Clear section headers
- Command examples
- Links to related documents

## Quick Access

| Document | Purpose | Audience |
|----------|---------|----------|
| [CHECKPOINTS.md](CHECKPOINTS.md) | Complete checkpoint guide | All users |
| [CHECKPOINTS_DIRECTORY.md](CHECKPOINTS_DIRECTORY.md) | Directory structure | Developers |
| [README_CHECKPOINTS.md](README_CHECKPOINTS.md) | Quick reference | All users |
| [RP1_summary.md](RP1_summary.md) | Pipeline overview | Developers |
| [../tests/README.md](../tests/README.md) | Test documentation | Developers |

## Contributing Documentation

When adding new documentation:
1. Place files in `RP1_antibody_pipeline/Docs/`
2. Update this index
3. Add links from related documents
4. Follow existing naming conventions

---

**Last Updated**: February 26, 2026
**Documentation Location**: `RP1_antibody_pipeline/Docs/`
