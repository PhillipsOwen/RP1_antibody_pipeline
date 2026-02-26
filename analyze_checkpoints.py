"""
Convenience wrapper to run checkpoint analyzer.

This script must be run from the project root (MISM/) directory:
    cd /path/to/MISM
    python RP1_antibody_pipeline/analyze_checkpoints.py list
    python RP1_antibody_pipeline/analyze_checkpoints.py summary <run_id> <stage>

Or as a module:
    python -m RP1_antibody_pipeline.analyze_checkpoints list
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to Python path if not already there
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from RP1_antibody_pipeline.utils.analyze_checkpoints import main
    main()
