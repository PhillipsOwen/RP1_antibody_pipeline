"""
Convenience wrapper to run checkpoint tests.

This script must be run from the project root (MISM/) directory:
    cd /path/to/MISM
    python RP1_antibody_pipeline/test_checkpoints.py

Or as a module:
    python -m RP1_antibody_pipeline.test_checkpoints
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to Python path if not already there
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from RP1_antibody_pipeline.tests.test_checkpoints import test_checkpoint_manager

    run_dir = test_checkpoint_manager()
    print(f"\nTo clean up test data, delete: {run_dir}")
