"""
test_checkpoints.py - Quick test of the checkpoint manager functionality.

Run this to verify checkpoint saving/loading works correctly.
"""

import numpy as np
from RP1_antibody_pipeline.utils.checkpoint_manager import CheckpointManager


def test_checkpoint_manager():
    """Test basic checkpoint save/load functionality."""
    print("Testing CheckpointManager...")

    # Create a test checkpoint manager
    manager = CheckpointManager(checkpoint_dir="RP1_antibody_pipeline/experiments/checkpoints/test")
    print(f"[OK] Initialized CheckpointManager: {manager.run_dir}")

    # Test Stage 0: Save escape panel data
    test_data_0 = {
        "escape_panel": ["mutant_1", "mutant_2", "mutant_3"],
        "panel_size": 3,
        "epitope_residues": [417, 445, 484, 501],
    }
    stage_0_path = manager.save_stage("test_stage_0", test_data_0,
                                      metadata={"test": "stage_0"})
    print(f"[OK] Saved Stage 0 checkpoint: {stage_0_path}")

    # Test Stage 2: Save LM scoring data
    test_sequences = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKG",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVRQAPGQG",
        "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKG",
    ]
    test_scores = np.array([0.85, 0.92, 0.78])
    test_data_2 = {
        "sequences": test_sequences,
        "lm_scores": test_scores,
        "n_sequences": len(test_sequences),
    }
    stage_2_path = manager.save_stage("test_stage_2", test_data_2,
                                      metadata={"test": "stage_2"})
    print(f"[OK] Saved Stage 2 checkpoint: {stage_2_path}")

    # Test Stage 2b: Save binding matrix
    test_binding_matrix = np.random.rand(3, 2)  # 3 antibodies x 2 antigens
    test_data_2b = {
        "binding_matrix": test_binding_matrix,
        "antibody_sequences": test_sequences,
        "antigen_sequences": ["ANTIGEN_A", "ANTIGEN_B"],
    }
    stage_2b_path = manager.save_stage("test_stage_2b", test_data_2b,
                                       metadata={"test": "stage_2b"})
    print(f"[OK] Saved Stage 2b checkpoint: {stage_2b_path}")

    # List all stages
    stages = manager.list_stages()
    print(f"[OK] Saved stages: {stages}")

    # Test loading Stage 2
    loaded_stage_2 = manager.load_stage("test_stage_2")
    print(f"[OK] Loaded Stage 2 checkpoint")

    # Verify loaded data
    assert loaded_stage_2["n_sequences"] == 3, "Sequence count mismatch"
    assert len(loaded_stage_2["sequences"]) == 3, "Sequences list mismatch"
    assert np.allclose(loaded_stage_2["lm_scores"], test_scores), "Scores mismatch"
    print(f"[OK] Data integrity verified")
    print(f"  - Sequences: {len(loaded_stage_2['sequences'])}")
    print(f"  - Scores: {loaded_stage_2['lm_scores']}")

    # Test loading Stage 2b
    loaded_stage_2b = manager.load_stage("test_stage_2b")
    print(f"[OK] Loaded Stage 2b checkpoint")
    assert np.allclose(loaded_stage_2b["binding_matrix"], test_binding_matrix), \
        "Binding matrix mismatch"
    print(f"  - Binding matrix shape: {loaded_stage_2b['binding_matrix'].shape}")

    # List all runs
    all_runs = manager.list_runs()
    print(f"[OK] All runs: {all_runs}")

    print("\n[SUCCESS] All tests passed!")
    print(f"\nCheckpoint location: {manager.run_dir}")
    print("You can inspect the saved files manually.")

    return manager.run_dir


if __name__ == "__main__":
    run_dir = test_checkpoint_manager()
    print(f"\nTo clean up test data, delete: {run_dir}")
