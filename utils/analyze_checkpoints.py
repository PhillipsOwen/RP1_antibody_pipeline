"""
analyze_checkpoints.py - Utility script to analyze and compare pipeline checkpoints.

This script provides tools to:
- List all checkpoint runs
- Display summary statistics for each stage
- Compare results across multiple runs
- Export checkpoint data for further analysis
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from .checkpoint_manager import CheckpointManager


def list_all_runs(checkpoint_dir: str = "experiments/checkpoints"):
    """List all available pipeline runs with their stages."""
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    runs = manager.list_runs()

    if not runs:
        print("No checkpoint runs found.")
        return

    print(f"Found {len(runs)} pipeline runs:\n")

    for run_id in sorted(runs, reverse=True):
        stages = manager.list_stages(run_id=run_id)
        print(f"  Run ID: {run_id}")
        print(f"  Stages: {len(stages)}")
        print(f"  Path:   {Path(checkpoint_dir) / run_id}")

        # Try to read stage 0 metadata if it exists
        if "stage_0_escape_panel" in stages:
            try:
                meta_path = Path(checkpoint_dir) / run_id / "stage_0_escape_panel" / "metadata.json"
                with open(meta_path) as f:
                    meta = json.load(f)
                    timestamp = meta.get("timestamp", "unknown")
                    print(f"  Time:   {timestamp}")
            except Exception:
                pass

        print()


def show_stage_summary(run_id: str, stage_name: str,
                      checkpoint_dir: str = "experiments/checkpoints"):
    """Display summary statistics for a specific stage."""
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    # Read metadata
    meta_path = Path(checkpoint_dir) / run_id / stage_name / "metadata.json"
    summary_path = Path(checkpoint_dir) / run_id / stage_name / "summary.json"

    if not meta_path.exists():
        print(f"Stage {stage_name} not found in run {run_id}")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\n=== Stage: {stage_name} ===")
    print(f"Run ID:    {run_id}")
    print(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
    print(f"Description: {metadata.get('description', 'N/A')}")

    if "config" in metadata:
        print(f"\nConfiguration:")
        for key, value in metadata["config"].items():
            print(f"  {key}: {value}")

    print(f"\nData Summary:")
    for key, info in summary.items():
        if info.get("type") == "ndarray":
            shape = tuple(info["shape"])
            print(f"  {key}:")
            print(f"    Shape: {shape}")
            print(f"    Mean:  {info.get('mean', 'N/A'):.4f}" if info.get('mean') is not None else f"    Mean:  N/A")
            print(f"    Std:   {info.get('std', 'N/A'):.4f}" if info.get('std') is not None else f"    Std:   N/A")
            print(f"    Range: [{info.get('min', 'N/A'):.4f}, {info.get('max', 'N/A'):.4f}]" if info.get('min') is not None else f"    Range: N/A")
        elif info.get("type") == "list":
            print(f"  {key}:")
            print(f"    Length: {info['length']}")
            if "sample" in info:
                sample = info["sample"]
                if len(sample) > 60:
                    sample = sample[:60] + "..."
                print(f"    Sample: {sample}")
        elif info.get("type") == "dict":
            print(f"  {key}:")
            print(f"    Keys: {', '.join(info['keys'])}")


def compare_runs(run_ids: List[str], stage_name: str,
                checkpoint_dir: str = "experiments/checkpoints"):
    """Compare the same stage across multiple runs."""
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    print(f"\n=== Comparing Stage: {stage_name} ===\n")

    results = {}

    for run_id in run_ids:
        try:
            summary_path = Path(checkpoint_dir) / run_id / stage_name / "summary.json"
            with open(summary_path) as f:
                results[run_id] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Stage {stage_name} not found in run {run_id}")
            continue

    if not results:
        print("No data to compare.")
        return

    # Compare numeric fields
    print(f"{'Metric':<30} | " + " | ".join(f"{run_id[-8:]}" for run_id in results.keys()))
    print("-" * 80)

    # Gather all keys
    all_keys = set()
    for summary in results.values():
        all_keys.update(summary.keys())

    for key in sorted(all_keys):
        values = []
        for run_id in results.keys():
            if key in results[run_id]:
                info = results[run_id][key]
                if info.get("type") == "ndarray" and "mean" in info:
                    values.append(f"{info['mean']:.4f}")
                elif info.get("type") == "list":
                    values.append(f"{info['length']}")
                else:
                    values.append("-")
            else:
                values.append("N/A")

        print(f"{key:<30} | " + " | ".join(f"{v:>8}" for v in values))


def export_stage_data(run_id: str, stage_name: str, output_dir: str = ".",
                     checkpoint_dir: str = "experiments/checkpoints"):
    """Export checkpoint data for external analysis."""
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    try:
        data = manager.load_stage(stage_name, run_id=run_id)
    except FileNotFoundError:
        print(f"Error: Stage {stage_name} not found in run {run_id}")
        return

    output_path = Path(output_dir) / f"{run_id}_{stage_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting stage data to: {output_path}")

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.save(output_path / f"{key}.npy", value)
            print(f"  Saved: {key}.npy (shape: {value.shape})")
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            with open(output_path / f"{key}.txt", "w") as f:
                f.write("\n".join(value))
            print(f"  Saved: {key}.txt ({len(value)} lines)")
        elif isinstance(value, dict):
            with open(output_path / f"{key}.json", "w") as f:
                json.dump(value, f, indent=2, default=str)
            print(f"  Saved: {key}.json")

    print(f"\nExport complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare RP1 antibody pipeline checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="experiments/checkpoints",
        help="Checkpoint directory"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    subparsers.add_parser("list", help="List all checkpoint runs")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show stage summary")
    summary_parser.add_argument("run_id", help="Run ID")
    summary_parser.add_argument("stage", help="Stage name (e.g., stage_2_lm_scoring)")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare stage across runs")
    compare_parser.add_argument("stage", help="Stage name")
    compare_parser.add_argument("run_ids", nargs="+", help="Run IDs to compare")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export stage data")
    export_parser.add_argument("run_id", help="Run ID")
    export_parser.add_argument("stage", help="Stage name")
    export_parser.add_argument("--output", default=".", help="Output directory")

    args = parser.parse_args()

    if args.command == "list":
        list_all_runs(args.checkpoint_dir)
    elif args.command == "summary":
        show_stage_summary(args.run_id, args.stage, args.checkpoint_dir)
    elif args.command == "compare":
        compare_runs(args.run_ids, args.stage, args.checkpoint_dir)
    elif args.command == "export":
        export_stage_data(args.run_id, args.stage, args.output, args.checkpoint_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
