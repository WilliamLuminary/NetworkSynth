"""Re-plot hybrid snapshot PNGs from saved .npy data.

Scans a snapshots directory for *_positions.npy / *_edges.npy pairs
and renders any that are missing a corresponding .png (or all, with
--force).  Useful for re-plotting with different visual settings
without re-running the full pipeline.

Usage:
    python scripts/helpers/plot_hybrid_snapshots.py <snapshots_dir>
    python scripts/helpers/plot_hybrid_snapshots.py <snapshots_dir> --dpi 300
    python scripts/helpers/plot_hybrid_snapshots.py <snapshots_dir> --force
    python scripts/helpers/plot_hybrid_snapshots.py <snapshots_dir> --workers 8

Examples:
    # Re-plot missing PNGs from an existing run:
    python scripts/helpers/plot_hybrid_snapshots.py \\
        data/output/hybrid_mode_.../sample_A/snapshots

    # Re-plot ALL with custom style:
    python scripts/helpers/plot_hybrid_snapshots.py \\
        data/output/hybrid_mode_.../sample_A/snapshots \\
        --force --dpi 300 --node-size 0.05 --line-width 0.2
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def find_snapshot_pairs(snapshot_dir):
    """Return sorted list of (index, pos_path, edge_path) tuples."""
    pattern = re.compile(r"^snapshot_(\d+)_positions\.npy$")
    pairs = []
    for fname in os.listdir(snapshot_dir):
        m = pattern.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        pos_path = os.path.join(snapshot_dir, fname)
        edge_path = os.path.join(snapshot_dir, f"snapshot_{idx:05d}_edges.npy")
        if os.path.isfile(edge_path):
            pairs.append((idx, pos_path, edge_path))
    pairs.sort(key=lambda t: t[0])
    return pairs


def plot_one(idx, pos_path, edge_path, snapshot_dir, frame, style):
    import numpy as np

    from utils import save_hybrid_snapshot

    positions = np.load(pos_path).tolist()
    edges = np.load(edge_path).tolist()
    save_hybrid_snapshot(positions, edges, frame, idx, snapshot_dir, **style)


def main():
    parser = argparse.ArgumentParser(
        description="Re-plot hybrid snapshot PNGs from .npy data"
    )
    parser.add_argument("snapshot_dir", help="Path to the snapshots/ directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-plot all snapshots, even if PNG already exists",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--node-size", type=float, default=0.01)
    parser.add_argument("--line-width", type=float, default=0.1)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel threads (default: cpu_count)",
    )
    args = parser.parse_args()

    snapshot_dir = os.path.abspath(args.snapshot_dir)
    if not os.path.isdir(snapshot_dir):
        sys.exit(f"Directory not found: {snapshot_dir}")

    pairs = find_snapshot_pairs(snapshot_dir)
    if not pairs:
        sys.exit(f"No *_positions.npy / *_edges.npy pairs found in {snapshot_dir}")

    if not args.force:
        pairs = [
            (idx, pp, ep)
            for idx, pp, ep in pairs
            if not os.path.isfile(os.path.join(snapshot_dir, f"snapshot_{idx:05d}.png"))
        ]
        if not pairs:
            print("All PNGs already exist. Use --force to re-plot.")
            return

    import numpy as np

    all_pos = np.load(pairs[-1][1])
    x_min, y_min = all_pos.min(axis=0)
    x_max, y_max = all_pos.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min) * 0.02
    frame = (
        (x_min - margin, x_max + margin),
        (y_min - margin, y_max + margin),
    )

    style = {
        "dpi": args.dpi,
        "node_size": args.node_size,
        "line_width": args.line_width,
        "margin_frac": 0.0,
    }

    workers = args.workers or os.cpu_count() or 1
    print(
        f"Plotting {len(pairs)} snapshot(s) with {workers} thread(s), "
        f"dpi={args.dpi}, node_size={args.node_size}, line_width={args.line_width}"
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(plot_one, idx, pp, ep, snapshot_dir, frame, style): idx
            for idx, pp, ep in pairs
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            idx = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  snapshot_{idx:05d} FAILED: {exc}")
            else:
                if done % max(1, len(pairs) // 10) == 0 or done == len(pairs):
                    print(f"  [{done}/{len(pairs)}] done")

    print(f"Complete. PNGs saved to {snapshot_dir}")


if __name__ == "__main__":
    main()
