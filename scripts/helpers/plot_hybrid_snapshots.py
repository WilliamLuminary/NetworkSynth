import os
import re
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import save_hybrid_snapshot


def find_snapshot_pairs(snapshot_dir):
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


def _plot_one(idx, pos_path, edge_path, snapshot_dir, frame, style):
    import numpy as np

    positions = np.load(pos_path).tolist()
    edges = np.load(edge_path).tolist()
    save_hybrid_snapshot(positions, edges, frame, idx, snapshot_dir, style)


def plot(
    snapshot_dir: str,
    force: bool = False,
    dpi: int = 600,
    node_size: float = 0.01,
    line_width: float = 0.1,
    workers: int = None,
):
    """Re-plot hybrid snapshot PNGs from saved .npy data.

    Args:
        snapshot_dir: Path to the snapshots/ directory.
        force:        Re-plot all, even if PNG already exists.
        dpi:          Output resolution in dots per inch.
        node_size:    Scatter marker area in points².
        line_width:   Edge line width in points.
        workers:      Parallel threads (default: min(cpu_count // 2, 20)).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import numpy as np

    snapshot_dir = os.path.abspath(snapshot_dir)
    if not os.path.isdir(snapshot_dir):
        sys.exit(f"Directory not found: {snapshot_dir}")

    pairs = find_snapshot_pairs(snapshot_dir)
    if not pairs:
        sys.exit(f"No *_positions.npy / *_edges.npy pairs found in {snapshot_dir}")

    if not force:
        pairs = [
            (idx, pp, ep)
            for idx, pp, ep in pairs
            if not os.path.isfile(os.path.join(snapshot_dir, f"snapshot_{idx:05d}.png"))
        ]
        if not pairs:
            print("All PNGs already exist. Use --force to re-plot.")
            return

    all_pos = np.load(pairs[-1][1])
    x_min, y_min = all_pos.min(axis=0)
    x_max, y_max = all_pos.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min) * 0.02
    frame = (
        (x_min - margin, x_max + margin),
        (y_min - margin, y_max + margin),
    )

    from configs import RenderStyle

    style = RenderStyle(
        dpi=dpi,
        node_size=node_size,
        line_width=line_width,
        margin_frac=0.0,
    )

    if workers is None:
        cpu_count = os.cpu_count() or 1
        n_workers = min(max(1, cpu_count // 2), 20, len(pairs))
    else:
        n_workers = min(workers, len(pairs))
    print(
        f"Plotting {len(pairs)} snapshot(s) with {n_workers} worker(s), "
        f"dpi={dpi}, node_size={node_size}, line_width={line_width}"
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_plot_one, idx, pp, ep, snapshot_dir, frame, style): idx
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
    import fire

    fire.Fire(plot)
