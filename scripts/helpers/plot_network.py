import os
import sys

# Ensure project root is on sys.path when invoked as a script.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_graph(path):
    """Positions travel inside the file now, so there is nothing to stitch on."""
    from graphs.graph_loader import load_graphs

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")

    graph = load_graphs(path)[0]
    print(
        f"Loaded: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges, weighted={graph.is_weighted()}"
    )
    return graph


def plot_single(
    network: str,
    output: str = None,
    fmt: str = "webp",
    dpi: int = None,
    margin: float = 0.02,
):
    """Plot a network using the CV2-based renderer.

    Args:
        network: Path to the network file.
        output: Output path (default: <input_stem>.<fmt> in same dir).
        fmt: Output format — webp (default) or png.
        dpi: Resolution. Auto-calculated from node count if omitted.
        margin: Fractional margin around the network bounding box.
    """
    fmt = fmt.lower()
    if fmt not in ("webp", "png"):
        sys.exit(f"Unsupported format: {fmt}. Use webp or png.")

    graph = load_graph(network)

    from configs import RenderStyle
    from utils import recommend_dpi_cv2, render_network

    if dpi is None:
        dpi = recommend_dpi_cv2(graph.number_of_nodes())
    print(f"DPI: {dpi} (for {graph.number_of_nodes():,} nodes)")

    img = render_network(graph, RenderStyle(dpi=dpi, margin_frac=margin))

    if output is None:
        output = network.rsplit(".", 1)[0] + f".{fmt}"

    # img is a BGR ndarray; the production serializers handle it.
    from configs.file_definitions import save_png, save_webp

    (save_webp if fmt == "webp" else save_png)(img, output)
    print(f"Saved: {output} ({img.shape[1]}x{img.shape[0]})")


def main(
    network: str = None,
    batch_dir: str = None,
    **kwargs,
):
    """Plot one or many networks.

    Args:
        network: Path to a single network file.
        batch_dir: Directory to scan recursively for network files.
                   Each file is plotted; errors are reported and skipped.

    All other flags (fmt, dpi, margin) are forwarded to the per-file
    plot function.
    """
    if network:
        plot_single(network, **kwargs)
    elif batch_dir:
        import glob

        files = sorted(
            glob.glob(os.path.join(batch_dir, "**", "*.graphml.gz"), recursive=True)
        )
        if not files:
            print(f"No network files found under {batch_dir}")
            return
        print(f"Found {len(files)} network files under {batch_dir}")
        failed = []
        for i, f in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {f}")
            try:
                plot_single(f, **kwargs)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                failed.append(f)
        if failed:
            print(f"\n{len(failed)}/{len(files)} failed:")
            for f in failed:
                print(f"  {f}")
        else:
            print(f"\nAll {len(files)} files plotted successfully.")
    else:
        sys.exit("Provide --network <file> or --batch_dir <directory>.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
