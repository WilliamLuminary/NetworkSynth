"""Plot a saved NetworKit network using the production rendering code.

Reads a .nkbin file (with companion _positions.npy), wraps it in a
SynthGraph, and calls the same plot function used by the hybrid pipeline.

Usage:
    python scripts/helpers/plot_network.py network.nkbin
    python scripts/helpers/plot_network.py network.nkbin --output out.webp
    python scripts/helpers/plot_network.py network.nkbin --fmt svg --node_size 0.01
"""

import os
import sys


def load_graph(nkbin_path):
    import networkit as nk
    import numpy as np

    nkbin_path = os.path.abspath(nkbin_path)
    if not os.path.isfile(nkbin_path):
        sys.exit(f"File not found: {nkbin_path}")

    pos_path = nkbin_path.rsplit(".", 1)[0] + "_positions.npy"
    if not os.path.isfile(pos_path):
        sys.exit(
            f"Companion positions file not found: {pos_path}\n"
            f"Expected alongside {nkbin_path}"
        )

    g = nk.readGraph(nkbin_path, nk.Format.NetworkitBinary)
    positions = np.load(pos_path)
    assert (
        positions.shape[0] == g.numberOfNodes()
    ), f"Position count ({positions.shape[0]}) != node count ({g.numberOfNodes()})"

    print(
        f"Loaded: {g.numberOfNodes():,} nodes, {g.numberOfEdges():,} edges, "
        f"weighted={g.isWeighted()}"
    )

    from graphs.synth_graph import SynthGraph

    return SynthGraph(g, positions)


def plot_single(
    nkbin: str,
    output: str = None,
    fmt: str = "webp",
    dpi: int = None,
    node_size: float = 0.01,
    line_width: float = 0.1,
    margin: float = 0.02,
):
    """Plot a .nkbin network using the production plot function.

    Args:
        nkbin: Path to the .nkbin file.
        output: Output path (default: <input_stem>.<fmt> in same dir).
        fmt: Output format — webp (default, compact), png, pdf, or svg.
        dpi: Resolution for raster output (webp/png). Auto-calculated from
             node count if omitted. Ignored for svg/pdf.
        node_size: Matplotlib scatter marker size.
        line_width: Edge line width.
        margin: Fractional margin around the network bounding box.
    """
    fmt = fmt.lower()
    if fmt not in ("webp", "png", "pdf", "svg"):
        sys.exit(f"Unsupported format: {fmt}. Use webp, png, pdf, or svg.")

    graph = load_graph(nkbin)

    from pipelines.hybrid import plot_hybrid_network
    from utils import recommend_dpi

    if dpi is None:
        dpi = recommend_dpi(graph.number_of_nodes())
        print(f"Auto DPI: {dpi} (for {graph.number_of_nodes():,} nodes)")

    fig = plot_hybrid_network(
        graph,
        node_size=node_size,
        line_width=line_width,
        margin_frac=margin,
        dpi=dpi,
    )

    if output is None:
        output = nkbin.rsplit(".", 1)[0] + f".{fmt}"

    if fmt == "webp":
        from utils import save_figure_as_webp

        try:
            save_figure_as_webp(fig, output, dpi=dpi)
            print(f"Saved: {output}")
        except Exception as e:
            print(f"WebP failed: {e}")
            output = nkbin.rsplit(".", 1)[0] + ".png"
            save_kwargs = {"format": "png", "bbox_inches": "tight", "dpi": dpi}
            fig.savefig(output, **save_kwargs)
            print(f"Saved as PNG instead: {output}")
    else:
        save_kwargs = {"format": fmt, "bbox_inches": "tight"}
        if fmt in ("png",):
            save_kwargs["dpi"] = dpi
        fig.savefig(output, **save_kwargs)
        print(f"Saved: {output}")

    from matplotlib import pyplot as plt

    plt.close(fig)


def main(
    nkbin: str = None,
    batch_dir: str = None,
    **kwargs,
):
    """Plot one or many .nkbin networks.

    Args:
        nkbin: Path to a single .nkbin file.
        batch_dir: Directory to scan recursively for .nkbin files.
                   Each file is plotted; errors are reported and skipped.

    All other flags (fmt, dpi, node_size, line_width, margin) are
    forwarded to the per-file plot function.
    """
    if nkbin:
        plot_single(nkbin, **kwargs)
    elif batch_dir:
        import glob

        files = sorted(
            glob.glob(os.path.join(batch_dir, "**", "*.nkbin"), recursive=True)
        )
        if not files:
            print(f"No .nkbin files found under {batch_dir}")
            return
        print(f"Found {len(files)} .nkbin files under {batch_dir}")
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
        sys.exit("Provide --nkbin <file> or --batch_dir <directory>.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
