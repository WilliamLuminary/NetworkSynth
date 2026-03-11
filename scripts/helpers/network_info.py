"""Print and save basic stats for a NetworKit .nkbin network.

Writes a .txt file alongside the .nkbin with node/edge counts.

Usage:
    python scripts/helpers/network_info.py network.nkbin
    python scripts/helpers/network_info.py network.nkbin --output stats.txt
"""

import os
import sys


def main(nkbin: str, output: str = None):
    """Load a .nkbin and save node/edge counts to a .txt file.

    Args:
        nkbin: Path to the .nkbin file.
        output: Output .txt path (default: <nkbin_stem>_info.txt in same dir).
    """
    import networkit as nk

    nkbin = os.path.abspath(nkbin)
    if not os.path.isfile(nkbin):
        sys.exit(f"File not found: {nkbin}")

    g = nk.readGraph(nkbin, nk.Format.NetworkitBinary)

    nodes = g.numberOfNodes()
    edges = g.numberOfEdges()
    weighted = g.isWeighted()

    lines = [
        f"file: {os.path.basename(nkbin)}",
        f"nodes: {nodes:,}",
        f"edges: {edges:,}",
        f"weighted: {weighted}",
    ]

    for line in lines:
        print(line)

    if output is None:
        output = nkbin.rsplit(".", 1)[0] + "_info.txt"

    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
