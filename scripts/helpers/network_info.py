import os
import sys


def main(nkbin: str, output: str = None):
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
