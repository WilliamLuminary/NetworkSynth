import os
import sys


def main(network: str, output: str = None):
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from graphs.graph_loader import load_graphs

    network = os.path.abspath(network)
    if not os.path.isfile(network):
        sys.exit(f"File not found: {network}")

    g = load_graphs(network)[0]

    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    weighted = g.is_weighted()

    lines = [
        f"file: {os.path.basename(network)}",
        f"nodes: {nodes:,}",
        f"edges: {edges:,}",
        f"weighted: {weighted}",
    ]

    for line in lines:
        print(line)

    if output is None:
        output = network.rsplit(".", 1)[0] + "_info.txt"

    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
