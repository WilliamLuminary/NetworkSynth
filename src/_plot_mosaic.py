# Temporary script: load saved 100x100 mosaic pickle and plot it.
import pickle
import sys

sys.path.insert(0, ".")

print("Loading pickle...")
pkl_path = (
    r"..\data\output\SampleConfig_results_20260216_013024"
    r"\A\20kX\synthetic"
    r"\mosaic_100x100_synthetic_network_20260216_013646.pkl"
)
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# The saver stores a list of synthetic networks
if isinstance(data, list):
    print(f"Pickle contains a list of {len(data)} graph(s)")
    graph = data[0]
else:
    graph = data

print(f"Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

print("Plotting (LineCollection + scatter)...")
from main_mosaic import plot_mosaic_network  # noqa: E402

img = plot_mosaic_network(graph)

out_path = (
    r"..\data\output\SampleConfig_results_20260216_013024"
    r"\A\20kX\synthetic"
    r"\mosaic_100x100_graph_replot.png"
)
import cv2  # noqa: E402

cv2.imwrite(out_path, img)
print(f"Saved plot to {out_path}")
