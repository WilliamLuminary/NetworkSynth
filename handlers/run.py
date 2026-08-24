from __future__ import annotations

import logging
from typing import Any, List

from configs import DatasetId
from graphs.synth_graph import SynthGraph
from utils import plot_network

from .run_paths import RunPaths
from .saver import Saver, build_saver

logger = logging.getLogger(__name__)

#: One layout for every network report, so ``report.txt`` reads the same
#: whichever pipeline wrote it.  Counts get thousands separators, measurements
#: four decimals, and labels share a column.
_REPORT_RULE = "=" * 40
_REPORT_LABEL_WIDTH = 21


def _report(title: str, graph: SynthGraph, attributes=None) -> str:
    def row(label: str, value: str) -> str:
        return f"{label + ':':<{_REPORT_LABEL_WIDTH}}{value}"

    lines = [
        f"{title} Report",
        _REPORT_RULE,
        row("Nodes", f"{graph.number_of_nodes():,}"),
        row("Edges", f"{graph.number_of_edges():,}"),
    ]
    if attributes is not None:
        lines.append(row("Average degree", f"{attributes.average_degree:.4f}"))
        lines.append(row("Average edge length", f"{attributes.average_length:.4f}"))
        if attributes.degree_distribution:
            lines.append("")
            lines.append("Degree distribution:")
            for degree in sorted(attributes.degree_distribution):
                fraction = attributes.degree_distribution[degree]
                lines.append(f"  degree {degree:>3d}: {fraction:.4f}")
    lines.append("")
    return "\n".join(lines)


class Run:
    """One dataset being processed, and the directory it writes to.

    Subclasses say what kind of run it is by what they load in their
    constructor.  There is no mode flag: a run is ready the moment it exists,
    and a method that only makes sense for one kind lives only on that kind.
    """

    def __init__(self, config, run_paths: RunPaths, dataset_id: DatasetId):
        self._config = config
        self.dataset_id = dataset_id
        self.saver: Saver = build_saver(config, run_paths.for_dataset(dataset_id))

    def save(self, content: Any, identifier: str, prefix: str = "") -> None:
        self.saver.save(content, identifier, prefix)


class GenerationRun(Run):
    """One original network, and the synthetic networks made from it.

    Everything about the input is measured in the constructor, so there is no
    half-built state and no ``prepare_data()`` step a pipeline can forget.
    Nothing here analyses the result: the quality gate is an
    :class:`~analysis.error_checker.ErrorChecker`, and a full spectrum belongs
    to :class:`ComparisonRun`.
    """

    def __init__(self, config, run_paths: RunPaths, dataset_id: DatasetId):
        super().__init__(config, run_paths, dataset_id)
        self.original: SynthGraph = config.ORIGINAL_NETWORK_FUNC(dataset_id)
        self.original_image = config.ORIGINAL_IMAGE_FUNC(dataset_id)

        from .attributes_calculator import AttributesCalculator
        from .mapper import Mapper

        self.attributes = AttributesCalculator().analyze(self.original)
        self.mapper = Mapper(self.original)
        self.synthetic: List[SynthGraph] = []

    def add_synthetic_graph(self, graph: SynthGraph) -> None:
        """NOT THREAD-SAFE."""
        self.synthetic.append(graph)

    def save_original(self) -> None:
        """Record the input: its image, the network, its measurements, a plot.

        One batch, so the four files share a timestamp and read as the group
        they are.  The image is skipped when the dataset has none — an input
        given as a CSV pair usually does not.
        """
        import dataclasses

        self.saver.begin_batch()
        if self.original_image is not None:
            self.save(self.original_image, "original_image")
        self.save(self.original, "original_network")
        self.save(dataclasses.asdict(self.attributes), "original_property")
        self.save(self.render_original(), "original_graph")
        self.saver.end_batch()

    def save_synthetic_outputs(self, prefix: str) -> None:
        """The whole batch as one file, then each network on its own."""
        self.save(self.synthetic, "synthetic_network", f"{prefix}_")
        for i, graph in enumerate(self.synthetic):
            self.saver.begin_batch()
            self.save(graph, "synthetic_export", f"{prefix}_n{i}_")
            self.saver.end_batch()

    def save_synthetic_plot(self, graph: SynthGraph, prefix: str = "") -> None:
        style = self._config.PLOT_STYLE
        self.save(
            plot_network(
                data_type="synthetic_graph",
                graph=graph,
                show=False,
                synthetic_frame_size=self._config.SYNTHETIC_FRAME_SIZE,
                node_size=style.get("node_size"),
                line_width=style.get("line_width"),
            ),
            "synthetic_graph",
            prefix,
        )

    def render_original(self):
        return plot_network(
            data_type="original_graph",
            graph=self.original,
            background=self.original_image,
            show=True,
            node_scale=self._config.ORIGINAL_GRAPH_NODE_SCALE,
            frame_size=self._config.FRAME_SIZE,
        )

    def original_report(self) -> str:
        """The input, measured: counts plus the attributes generation reads."""
        return _report("Original Network", self.original, self.attributes)

    def synthetic_report(self, graph: SynthGraph) -> str:
        """Counts only.

        Generation deliberately does not measure what it produced — that is
        ``ComparisonRun``'s job — so there are no averages to report here.
        """
        return _report("Synthetic Network", graph)


class ComparisonRun(Run):
    """Two sets of networks, measured against each other.

    Both sets are named by the caller; nothing is discovered and nothing is
    generated here.  Comparing a generate run against its input means pointing
    this at that run's two folders.
    """

    def __init__(
        self,
        config,
        run_paths: RunPaths,
        dataset_id: DatasetId,
        original_path: str,
        synthetic_path: str,
    ):
        super().__init__(config, run_paths, dataset_id)
        self.original = config.NETWORKS_FUNC(original_path)
        self.synthetic = config.NETWORKS_FUNC(synthetic_path)

        from analysis import MultifractalBatchProcessor

        self._processor = MultifractalBatchProcessor(
            self.original,
            self.synthetic,
            measure_weighted=config.MEASURE_WEIGHTED,
            full_q_band=config.FULL_Q_BAND,
        )

    def analyse(self) -> None:
        self._processor.process().plot()

    def save_analysis(self) -> None:
        self.save(
            {
                "original_multifractal_analysis_results": (
                    self._processor.get_original_data()
                ),
                "synthetic_multifractal_analysis_results": (
                    self._processor.get_synthetic_data()
                ),
            },
            "analysis_data",
        )
        for name, image in self._processor.get_images().items():
            self.save(image, "analysis_figure", f"{name}_")
