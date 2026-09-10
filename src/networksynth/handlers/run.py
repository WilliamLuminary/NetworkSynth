# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import logging
from typing import Any, Dict, List

from networksynth.configs import DatasetId
from networksynth.graphs.synth_graph import SynthGraph
from networksynth.utils import plot_network

from .run_paths import RunPaths
from .saver import Saver, build_saver

logger = logging.getLogger(__name__)

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

    def __init__(self, config, run_paths: RunPaths, dataset_id: DatasetId):
        self._config = config
        self.dataset_id = dataset_id
        self.saver: Saver = build_saver(config, run_paths.for_dataset(dataset_id))

    def save(self, content: Any, identifier: str, prefix: str = "") -> None:
        self.saver.save(content, identifier, prefix)


class GenerationRun(Run):

    def __init__(self, config, run_paths: RunPaths, dataset_id: DatasetId):
        super().__init__(config, run_paths, dataset_id)
        self.original: SynthGraph = config.load_original_network(dataset_id)
        self.original_image = config.load_original_image(dataset_id)

        from .attributes_calculator import AttributesCalculator
        from .mapper import Mapper

        self.attributes = AttributesCalculator().analyze(self.original)
        self.mapper = Mapper(self.original)
        self.synthetic: List[SynthGraph] = []

    def add_synthetic_graph(self, graph: SynthGraph) -> None:
        self.synthetic.append(graph)

    def save_original(self) -> None:
        import dataclasses

        self.saver.begin_batch()
        if self.original_image is not None:
            self.save(self.original_image, "original_image")
        self.save(self.original, "original_network")
        self.save(dataclasses.asdict(self.attributes), "original_property")
        self.save(self.render_original(), "original_graph")
        self.saver.end_batch()

    def save_synthetic_outputs(self, prefix: str) -> None:
        for i, graph in enumerate(self.synthetic):
            self.saver.begin_batch()
            self.save(graph, "synthetic_network", f"{prefix}_n{i}_")
            self.saver.end_batch()

    def save_synthetic_plot(self, graph: SynthGraph, prefix: str = "") -> None:
        self.save(
            plot_network(
                data_type="synthetic_graph",
                graph=graph,
                style=self._config.render("synthetic_graph"),
                synthetic_frame_size=self._config.SYNTHETIC_FRAME_SIZE,
            ),
            "synthetic_graph",
            prefix,
        )

    def render_original(self):
        return plot_network(
            data_type="original_graph",
            graph=self.original,
            style=self._config.render("original_graph"),
            background=self.original_image,
            frame_size=self._config.FRAME_SIZE,
        )

    def original_report(self) -> str:
        return _report("Original Network", self.original, self.attributes)

    def synthetic_report(self, graph: SynthGraph) -> str:
        return _report("Synthetic Network", graph)


class ComparisonRun(Run):

    def __init__(
        self,
        config,
        run_paths: RunPaths,
        dataset_id: DatasetId,
        original_path: str,
        synthetic_path: str,
    ):
        super().__init__(config, run_paths, dataset_id)
        self.original = config.load_networks(original_path)
        self.synthetic = config.load_networks(synthetic_path)
        self._measure_weighted = config.MEASURE_WEIGHTED
        self._full_q_band = config.FULL_Q_BAND
        self._results: Dict[str, list] = {}

    def analyse(self) -> None:
        from networksynth.analysis import MultifractalProcessor

        # Synthetic first: it is drawn first, so the originals sit on top.
        for label, graphs in (
            ("synthetic", self.synthetic),
            ("original", self.original),
        ):
            processor = MultifractalProcessor(
                graphs, self._measure_weighted, self._full_q_band
            )
            processor.analyze()
            self._results[label] = processor.get_summary_data()

    def save_analysis(self) -> None:
        from networksynth.analysis.spectra_plot import plot_dimensions, plot_spectra

        self.save(
            {
                "original_multifractal_analysis_results": self._results["original"],
                "synthetic_multifractal_analysis_results": self._results["synthetic"],
            },
            "analysis_data",
        )
        for name, figure in (
            ("spectra", plot_spectra(self._results)),
            ("dimensions", plot_dimensions(self._results)),
        ):
            self.save(figure, "analysis_figure", f"{name}_")
