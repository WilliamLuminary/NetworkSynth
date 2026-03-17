# src/handlers/run_agent.py
import inspect
import logging
from typing import Optional

from configs import (
    FILE_CONFIGURATIONS,
    BaseConfig,
    DatasetId,
    Mode,
    PlotConfig,
)
from graphs.synth_graph import SynthGraph

from .data_loader import DataLoader
from .saver import Saver

logger = logging.getLogger(__name__)


class RunAgent:
    def __init__(
        self,
        *,
        dataset_id: Optional[DatasetId] = None,
        networks_path: Optional[str] = None,
        attr_path: Optional[str] = None,
    ):
        if networks_path:
            self.mode = Mode.ANA
            self.data_loader = DataLoader(Mode.ANA, path=networks_path)
            self.saver = Saver(output_dir=networks_path)
            self.batch_processor = None
        elif dataset_id is not None:
            self.mode = Mode.GEN
            self.data_loader = DataLoader(Mode.GEN, dataset_id=dataset_id)
            self.saver = Saver(dataset_id=dataset_id)
            self.attributes = None
            self.mapper = None
            self.batch_processor = None
        elif attr_path:
            self.mode = Mode.ATR
            self.data_loader = DataLoader(Mode.ATR, path=attr_path)
            self.saver = Saver(output_dir=attr_path)
            self.attributes = None
            self.mapper = None
            self.batch_processor = None
        else:
            raise ValueError(
                "Invalid arguments: provide dataset_id, " "networks_path, or attr_path"
            )

    def prepare_data(self):
        if self.mode == Mode.GEN:
            self.data_loader.load()
            original_network = self.data_loader.get_original_network()

            from .attributes_calculator import AttributesCalculator

            self.attributes = AttributesCalculator().analyze(original_network)

            from .mapper import Mapper

            self.mapper = Mapper(original_network)

        elif self.mode == Mode.ANA:
            self.data_loader.load()
            original_networks = self.data_loader.get_original_network()
            synthetic_networks = self.data_loader.get_synthetic_networks()
            from analysis import MultifractalBatchProcessor

            self.batch_processor = MultifractalBatchProcessor(
                original_networks, synthetic_networks
            )

        elif self.mode == Mode.ATR:
            self.data_loader.load()
            from .attributes_calculator import AttributesCalculator

            self.attributes = AttributesCalculator(**self.data_loader.get_attr_dict())

    def multifractal_analysis_in_generate_mode(self):
        assert self.mode == Mode.GEN, "This method is only available in Generate mode."
        original_networks = [self.data_loader.get_original_network()]
        synthetic_networks = self.data_loader.get_synthetic_networks()
        from analysis import MultifractalBatchProcessor

        self.batch_processor = MultifractalBatchProcessor(
            original_networks, synthetic_networks
        )
        self.multifractal_analysis()

    def multifractal_analysis(self):
        assert self.mode in (
            Mode.ANA,
            Mode.GEN,
        ), "This method is only available in Analyze or Generate mode."
        assert self.batch_processor, "Batch processor is not initialized."
        self.batch_processor.process().plot()

    def add_synthetic_graph(self, graph: SynthGraph):
        """
        Add a synthetic graph to the list of synthetic graphs.
        NOT THREAD-SAFE.
        """
        self.data_loader.add_synthetic_graph(graph)

    def get_original_network(self):
        return self.data_loader.get_original_network()

    def save_synthetic_outputs(self, prefix: str):
        """Save synthetic networks (collection pkl + per-graph exports)."""
        self.save("synthetic_network", prefix)
        graphs = self.data_loader.get_synthetic_networks()
        for i, g in enumerate(graphs):
            Saver.begin_batch()
            self.saver.save(g, "synthetic_export", f"{prefix}_n{i}_")
            Saver.end_batch()

    def save(
        self,
        identifier: str,
        prefix: Optional[str] = None,
        *,
        content=None,
    ):
        """Save content identified by *identifier* (a plain string).

        If *content* is not provided, loads it from internal state.
        """
        is_plot = isinstance(FILE_CONFIGURATIONS.get(identifier), PlotConfig)
        if not self.saver and not is_plot:
            return

        prefix = f"{prefix}_" if prefix and not prefix.endswith("_") else (prefix or "")

        if content is not None:
            if identifier == "synthetic_graph":
                content = plot_network(
                    data_type=identifier,
                    graph=content,
                    show=not self.saver,
                )
                if not self.saver:
                    return
            self.saver.save(content, identifier, prefix)
            return

        if identifier == "original_image":
            content = self.data_loader.get_original_image()

        elif identifier == "original_network":
            content = self.data_loader.get_original_network()

        elif identifier == "original_property":
            import dataclasses

            assert self.attributes, "Attributes not initialized."
            content = dataclasses.asdict(self.attributes)

        elif identifier == "original_report":
            original_network = self.data_loader.get_original_network()
            content = _build_original_report(original_network, self.attributes)

        elif identifier == "original_graph":
            original_network = self.data_loader.get_original_network()
            original_image = self.data_loader.get_original_image()
            content = plot_network(
                data_type="original_graph",
                graph=original_network,
                background=original_image,
                show=True,
            )
            if not self.saver:
                return

        elif identifier == "synthetic_network":
            content = self.data_loader.get_synthetic_networks()

        elif identifier == "analysis_data":
            content = {
                "original_multifractal_analysis_results": (
                    self.batch_processor.get_original_data()
                ),
                "synthetic_multifractal_analysis_results": (
                    self.batch_processor.get_synthetic_data()
                ),
            }

        elif identifier == "analysis_figure":
            for image_name, image in self.batch_processor.get_images().items():
                self.saver.save(image, identifier, f"{image_name}_")
            return

        else:
            logger.error("No content loader for identifier: %s", identifier)
            return

        if content is not None:
            self.saver.save(content, identifier, prefix)


# ---------------------------------------------------------------------------
# Original-network report
# ---------------------------------------------------------------------------


def _build_original_report(graph: SynthGraph, attributes) -> str:
    lines = [
        "Original Network Report",
        "=" * 40,
        f"Nodes:              {graph.number_of_nodes()}",
        f"Edges:              {graph.number_of_edges()}",
    ]
    if attributes:
        lines.append(f"Average degree:     {attributes.average_degree:.4f}")
        lines.append(f"Average edge length: {attributes.average_length:.4f}")
        if attributes.degree_distribution:
            lines.append("")
            lines.append("Degree distribution:")
            for deg in sorted(attributes.degree_distribution):
                frac = attributes.degree_distribution[deg]
                lines.append(f"  degree {deg:>3d}: {frac:.4f}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Thread-safe plotting (no pyplot global state)
# ---------------------------------------------------------------------------


def plot_network(data_type: str, graph: SynthGraph, **kwargs):
    """Render a graph to an ndarray image.

    Uses matplotlib's OO API exclusively -- no pyplot globals --
    so it is safe to call from any thread or process.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    file_config = FILE_CONFIGURATIONS.get(data_type)
    assert isinstance(file_config, PlotConfig), (
        f"{data_type} shouldn't call " f"{inspect.currentframe().f_code.co_name}."
    )

    if data_type == "original_graph":
        frame = (
            (0, BaseConfig.FRAME_SIZE[0]),
            (0, BaseConfig.FRAME_SIZE[1]),
        )
    else:
        from utils import calculate_frame

        frame = calculate_frame(graph)

    frame_width = frame[0][1] - frame[0][0]
    frame_height = frame[1][1] - frame[1][0]
    aspect_ratio = frame_width / frame_height
    fig_height = 10
    fig_width = fig_height * aspect_ratio

    fig = Figure(figsize=(fig_width, fig_height), dpi=300)
    ax = fig.add_subplot(111)

    positions = graph.positions()
    edge_width = file_config.line_width
    for u, v in graph.edges():
        pos_u = positions[u]
        pos_v = positions[v]
        ax.plot(
            [pos_u[0], pos_v[0]],
            [pos_u[1], pos_v[1]],
            "r-",
            linewidth=edge_width,
            zorder=2,
        )

    node_size = file_config.node_size
    for node in graph.nodes():
        pos = positions[node]
        ax.plot(pos[0], pos[1], "bo", markersize=node_size, zorder=2)

    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])

    if data_type == "original_graph":
        image = kwargs.get("background", None)
        if image is not None:
            alpha = getattr(file_config, "alpha", 1.0)
            img_height, img_width = image.shape[:2]
            ax.imshow(
                image,
                cmap="gray",
                alpha=alpha,
                extent=(0, img_width, img_height, 0),
                aspect="auto",
            )
        else:
            logger.info("No background image provided.")
    else:
        ax.add_patch(
            Rectangle(
                (frame[0][0], frame[1][0]),
                frame[0][1] - frame[0][0],
                frame[1][1] - frame[1][0],
                facecolor="none",
                edgecolor=(0, 0, 0, 0.8),
                linewidth=2,
                zorder=1,
            )
        )

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    show_on_the_fly = (
        kwargs["show"]
        if "show" in kwargs
        else getattr(file_config, "show_on_the_fly", True)
    )
    from utils import finalize_plot

    return finalize_plot(fig, show_on_the_fly)
