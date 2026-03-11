# src/handlers/run_agent.py
import inspect
import logging
from typing import Optional

from configs import (
    FILE_CONFIGURATIONS,
    BaseConfig,
    DatasetId,
    DataType,
    FileTag,
    Mode,
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
        """Save synthetic networks in formats specified by
        BaseConfig.OUTPUT_FORMATS."""
        fmts = {f.lower() for f in BaseConfig.OUTPUT_FORMATS}
        if "pkl" in fmts:
            self.save(DataType.SYNTHETIC_NETWORK, prefix)
        graphs = self.data_loader.get_synthetic_networks()
        for i, g in enumerate(graphs):
            g_prefix = f"{prefix}_n{i}_"
            Saver.begin_batch()
            if "csv" in fmts:
                self.save(DataType.SYNTHETIC_EDGELIST, g_prefix, arg=g)
                self.save(DataType.SYNTHETIC_POSITIONS, g_prefix, arg=g)
            if "nkbin" in fmts:
                self.save(DataType.SYNTHETIC_NETWORK_NKI, g_prefix, arg=g)
            Saver.end_batch()

    def save(
        self,
        data_type: DataType,
        file_name_prefix: Optional[str] = None,
        arg=None,
    ):
        if not self.saver and not data_type.has_tag(FileTag.PLOT):
            return

        file_name_prefix = (
            f"{file_name_prefix}_"
            if file_name_prefix and file_name_prefix[-1] != "_"
            else (file_name_prefix or "")
        )

        if data_type == DataType.ORIGINAL_IMAGE:
            assert self.mode == Mode.GEN
            original_image = self.data_loader.get_original_image()
            self.saver.save_file(original_image, data_type, file_name_prefix)

        elif data_type == DataType.ORIGINAL_NETWORK:
            assert self.mode == Mode.GEN
            original_network = self.data_loader.get_original_network()
            self.saver.save_file(original_network, data_type, file_name_prefix)

        elif data_type == DataType.ORIGINAL_PROPERTY:
            assert self.mode == Mode.GEN
            assert self.attributes, "Attributes not initialized."
            import dataclasses

            self.saver.save_file(
                dataclasses.asdict(self.attributes),
                data_type,
                file_name_prefix,
            )  # type: ignore

        elif data_type == DataType.ORIGINAL_GRAPH:
            original_network = self.data_loader.get_original_network()
            original_image = self.data_loader.get_original_image()

            original_figure = plot_network(
                data_type=DataType.ORIGINAL_GRAPH,
                graph=original_network,
                background=original_image,
                show=True,
            )
            if self.saver:
                self.saver.save_file(original_figure, data_type, file_name_prefix)

        elif data_type == DataType.SYNTHETIC_GRAPH:
            assert isinstance(arg, SynthGraph)
            show_if_not_saving = bool(not self.saver and arg)
            synthetic_figure = plot_network(
                data_type=DataType.SYNTHETIC_GRAPH,
                graph=arg,
                show=show_if_not_saving,
            )
            if not show_if_not_saving:
                self.saver.save_file(
                    synthetic_figure,
                    DataType.SYNTHETIC_GRAPH,
                    file_name_prefix,
                )

        elif data_type == DataType.SYNTHETIC_NETWORK:
            synthetic_networks = self.data_loader.get_synthetic_networks()
            self.saver.save_file(synthetic_networks, data_type, file_name_prefix)

        elif data_type == DataType.SYNTHETIC_EDGELIST:
            assert isinstance(arg, SynthGraph)
            edgelist = _synth_to_edgelist_csv(arg)
            self.saver.save_file(edgelist, data_type, file_name_prefix)

        elif data_type == DataType.SYNTHETIC_POSITIONS:
            assert isinstance(arg, SynthGraph)
            positions = _synth_to_positions_csv(arg)
            self.saver.save_file(positions, data_type, file_name_prefix)

        elif data_type == DataType.SYNTHETIC_NETWORK_NKI:
            assert isinstance(arg, SynthGraph)
            self.saver.save_file(
                (arg.nk, arg.positions()),
                data_type,
                file_name_prefix,
            )

        elif data_type == DataType.ANALYSIS_DATA:
            assert self.mode == Mode.ANA
            self.saver.save_file(
                {
                    "original_multifractal_analysis_results": (
                        self.batch_processor.get_original_data()
                    ),
                    "synthetic_multifractal_analysis_results": (
                        self.batch_processor.get_synthetic_data()
                    ),
                },
                data_type,
                file_name_prefix,
            )

        elif data_type == DataType.ANALYSIS_FIGURE:
            assert self.mode == Mode.ANA
            for image_name, image in self.batch_processor.get_images().items():
                self.saver.save_file(image, data_type, f"{image_name}_")

        else:
            logger.error("Please configure save() for %s.", data_type)


# ---------------------------------------------------------------------------
# Thread-safe plotting (no pyplot global state)
# ---------------------------------------------------------------------------


def plot_network(data_type: DataType, graph: SynthGraph, **kwargs):
    """Render a graph to an ndarray image.

    Uses matplotlib's OO API exclusively -- no pyplot globals --
    so it is safe to call from any thread or process.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    assert data_type.has_tag(FileTag.PLOT), (
        f"{data_type} shouldn't call " f"{inspect.currentframe().f_code.co_name}."
    )

    file_config = FILE_CONFIGURATIONS.get(data_type)

    if data_type is DataType.ORIGINAL_GRAPH:
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

    if data_type is DataType.ORIGINAL_GRAPH:
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


def _synth_to_edgelist_csv(graph: SynthGraph):
    rows = [["source_index", "target_index", "edge_weight"]]
    for u, v, w in graph.edges_with_weights():
        rows.append([u, v, w])
    return rows


def _synth_to_positions_csv(graph: SynthGraph):
    positions = graph.positions()
    rows = [["x", "y"]]
    for pos in positions:
        rows.append([pos[0], pos[1]])
    return rows
