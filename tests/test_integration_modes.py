"""
Lightweight integration smoke tests for each pipeline mode.

Uses sample data with minimal generation counts to verify the full
pipeline path (config → load → generate → save) works end-to-end.
"""

import logging
import os

import pytest

pytestmark = pytest.mark.integration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "input", "samples")


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset Saver and BaseConfig class-level state between tests."""
    from configs.base_config import BaseConfig
    from handlers.saver import Saver

    snapshot = {}
    for name in dir(BaseConfig):
        if name.startswith("__"):
            continue
        if name.isupper() or name.startswith("save_"):
            snapshot[name] = getattr(BaseConfig, name)

    Saver.base_output_dir = None
    Saver.mode = None
    Saver._batch_timestamp = None
    yield
    Saver.base_output_dir = None
    Saver.mode = None
    Saver._batch_timestamp = None
    for name in list(vars(BaseConfig)):
        if name.startswith("save_") and name not in snapshot:
            delattr(BaseConfig, name)
    for name, value in snapshot.items():
        setattr(BaseConfig, name, value)


# ------------------------------------------------------------------ #
# Generate mode
# ------------------------------------------------------------------ #


class TestGenerateMode:
    def test_load_original_and_save(self, tmp_path):
        """Load original data, compute attributes, save originals."""
        from configs.generate_mode.config_sample import SampleConfig

        SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 0
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 0
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import BaseConfig
        from handlers import RunAgent, Saver

        Saver.initialize()
        dataset_id = BaseConfig.get_datasets()[0]
        agent = RunAgent(dataset_id=dataset_id)
        agent.prepare_data()

        agent.save("original_network")
        agent.save("original_graph")
        agent.save("original_property")

        out = agent.saver.output_dir
        assert os.path.isdir(out), f"Output dir not created: {out}"
        files = []
        for root, _, fnames in os.walk(out):
            files.extend(fnames)
        assert len(files) >= 3, f"Expected ≥3 output files, got {files}"
        logger.info(f"Generate (originals): {len(files)} files in {out}")

    def test_generate_one_network(self, tmp_path):
        """Generate a single synthetic network and save it."""
        from configs.generate_mode.config_sample import SampleConfig

        SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 1
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 1
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import BaseConfig
        from graphs import GraphGenerator
        from handlers import RunAgent, Saver
        from utils import trim_graph

        Saver.initialize()
        dataset_id = BaseConfig.get_datasets()[0]
        agent = RunAgent(dataset_id=dataset_id)
        agent.prepare_data()

        generator = GraphGenerator(agent.attributes)
        synth = generator.generate_network()
        synth = trim_graph(synth, agent.attributes.average_degree)
        agent.mapper.assign_weights(synth)

        assert synth.number_of_nodes() > 50
        assert synth.number_of_edges() > 50

        agent.add_synthetic_graph(synth)
        agent.save("synthetic_graph", content=synth)
        agent.save_synthetic_outputs("test_")

        out = agent.saver.output_dir
        files = []
        for root, _, fnames in os.walk(out):
            files.extend(fnames)
        synth_files = [f for f in files if "synthetic" in f]
        assert len(synth_files) >= 1, f"No synthetic files: {files}"
        logger.info(
            f"Generate (1 network): {synth.number_of_nodes()} nodes, "
            f"{synth.number_of_edges()} edges, {len(synth_files)} synth files"
        )


# ------------------------------------------------------------------ #
# From-props mode
# ------------------------------------------------------------------ #


class TestFromPropsMode:
    def test_generate_from_properties(self, tmp_path):
        """Load pre-computed attributes and generate one network."""
        from configs.attr_generate_mode.config_sample import SampleConfig

        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 1
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 1
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import BaseConfig
        from graphs import GraphGenerator
        from handlers import RunAgent
        from utils import trim_graph

        agent = RunAgent(attr_path=BaseConfig.ATTRIBUTES_DICT_DATA_PATH)
        agent.prepare_data()

        generator = GraphGenerator(agent.attributes)
        synth = generator.generate_network()
        synth = trim_graph(synth, agent.attributes.average_degree)

        assert synth.number_of_nodes() > 50
        logger.info(
            f"From-props: {synth.number_of_nodes()} nodes, "
            f"{synth.number_of_edges()} edges"
        )


# ------------------------------------------------------------------ #
# Mosaic mode
# ------------------------------------------------------------------ #


class TestMosaicMode:
    def test_mosaic_2x2(self, tmp_path):
        """Generate a 2x2 mosaic and stitch it."""
        from configs.mosaic_mode.config_sample import SampleConfig

        SampleConfig.GRID_ROWS = 2
        SampleConfig.GRID_COLS = 2
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import BaseConfig
        from graphs import GraphGenerator
        from graphs.mosaic_stitcher import MosaicStitcher
        from handlers import RunAgent, Saver
        from utils import trim_graph

        Saver.initialize()
        dataset_id = BaseConfig.get_datasets()[0]
        agent = RunAgent(dataset_id=dataset_id)
        agent.prepare_data()
        attributes = agent.attributes

        tile_w, tile_h = BaseConfig.TILE_FRAME_SIZE
        overlap = BaseConfig.OVERLAP_MARGIN_FRACTION
        tile_gen_frame = (
            round(tile_w + 2 * overlap * tile_w),
            round(tile_h + 2 * overlap * tile_h),
        )

        tile_graphs = {}
        for r in range(BaseConfig.GRID_ROWS):
            for c in range(BaseConfig.GRID_COLS):
                gen = GraphGenerator(attributes)
                g = gen.generate_network(frame_range=tile_gen_frame)
                g = trim_graph(g, attributes.average_degree)
                positions = g.positions()
                positions[:, 0] += c * tile_w + tile_w / 2.0
                positions[:, 1] += r * tile_h + tile_h / 2.0
                tile_graphs[(r, c)] = g

        merge_threshold = attributes.average_length * BaseConfig.CLOSED_NODES_FACTOR
        stitcher = MosaicStitcher(merge_threshold=merge_threshold)
        mosaic = stitcher.stitch(tile_graphs)
        agent.mapper.assign_weights(mosaic)

        assert mosaic.number_of_nodes() > 100
        assert mosaic.number_of_edges() > 100

        Saver.begin_batch()
        agent.saver.save(mosaic, "synthetic_export", "mosaic_2x2_")
        Saver.end_batch()

        out = agent.saver.output_dir
        files = []
        for root, _, fnames in os.walk(out):
            files.extend(fnames)
        assert any("mosaic" in f for f in files)
        logger.info(
            f"Mosaic 2x2: {mosaic.number_of_nodes()} nodes, "
            f"{mosaic.number_of_edges()} edges, {len(files)} files"
        )


# ------------------------------------------------------------------ #
# Scaling mode
# ------------------------------------------------------------------ #


class TestScalingMode:
    def test_scaling_2x2(self, tmp_path):
        """Generate a 2x2 scaled network via multi-root BFS."""
        from configs.scaling_mode.config_sample import SampleConfig

        SampleConfig.SCALE_ROWS = 2
        SampleConfig.SCALE_COLS = 2
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import BaseConfig
        from graphs import GraphGenerator
        from handlers import RunAgent, Saver
        from utils import trim_graph

        Saver.initialize()
        dataset_id = BaseConfig.get_datasets()[0]
        agent = RunAgent(dataset_id=dataset_id)
        agent.prepare_data()

        gen = GraphGenerator(agent.attributes)
        scaled = gen.generate_scaled_network(
            scale_rows=BaseConfig.SCALE_ROWS,
            scale_cols=BaseConfig.SCALE_COLS,
            max_rounds=BaseConfig.MAX_GENERATION_ROUNDS,
            root_spacing_factor=BaseConfig.ROOT_SPACING_FACTOR,
        )
        scaled = trim_graph(scaled, agent.attributes.average_degree)
        agent.mapper.assign_weights(scaled)

        assert scaled.number_of_nodes() > 200
        assert scaled.number_of_edges() > 200

        Saver.begin_batch()
        agent.saver.save(scaled, "synthetic_export", "scaled_2x2_")
        Saver.end_batch()

        out = agent.saver.output_dir
        files = []
        for root, _, fnames in os.walk(out):
            files.extend(fnames)
        assert any("scaled" in f for f in files)
        logger.info(
            f"Scaling 2x2: {scaled.number_of_nodes()} nodes, "
            f"{scaled.number_of_edges()} edges, {len(files)} files"
        )


# ------------------------------------------------------------------ #
# Analyze mode
# ------------------------------------------------------------------ #


class TestAnalyzeMode:
    def test_analyze_graph(self, tmp_path):
        """Run multifractal analysis on a generated graph."""
        from configs.generate_mode.config_sample import SampleConfig

        SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 0
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 0
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from analysis import MultifractalAnalyzer
        from configs import BaseConfig
        from handlers import RunAgent, Saver

        Saver.initialize()
        dataset_id = BaseConfig.get_datasets()[0]
        agent = RunAgent(dataset_id=dataset_id)
        agent.prepare_data()

        original = agent.get_original_network()
        analyzer = MultifractalAnalyzer(original)
        result = analyzer.analyze_graph()

        assert "tau_list" in result
        assert "diameter" in result
        assert "assortativity" in result
        assert len(result["tau_list"]) > 0
        logger.info(
            f"Analyze: diameter={result['diameter']}, "
            f"assortativity={result['assortativity']:.4f}, "
            f"tau_list length={len(result['tau_list'])}"
        )


# ------------------------------------------------------------------ #
# Snapshot mode
# ------------------------------------------------------------------ #


class TestSnapshotMode:
    def test_generate_with_snapshots(self, tmp_path):
        """Generate a single network with BFS snapshots via Snapshot1x1Config."""
        from configs.generate_mode.config_snapshot_1x1 import Snapshot1x1Config

        Snapshot1x1Config.BASE_OUTPUT_PATH = str(tmp_path)
        Snapshot1x1Config.initialize()

        from configs import BaseConfig
        from handlers import RunAgent, Saver

        Saver.initialize()
        dataset_id = BaseConfig.get_datasets()[0]
        agent = RunAgent(dataset_id=dataset_id)
        agent.prepare_data()

        from graphs import GraphGenerator
        from utils import save_bfs_snapshot, trim_graph

        snapshot_dir = os.path.join(agent.saver.output_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)

        style = getattr(BaseConfig, "PLOT_STYLE", {})
        recorded_calls = []

        def on_snapshot(positions, edges, frame, step_idx):
            recorded_calls.append(step_idx)
            # Use low dpi for speed; pass node_size/line_width from PLOT_STYLE
            save_bfs_snapshot(
                positions,
                edges,
                frame,
                step_idx,
                snapshot_dir,
                dpi=72,
                node_size=style.get("node_size", 6.0),
                line_width=style.get("line_width", 3.0),
            )

        generator = GraphGenerator(agent.attributes)
        synth = generator.generate_network_with_snapshots(
            snapshot_callback=on_snapshot,
            snapshot_interval=BaseConfig.SNAPSHOT_INTERVAL,
        )
        synth = trim_graph(synth, agent.attributes.average_degree)
        agent.mapper.assign_weights(synth)

        assert synth.number_of_nodes() > 50
        assert synth.number_of_edges() > 50

        # Verify snapshot callback was invoked
        assert len(recorded_calls) > 0, "No snapshots were recorded"

        # Verify PNG files were created (fewer than calls due to BFS retries
        # overwriting snapshots from failed attempts)
        snapshot_files = [f for f in os.listdir(snapshot_dir) if f.endswith(".png")]
        assert len(snapshot_files) > 0, "No snapshot PNGs found"

        logger.info(
            f"Snapshot mode: {synth.number_of_nodes()} nodes, "
            f"{synth.number_of_edges()} edges, "
            f"{len(snapshot_files)} snapshots"
        )

    def test_compute_and_rank_metrics(self, tmp_path):
        """Generate a network and compute quality metrics for ranking."""
        from configs.generate_mode.config_sample import SampleConfig

        SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 0
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 0
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import BaseConfig
        from handlers import RunAgent, Saver
        from utils import compute_network_metrics, metric_distance

        Saver.initialize()
        dataset_id = BaseConfig.get_datasets()[0]
        agent = RunAgent(dataset_id=dataset_id)
        agent.prepare_data()

        original = agent.get_original_network()
        ref_metrics = compute_network_metrics(original)

        # Metrics should be well-formed
        assert ref_metrics["node_count"] > 0
        assert ref_metrics["avg_degree"] > 0

        # Distance to self should be zero
        assert metric_distance(ref_metrics, ref_metrics) == pytest.approx(0.0)

        logger.info(
            f"Metrics: nodes={ref_metrics['node_count']}, "
            f"avg_deg={ref_metrics['avg_degree']:.2f}, "
            f"avg_clust={ref_metrics['avg_clustering']:.4f}"
        )
