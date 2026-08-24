import logging
import os

import pytest

pytestmark = pytest.mark.integration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "input", "samples")


# ------------------------------------------------------------------ #
# Generate mode
# ------------------------------------------------------------------ #


class TestGenerateMode:
    def test_load_original_and_save(self, tmp_path):
        from configs.generate_mode.config_sample import SampleConfig

        SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 0
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 0
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from handlers import GenerationRun, create_run_paths

        run_paths = create_run_paths(SampleConfig)
        dataset_id = SampleConfig.get_datasets()[0]
        agent = GenerationRun(SampleConfig, run_paths, dataset_id)

        agent.save_original()

        out = agent.saver.output_dir
        assert os.path.isdir(out), f"Output dir not created: {out}"
        files = []
        for root, _, fnames in os.walk(out):
            files.extend(fnames)
        assert len(files) >= 3, f"Expected ≥3 output files, got {files}"
        logger.info(f"Generate (originals): {len(files)} files in {out}")

    def test_generate_one_network(self, tmp_path):
        from configs.generate_mode.config_sample import SampleConfig

        SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 1
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 1
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import SynthParams
        from graphs import GraphGenerator
        from handlers import GenerationRun, create_run_paths
        from utils import trim_graph

        run_paths = create_run_paths(SampleConfig)
        dataset_id = SampleConfig.get_datasets()[0]
        agent = GenerationRun(SampleConfig, run_paths, dataset_id)

        generator = GraphGenerator(
            agent.attributes, SynthParams.from_config(SampleConfig)
        )
        synth = generator.generate_network()
        synth = trim_graph(synth, agent.attributes.average_degree)
        agent.mapper.assign_weights(synth)

        assert synth.number_of_nodes() > 50
        assert synth.number_of_edges() > 50

        agent.add_synthetic_graph(synth)
        agent.save_synthetic_plot(synth)
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
# Mosaic mode
# ------------------------------------------------------------------ #


class TestMosaicMode:
    def test_mosaic_2x2(self, tmp_path):
        from configs.mosaic_mode.config_sample import SampleConfig

        SampleConfig.GRID_ROWS = 2
        SampleConfig.GRID_COLS = 2
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import SynthParams
        from graphs import GraphGenerator
        from graphs.mosaic_stitcher import MosaicStitcher
        from handlers import GenerationRun, create_run_paths
        from utils import trim_graph

        run_paths = create_run_paths(SampleConfig)
        dataset_id = SampleConfig.get_datasets()[0]
        agent = GenerationRun(SampleConfig, run_paths, dataset_id)
        attributes = agent.attributes

        tile_w, tile_h = SampleConfig.TILE_FRAME_SIZE
        overlap = SampleConfig.OVERLAP_MARGIN_FRACTION
        tile_gen_frame = (
            round(tile_w + 2 * overlap * tile_w),
            round(tile_h + 2 * overlap * tile_h),
        )

        tile_graphs = {}
        for r in range(SampleConfig.GRID_ROWS):
            for c in range(SampleConfig.GRID_COLS):
                gen = GraphGenerator(attributes, SynthParams.from_config(SampleConfig))
                g = gen.generate_network(frame_range=tile_gen_frame)
                g = trim_graph(g, attributes.average_degree)
                positions = g.positions()
                positions[:, 0] += c * tile_w + tile_w / 2.0
                positions[:, 1] += r * tile_h + tile_h / 2.0
                tile_graphs[(r, c)] = g

        merge_threshold = attributes.average_length * SampleConfig.CLOSED_NODES_FACTOR
        stitcher = MosaicStitcher(merge_threshold=merge_threshold)
        mosaic = stitcher.stitch(tile_graphs)
        agent.mapper.assign_weights(mosaic)

        assert mosaic.number_of_nodes() > 100
        assert mosaic.number_of_edges() > 100

        agent.saver.begin_batch()
        agent.saver.save(mosaic, "synthetic_export", "mosaic_2x2_")
        agent.saver.end_batch()

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
        from configs.scaling_mode.config_sample import SampleConfig

        SampleConfig.SCALE_ROWS = 2
        SampleConfig.SCALE_COLS = 2
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from configs import SynthParams
        from graphs import GraphGenerator
        from handlers import GenerationRun, create_run_paths
        from utils import trim_graph

        run_paths = create_run_paths(SampleConfig)
        dataset_id = SampleConfig.get_datasets()[0]
        agent = GenerationRun(SampleConfig, run_paths, dataset_id)

        gen = GraphGenerator(agent.attributes, SynthParams.from_config(SampleConfig))
        scaled = gen.generate_scaled_network(
            scale_rows=SampleConfig.SCALE_ROWS,
            scale_cols=SampleConfig.SCALE_COLS,
            max_rounds=SampleConfig.MAX_GENERATION_ROUNDS,
            root_spacing_factor=SampleConfig.ROOT_SPACING_FACTOR,
        )
        scaled = trim_graph(scaled, agent.attributes.average_degree)
        agent.mapper.assign_weights(scaled)

        assert scaled.number_of_nodes() > 200
        assert scaled.number_of_edges() > 200

        agent.saver.begin_batch()
        agent.saver.save(scaled, "synthetic_export", "scaled_2x2_")
        agent.saver.end_batch()

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
        from configs.generate_mode.config_sample import SampleConfig

        SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 0
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 0
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from analysis import MultifractalAnalyzer
        from handlers import GenerationRun, create_run_paths

        run_paths = create_run_paths(SampleConfig)
        dataset_id = SampleConfig.get_datasets()[0]
        agent = GenerationRun(SampleConfig, run_paths, dataset_id)

        original = agent.original
        analyzer = MultifractalAnalyzer(
            original, SampleConfig.MEASURE_WEIGHTED, SampleConfig.FULL_Q_BAND
        )
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
        from configs.generate_mode.config_snapshot_1x1 import Snapshot1x1Config

        Snapshot1x1Config.BASE_OUTPUT_PATH = str(tmp_path)
        Snapshot1x1Config.initialize()

        from configs import SynthParams
        from handlers import GenerationRun, create_run_paths

        run_paths = create_run_paths(Snapshot1x1Config)
        dataset_id = Snapshot1x1Config.get_datasets()[0]
        agent = GenerationRun(Snapshot1x1Config, run_paths, dataset_id)

        from graphs import GraphGenerator
        from utils import save_bfs_snapshot, trim_graph

        snapshot_dir = os.path.join(agent.saver.output_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)

        style = getattr(Snapshot1x1Config, "PLOT_STYLE", {})
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

        generator = GraphGenerator(
            agent.attributes, SynthParams.from_config(Snapshot1x1Config)
        )
        synth = generator.generate_network_with_snapshots(
            snapshot_callback=on_snapshot,
            snapshot_interval=Snapshot1x1Config.SNAPSHOT_INTERVAL,
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
        from configs.generate_mode.config_sample import SampleConfig

        SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
        SampleConfig.SYNTHETIC_NETWORK_NUMBER = 0
        SampleConfig.SYNTHETIC_GRAPH_NUMBER = 0
        SampleConfig.BASE_OUTPUT_PATH = str(tmp_path)
        SampleConfig.initialize()

        from handlers import GenerationRun, create_run_paths
        from utils import compute_network_metrics, metric_distance

        run_paths = create_run_paths(SampleConfig)
        dataset_id = SampleConfig.get_datasets()[0]
        agent = GenerationRun(SampleConfig, run_paths, dataset_id)

        original = agent.original
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
