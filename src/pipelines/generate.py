# src/pipelines/generate.py
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from typing import List

import numpy as np

from analysis import MultifractalAnalyzer
from configs import BaseConfig, DatasetId
from configs.generate_mode import GenConfigSnapshot as GenConfig
from graphs import GraphGenerator
from handlers import AttributesCalculator, Mapper, RunAgent, Saver
from utils import save_bfs_snapshot, trim_graph

GenConfig.initialize()
# BaseConfig.disable_saving("debug")

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def _should_exit(exit_event) -> bool:
    if exit_event.is_set():
        logger.info(SIGINT_INFO)
        return True
    return False


def generate_synthetic_network(
    exit_event, std_err_fea, attributes: AttributesCalculator, mapper: Mapper
):
    if _should_exit(exit_event):
        return None, float("inf")

    generator = GraphGenerator(attributes)
    for attempt in range(BaseConfig.MAX_ATTEMPTS):
        try:
            synthetic_graph = generator.generate_network()
            synthetic_graph = trim_graph(synthetic_graph, attributes.average_degree)
            mapper.assign_weights(synthetic_graph)

            if _should_exit(exit_event):
                return None, float("inf")

            err_fea = MultifractalAnalyzer(synthetic_graph).analyze_error_features()
            error_ = MultifractalAnalyzer.analyze_error(err_fea, std_err_fea)
            if error_ < BaseConfig.ERROR_TOLERANCE:
                return synthetic_graph, error_

        except KeyboardInterrupt:
            logger.info(SIGINT_INFO)
            raise
        except Exception as exc:
            logger.error(f"Exception occurred: {exc}. Retrying...", exc_info=True)

    logger.warning("Max attempts reached. Aborting!")
    return None, float("inf")


def generate_with_multiprocessing(data_agent: RunAgent):
    num_network, num_figures = (
        BaseConfig.SYNTHETIC_NETWORK_NUMBER,
        BaseConfig.SYNTHETIC_GRAPH_NUMBER,
    )
    errors, futures = [], []
    from multiprocessing import Manager

    exit_event = Manager().Event()
    std_err_fea = (
        None
        if BaseConfig.SYNTHETIC_NETWORK_NUMBER == 0
        else MultifractalAnalyzer(
            data_agent.get_original_network()
        ).analyze_error_features()
    )

    max_workers = BaseConfig.get_max_workers(num_network)
    logger.info(f"Using {max_workers} worker(s) for {num_network} networks")

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    generate_synthetic_network,
                    exit_event,
                    std_err_fea,
                    data_agent.attributes,
                    data_agent.mapper,
                )
                for _ in range(num_network)
            ]
            next_log = 10
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error = future.result()
                if not synthetic_graph:
                    break

                progress = round(idx / num_network * 100, 2)
                if progress >= next_log:
                    logger.info(f"({progress}%) Synthetic graph generated.")
                    next_log += 10

                if (num_figures := num_figures - 1) >= 0:
                    data_agent.save("synthetic_graph", content=synthetic_graph)
                data_agent.add_synthetic_graph(synthetic_graph)
                errors.append(error)
    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        exit_event.set()
        for future in futures:
            future.cancel()

    avg_err = compute_average_error(errors)
    prefix = f"len_{len(errors)}_err_{avg_err:.3f}"
    data_agent.save_synthetic_outputs(prefix)

    if BaseConfig.FULL_ANALYSIS:
        data_agent.multifractal_analysis_in_generate_mode()
        data_agent.save("analysis_data")
        data_agent.save("analysis_figure")


def compute_average_error(errors: List) -> float:
    valid_errors = [e for e in errors if not np.isinf(e)]
    if not valid_errors:
        return float("inf")

    mean_e, std_e = np.mean(valid_errors), np.std(valid_errors)
    non_outliers = [e for e in valid_errors if abs(e - mean_e) <= 2.0 * std_e]
    return round(np.mean(non_outliers), 3) if non_outliers else float("inf")


def generate_with_snapshots(data_agent: RunAgent):
    """Generate a single network while saving intermediate BFS snapshots."""
    snapshot_dir = os.path.join(data_agent.saver.output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    interval = BaseConfig.SNAPSHOT_INTERVAL

    def on_snapshot(positions, edges, frame, step_idx):
        save_bfs_snapshot(positions, edges, frame, step_idx, snapshot_dir)

    generator = GraphGenerator(data_agent.attributes)
    synthetic_graph = generator.generate_network_with_snapshots(
        snapshot_callback=on_snapshot,
        snapshot_interval=interval,
    )
    synthetic_graph = trim_graph(synthetic_graph, data_agent.attributes.average_degree)
    data_agent.mapper.assign_weights(synthetic_graph)

    data_agent.save("synthetic_graph", content=synthetic_graph)
    data_agent.add_synthetic_graph(synthetic_graph)
    data_agent.save_synthetic_outputs(prefix="snapshot_run")

    logger.info(f"Snapshot generation complete. Snapshots saved to {snapshot_dir}")


def run_for_dataset(dataset_id: DatasetId):
    """Process a single dataset identified by DatasetId."""
    logger.info(f"Processing dataset: {dataset_id}")
    logger.info(BaseConfig())
    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    data_agent.save("original_image")
    data_agent.save("original_network")
    data_agent.save("original_property")
    data_agent.save("original_graph")

    if BaseConfig.SNAPSHOT_INTERVAL > 0:
        generate_with_snapshots(data_agent)
    else:
        generate_with_multiprocessing(data_agent)


def main():
    try:
        Saver.initialize()
        for dataset_id in BaseConfig.get_datasets():
            run_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
