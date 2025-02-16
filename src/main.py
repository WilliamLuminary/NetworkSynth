# src/main.py
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import List

import numpy as np

from analysis import MultifractalAnalyzer
from config import Config, DataType, Resolution, SetName
# noinspection PyUnresolvedReferences
from config import Config1, Config2, ConfigSample
from graph import GraphAttrAgent, GraphGenerator
from handlers import DataAgent, Mapper, Saver
from utils.utils import trim_graph

ConfigSample.initialize()

preview = False
if preview:
    Config.disable_saving("Preview")

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def _should_exit(exit_event) -> bool:
    if exit_event.is_set():
        logger.info(SIGINT_INFO)
        return True
    return False


def generate_synthetic_network(exit_event, std_err_fea, attributes: GraphAttrAgent, mapper: Mapper):
    if _should_exit(exit_event):
        return None, float('inf')

    generator = GraphGenerator(attributes)
    for attempt in range(Config.MAX_ATTEMPTS):
        try:
            synthetic_graph = generator.generate_network()
            synthetic_graph = trim_graph(synthetic_graph, attributes.average_degree)
            mapper.assign_weights(synthetic_graph)

            if _should_exit(exit_event):
                return None, float('inf')

            err_fea = MultifractalAnalyzer(synthetic_graph).analyze_error_features()
            error_ = MultifractalAnalyzer.analyze_error(err_fea, std_err_fea)
            if error_ < Config.ERROR_TOLERANCE:
                return synthetic_graph, error_

        except KeyboardInterrupt:
            logger.info(SIGINT_INFO)
            raise
        except Exception as exc:
            logger.error(f"Exception occurred: {exc}. Retrying...")

    logger.warning("Max attempts reached. Aborting!")
    return None, float('inf')


def generate_with_multiprocessing(data_agent: DataAgent, std_err_fea):
    num_network, num_figures = Config.SYNTHETIC_NETWORK_NUMBER, Config.SYNTHETIC_GRAPH_NUMBER
    errors, futures = [], []
    from multiprocessing import Manager
    exit_event = Manager().Event()
    try:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(generate_synthetic_network, exit_event, std_err_fea,
                                       data_agent.attributes, data_agent.mapper) for _ in range(num_network)]
            next_log = 0
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error = future.result()
                if not synthetic_graph:
                    break

                if (progress := round(idx / num_network * 100, 2)) >= (next_log := next_log + 10):
                    logger.info(f"({progress}%) Synthetic graph generated.")

                if (num_figures := num_figures - 1) >= 0:
                    data_agent.save(data_type=DataType.SYNTHETIC_GRAPH, arg=synthetic_graph)
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
        executor.shutdown(wait=True, cancel_futures=True)

    avg_err = _compute_average_error(errors)
    data_agent.save(data_type=DataType.SYNTHETIC_NETWORK, file_name_prefix=f'len_{len(errors)}_err_{avg_err:.3f}')
    if Config.FULL_ANALYSIS:
        data_agent.multifractal_analysis_in_generate_mode()
        data_agent.save(data_type=DataType.ANALYSIS_DATA)
        data_agent.save(data_type=DataType.ANALYSIS_FIGURE)

def _compute_average_error(errors: List) -> float:
    valid_errors = [e for e in errors if not np.isinf(e)]
    if not valid_errors:
        return float('inf')

    mean_e, std_e = np.mean(valid_errors), np.std(valid_errors)
    non_outliers = [e for e in valid_errors if abs(e - mean_e) <= 2.0 * std_e]
    return round(np.mean(non_outliers), 3) if non_outliers else float('inf')


def run_for_each_set_resolution(set_name: SetName, resolution: Resolution):
    logger.info(Config())
    data_agent = DataAgent(set_name, resolution)
    data_agent.prepare_data()
    data_agent.save(DataType.ORIGINAL_IMAGE)
    data_agent.save(DataType.ORIGINAL_NETWORK)
    data_agent.save(DataType.ORIGINAL_PROPERTY)
    data_agent.save(DataType.ORIGINAL_GRAPH)

    std_err_fea = None if Config.SYNTHETIC_NETWORK_NUMBER == 0 else MultifractalAnalyzer(
        data_agent.get_original_network()).analyze_error_features()
    generate_with_multiprocessing(data_agent, std_err_fea)


def main():
    try:
        Saver.initialize()
        for set_name_, resolution_ in product(Config.SETS, Config.RESOLUTIONS):
            run_for_each_set_resolution(set_name_, resolution_)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == '__main__':
    main()
