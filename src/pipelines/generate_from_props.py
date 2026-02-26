# src/pipelines/generate_from_props.py
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging

from config import AttrConfig, BaseConfig, DataType
from graph import GraphGenerator
from handlers import AttributesCalculator, RunAgent
from utils import trim_graph

AttrConfig.initialize()
# Config.disable_saving("Preview")

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def _should_exit(exit_event) -> bool:
    if exit_event.is_set():
        logger.info(SIGINT_INFO)
        return True
    return False


def generate_synthetic_network(exit_event, attributes: AttributesCalculator):
    if _should_exit(exit_event):
        return None

    generator = GraphGenerator(attributes)
    for attempt in range(BaseConfig.MAX_ATTEMPTS):
        try:
            synthetic_graph = generator.generate_network()

            if _should_exit(exit_event):
                return None

            synthetic_graph = trim_graph(synthetic_graph, attributes.average_degree)
            return synthetic_graph

        except KeyboardInterrupt:
            logger.info(SIGINT_INFO)
            raise
        except Exception as exc:
            logger.error(f"Exception occurred: {exc}. Retrying...")

    logger.warning("Max attempts reached. Aborting!")
    return None


def generate_with_multiprocessing(data_agent: RunAgent):
    num_network, num_figures = (
        BaseConfig.SYNTHETIC_NETWORK_NUMBER,
        BaseConfig.SYNTHETIC_GRAPH_NUMBER,
    )
    futures = []
    from multiprocessing import Manager

    exit_event = Manager().Event()
    max_workers = BaseConfig.get_max_workers(num_network)
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    generate_synthetic_network, exit_event, data_agent.attributes
                )
                for _ in range(num_network)
            ]
            next_log = 0
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph = future.result()
                if not synthetic_graph:
                    break

                if (progress := round(idx / num_network * 100, 2)) >= (
                    next_log := next_log + 10
                ):
                    logger.info(f"({progress}%) Synthetic graph generated.")

                if (num_figures := num_figures - 1) >= 0:
                    data_agent.save(
                        data_type=DataType.SYNTHETIC_GRAPH, arg=synthetic_graph
                    )
                data_agent.add_synthetic_graph(synthetic_graph)

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

    data_agent.save_synthetic_outputs("")


def run():
    logger.info(BaseConfig())
    data_agent = RunAgent(attr_path=BaseConfig.ATTRIBUTES_DICT_DATA_PATH)
    data_agent.prepare_data()

    generate_with_multiprocessing(data_agent)


def main():
    try:
        run()
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
