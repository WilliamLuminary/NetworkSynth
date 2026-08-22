import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging

import networkit as nk

from configs import AttrConfig, SynthParams
from configs.base_config import tagged
from graphs import GraphGenerator
from handlers import AttributesCalculator, RunAgent
from utils import apply_seed, spawn_context, trim_graph

AttrConfig.initialize()
# Config.DISABLE_SAVING = True  # preview only

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def _should_exit(exit_event) -> bool:
    if exit_event.is_set():
        logger.info(SIGINT_INFO)
        return True
    return False


def generate_synthetic_network(
    exit_event, attributes: AttributesCalculator, params: SynthParams
):
    # See generate_synthetic_network: a forked child inherits an OpenMP runtime
    # whose threads do not exist in it and spins instead of working.
    nk.setNumberOfThreads(1)

    if _should_exit(exit_event):
        return None

    apply_seed(params.seed)
    generator = GraphGenerator(attributes, params)
    for attempt in range(params.max_attempts):
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


def generate_with_multiprocessing(data_agent: RunAgent, config):
    num_network, num_figures = (
        config.SYNTHETIC_NETWORK_NUMBER,
        config.SYNTHETIC_GRAPH_NUMBER,
    )
    futures = []

    exit_event = spawn_context().Manager().Event()
    max_workers = config.get_max_workers(num_network)
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn_context()
        ) as executor:
            base_params = SynthParams.from_config(config)
            futures = [
                executor.submit(
                    generate_synthetic_network,
                    exit_event,
                    data_agent.attributes,
                    base_params.for_worker(i),
                )
                for i in range(num_network)
            ]
            next_log = 0
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph = future.result()
                if not synthetic_graph:
                    break

                if (progress := round(idx / num_network * 100, 2)) >= (
                    next_log := next_log + 10
                ):
                    logger.info(
                        f"({progress}%) Synthetic graph generated.",
                        extra=tagged("PROGRESS", percent=progress),
                    )

                if (num_figures := num_figures - 1) >= 0:
                    data_agent.save("synthetic_graph", content=synthetic_graph)
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


def run(config):
    logger.info(config())
    data_agent = RunAgent(config, attr_path=config.ATTRIBUTES_DICT_DATA_PATH)
    data_agent.prepare_data()

    generate_with_multiprocessing(data_agent, config)


def main(config_cls=AttrConfig):
    config_cls.initialize()
    try:
        run(config_cls)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")
        # Re-raised so the entry point can exit non-zero: a cancelled
        # run must not look like a completed one to a caller.
        raise


if __name__ == "__main__":
    main()
