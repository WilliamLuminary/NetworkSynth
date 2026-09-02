# SPDX-License-Identifier: GPL-3.0-or-later
import logging

from networksynth.configs import SynthParams
from networksynth.configs.scaling_mode.config_sample import (
    SampleConfig as ScalingConfig,
)
from networksynth.graphs import GraphGenerator
from networksynth.handlers import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    GenerationRun,
    attach_run_log,
    create_run_paths,
    write_manifest,
)
from networksynth.utils import apply_seed, render_network, trim_graph

logger = logging.getLogger(__name__)


def run_scaling_for_dataset(dataset_id, config, run_paths):
    logger.info(f"=== Scaling pipeline for dataset: {dataset_id} ===")
    logger.info(config())

    run = GenerationRun(config, run_paths, dataset_id)
    attributes = run.attributes

    params = SynthParams.from_config(config)
    apply_seed(params.seed)
    generator = GraphGenerator(attributes, params)
    scaled_graph = generator.generate_scaled_network(
        scale_rows=config.SCALE_ROWS,
        scale_cols=config.SCALE_COLS,
        max_rounds=config.MAX_GENERATION_ROUNDS,
        root_spacing_factor=config.ROOT_SPACING_FACTOR,
    )

    scaled_graph = trim_graph(scaled_graph, attributes.average_degree)

    run.mapper.assign_weights(scaled_graph)

    run.add_synthetic_graph(scaled_graph)
    prefix = f"scaled_{config.SCALE_ROWS}x{config.SCALE_COLS}"

    run.saver.begin_batch()
    run.save(scaled_graph, "synthetic_network", f"{prefix}_")
    scaled_img = render_network(scaled_graph, config.render("scaled_graph"))
    run.save(scaled_img, "synthetic_graph", f"{prefix}_")
    run.saver.end_batch()

    logger.info(
        f"Scaling complete — "
        f"{scaled_graph.number_of_nodes():,} nodes, "
        f"{scaled_graph.number_of_edges():,} edges"
    )


def main(config_cls=ScalingConfig):
    config_cls.initialize()
    run_paths = create_run_paths(config_cls)
    attach_run_log(run_paths.root, config_cls.RUN_ID)
    status, error = STATUS_OK, None
    try:
        for dataset_id in config_cls.get_datasets():
            run_scaling_for_dataset(dataset_id, config_cls, run_paths)
    except KeyboardInterrupt:
        status = STATUS_CANCELLED
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")
        raise
    except Exception as exc:
        status = STATUS_FAILED
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not config_cls.DISABLE_SAVING:
            write_manifest(run_paths, status=status, error=error)


if __name__ == "__main__":
    main()
