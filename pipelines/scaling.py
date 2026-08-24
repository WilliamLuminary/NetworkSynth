import logging

from configs import SynthParams
from configs.scaling_mode import ScalingConfig
from graphs import GraphGenerator
from handlers import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    GenerationRun,
    attach_run_log,
    create_run_paths,
    write_manifest,
)
from utils import apply_seed, render_network, trim_graph

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Main pipeline for one dataset
# ------------------------------------------------------------------ #
def run_scaling_for_dataset(dataset_id, config, run_paths):
    logger.info(f"=== Scaling pipeline for dataset: {dataset_id} ===")
    logger.info(config())

    # 1. Load original network → compute structural attributes
    run = GenerationRun(config, run_paths, dataset_id)
    attributes = run.attributes

    # 2. Generate scaled network
    params = SynthParams.from_config(config)
    apply_seed(params.seed)
    generator = GraphGenerator(attributes, params)
    scaled_graph = generator.generate_scaled_network(
        scale_rows=config.SCALE_ROWS,
        scale_cols=config.SCALE_COLS,
        max_rounds=config.MAX_GENERATION_ROUNDS,
        root_spacing_factor=config.ROOT_SPACING_FACTOR,
    )

    # 3. Trim to match average degree
    scaled_graph = trim_graph(scaled_graph, attributes.average_degree)

    # 4. Assign edge weights from original network's length→weight distribution
    run.mapper.assign_weights(scaled_graph)

    # 5. Save network data + plot
    run.add_synthetic_graph(scaled_graph)
    prefix = f"scaled_{config.SCALE_ROWS}x{config.SCALE_COLS}"

    run.saver.begin_batch()
    run.save(scaled_graph, "synthetic_export", f"{prefix}_")
    scaled_img = render_network(
        scaled_graph, max_px=getattr(config, "RENDER_MAX_PX", None)
    )
    run.save(scaled_img, "synthetic_graph", f"{prefix}_")
    run.saver.end_batch()

    logger.info(
        f"Scaling complete — "
        f"{scaled_graph.number_of_nodes():,} nodes, "
        f"{scaled_graph.number_of_edges():,} edges"
    )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main(config_cls=ScalingConfig):
    config_cls.initialize()
    # Built before the try so the finally below can always name the run, even if
    # the very first dataset fails.
    run_paths = create_run_paths(config_cls)
    attach_run_log(run_paths.root, config_cls.RUN_ID)
    status, error = STATUS_OK, None
    try:
        for dataset_id in config_cls.get_datasets():
            run_scaling_for_dataset(dataset_id, config_cls, run_paths)
    except KeyboardInterrupt:
        status = STATUS_CANCELLED
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")
        # Re-raised so the entry point can exit non-zero: a cancelled
        # run must not look like a completed one to a caller.
        raise
    except Exception as exc:
        status = STATUS_FAILED
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Written even on cancel/failure: a caller must be able to tell
        # "manifest says cancelled" from "no manifest, we died hard".
        if not config_cls.DISABLE_SAVING:
            write_manifest(run_paths, status=status, error=error)


if __name__ == "__main__":
    main()
