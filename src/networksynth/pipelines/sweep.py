# SPDX-License-Identifier: GPL-3.0-or-later
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from networksynth.analysis.error_checker import ErrorChecker, create_error_checker
from networksynth.configs import DatasetId, SynthParams
from networksynth.handlers import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    GenerationRun,
    attach_run_log,
    create_run_paths,
    write_manifest,
)
from networksynth.pipelines.generate import compute_average_error, generate_candidate
from networksynth.utils import spawn_context, tagged


def _init_config():
    from dataclasses import replace

    from networksynth.configs.sweep_mode.config_sample import CONFIG

    return replace(CONFIG, SYNTHETIC_NETWORK_NUMBER=100)


EXPERIMENT_PROJECT_NAME = "hyperparam-tuning"


def _wandb():
    import wandb

    return wandb


def _build_factors(lo, hi, step):
    n = round((hi - lo) / step) + 1
    return [round(lo + step * i, 6) for i in range(n)]


logger = logging.getLogger(__name__)


def generate_networks(run: GenerationRun, error_checker: ErrorChecker, nf, ef, config):
    num_network = config.SYNTHETIC_NETWORK_NUMBER
    exit_event = spawn_context().Manager().Event()
    max_workers = config.get_max_workers(num_network)

    trial_params = SynthParams(
        synthetic_frame_size=config.SYNTHETIC_FRAME_SIZE,
        closed_nodes_factor=nf,
        closed_edges_factor=ef,
        max_attempts=config.MAX_ATTEMPTS,
        seed=config.SEED,
    )

    errors = []
    futures = []
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn_context()
        ) as executor:
            futures = [
                executor.submit(
                    generate_candidate,
                    exit_event,
                    error_checker,
                    run.attributes,
                    run.mapper,
                    trial_params.for_worker(i),
                )
                for i in range(num_network)
            ]
            next_log = 10
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error, _ = future.result()
                if synthetic_graph is None:
                    continue

                progress = round(idx / num_network * 100, 2)
                if progress >= next_log:
                    logger.info(
                        f"({progress}%) Synthetic graph generated.",
                        extra=tagged("PROGRESS", percent=progress),
                    )
                    next_log += 10

                errors.append(error)
    except KeyboardInterrupt:
        logger.info("SIGINT received. Terminating child processes...")
        raise
    finally:
        exit_event.set()
        for f in futures:
            f.cancel()

    success_rate = len(errors) / num_network if num_network > 0 else 0.0
    avg_error = compute_average_error(errors)
    return avg_error, success_rate


def _report_rows(results) -> list:
    rows = [["node_factor", "edge_factor", "error", "success_rate"]]
    for nf, ef, error, success_rate in sorted(results):
        rows.append([nf, ef, f"{error:.6f}", f"{success_rate:.4f}"])
    return rows


def run_for_dataset(
    dataset_id: DatasetId, node_factors, edge_factors, config, run_paths
) -> None:
    logger.info(f"Processing dataset: {dataset_id}")
    logger.info(
        f"  nf: {node_factors[0]}-{node_factors[-1]} ({len(node_factors)} values)"
    )
    logger.info(
        f"  ef: {edge_factors[0]}-{edge_factors[-1]} ({len(edge_factors)} values)"
    )
    logger.info(f"  total trials: {len(node_factors) * len(edge_factors)}")

    run = GenerationRun(config, run_paths, dataset_id)
    error_checker = create_error_checker(config)
    error_checker.compute_reference(run.original)

    sweep_config = {
        "name": f"sweep-{dataset_id}",
        "method": "grid",
        "metric": {"name": "error", "goal": "minimize"},
        "parameters": {
            "node_factor": {"values": node_factors},
            "edge_factor": {"values": edge_factors},
        },
    }

    results = []

    def run_trial(nf, ef):
        logger.info(f"Trial: nf={nf}, ef={ef}")
        error, success_rate = generate_networks(run, error_checker, nf, ef, config)
        if error is None:
            error = float("inf")
        results.append((nf, ef, error, success_rate))
        return error, success_rate

    if config.USE_WANDB:
        wandb = _wandb()

        def trial():
            with wandb.init():
                error, success_rate = run_trial(
                    wandb.config.node_factor, wandb.config.edge_factor
                )
                wandb.log({"error": error, "success_rate": success_rate})

        sweep_id = wandb.sweep(sweep_config, project=EXPERIMENT_PROJECT_NAME)
        wandb.agent(sweep_id, function=trial)
    else:
        logger.info("wandb is off — walking the grid locally.")
        for nf in node_factors:
            for ef in edge_factors:
                run_trial(nf, ef)

    run.save(_report_rows(results), "sweep_report")
    logger.info(f"Sweep report written for {len(results)} trial(s).")


def main(config=None):
    cfg = _init_config() if config is None else config

    nf_lo, nf_hi = cfg.NF_RANGE
    ef_lo, ef_hi = cfg.EF_RANGE
    step = cfg.SWEEP_STEP
    node_factors = _build_factors(nf_lo, nf_hi, step)
    edge_factors = _build_factors(ef_lo, ef_hi, step)

    run_paths = create_run_paths(cfg)
    attach_run_log(run_paths.root, cfg.RUN_ID)
    status, error = STATUS_OK, None
    try:
        assert (
            cfg.SYNTHETIC_NETWORK_NUMBER != 0
        ), "Sweeping experiments require synthetic networks."
        if cfg.USE_WANDB:
            _wandb().login(
                key=os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
            )
        for dataset_id in cfg.DATASETS:
            run_for_dataset(dataset_id, node_factors, edge_factors, cfg, run_paths)
    except KeyboardInterrupt:
        status = STATUS_CANCELLED
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")
        raise
    except Exception as exc:
        status = STATUS_FAILED
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not cfg.DISABLE_SAVING:
            write_manifest(run_paths, status=status, error=error)


if __name__ == "__main__":
    main()
