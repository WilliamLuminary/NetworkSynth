# run.py
"""
Unified entry point for all NetworkSynth pipelines.

Usage
-----
    python run.py generate                          Standard generation (default config)
    python run.py generate --config Snapshot1x1     Single network + BFS snapshots
    python run.py generate_select --config Snapshot3x3   Generate 20, rank, pick top 3
    python run.py from_props                        Generate from pre-computed attributes
    python run.py mosaic                            Mosaic: parallel tiles + stitch
    python run.py scaling                           Scaling: multi-root synchronized BFS
    python run.py hybrid                            Hybrid: seed tiles + frontier continuation
    python run.py hybrid --config snapshot          Hybrid config variant
    python run.py hybrid --dataset A                Single dataset
    python run.py sweep                             Hyperparameter sweep (wandb)
    python run.py sweep --config A --nf_range '(2.1,3.0)' --ef_range '(2.1,3.0)'
    python run.py analyze                           Multifractal analysis on existing results
"""
import logging
import sys

import fire

from handlers import configure_console


class CLI:
    """NetworkSynth — unified pipeline runner."""

    def generate(self, config: str = None):
        """Standard network generation (parallel, quality-checked).

        Args:
            config: Config variant name, e.g. 'Snapshot1x1', 'Snapshot3x3'.
                    Maps to GenConfig<Name>. Omit for default (Snapshot).
        """
        config_cls = None
        if config:
            import configs.generate_mode as gm

            cls_name = f"GenConfig{config}"
            config_cls = getattr(gm, cls_name, None)
            if config_cls is None:
                available = [n for n in dir(gm) if n.startswith("GenConfig")]
                raise SystemExit(
                    f"Unknown generate config '{config}'. " f"Available: {available}"
                )

        from pipelines.generate import main

        main(config_cls=config_cls)

    def generate_select(self, config: str = None):
        """Generate N candidates, rank by metrics, select top K.

        Args:
            config: Config variant name, e.g. 'Snapshot3x3'.
                    Maps to GenConfig<Name>. Omit for default (Snapshot).
        """
        config_cls = None
        if config:
            import configs.generate_mode as gm

            cls_name = f"GenConfig{config}"
            config_cls = getattr(gm, cls_name, None)
            if config_cls is None:
                available = [n for n in dir(gm) if n.startswith("GenConfig")]
                raise SystemExit(
                    f"Unknown generate config '{config}'. " f"Available: {available}"
                )

        from pipelines.generate_select import main

        main(config_cls=config_cls)

    def from_props(self):
        """Generate from pre-computed structural attributes."""
        from pipelines.generate_from_props import main

        main()

    def mosaic(self):
        """Mosaic: parallel tiles + stitch."""
        from pipelines.mosaic import main

        main()

    def scaling(self):
        """Scaling: multi-root synchronized BFS."""
        from pipelines.scaling import main

        main()

    def hybrid(self, config: str = None, dataset: str = None):
        """Hybrid: parallel seed tiles + frontier continuation.

        Args:
            config: Config variant name (e.g. 'snapshot').
                    Maps to HybridConfig<Name>. Omit for default.
            dataset: Single dataset letter (A/B/C/D). Omit to run all.
        """
        config_cls = None
        if config:
            import configs.hybrid_mode as hm

            cls_name = f"HybridConfig{config.capitalize()}"
            config_cls = getattr(hm, cls_name, None)
            if config_cls is None:
                available = [n for n in dir(hm) if n.startswith("HybridConfig")]
                raise SystemExit(
                    f"Unknown hybrid config '{config}'. " f"Available: {available}"
                )

        if dataset:
            if config_cls is None:
                from configs.hybrid_mode import HybridConfig as _HC

                config_cls = _HC
            from configs.enums import DatasetId

            config_cls.DATASETS = [DatasetId(f"sample_{dataset}")]

        from pipelines.hybrid import main

        main(config_cls=config_cls)

    def sweep(self, nf_range: tuple = None, ef_range: tuple = None):
        """Hyperparameter sweep (wandb) over every dataset in the sweep config.

        Args:
            nf_range: Node factor range as tuple, e.g. '(2.1, 3.0)'.
                      Defaults to the config's NF_RANGE.
            ef_range: Edge factor range as tuple, e.g. '(2.1, 3.0)'.
                      Defaults to the config's EF_RANGE.
        """
        from pipelines.sweep import main

        main(nf_range=nf_range, ef_range=ef_range)

    def analyze(self):
        """Multifractal analysis on existing results."""
        from pipelines.analyze import main

        main()


# SIGINT exit code. 128 + SIGINT(2), the conventional value for a process
# terminated by Ctrl-C.  Note `fire` swallows SystemExit and reports 2, so the
# handler must live out here rather than inside a pipeline.
_EXIT_INTERRUPTED = 130


def main() -> None:
    """Run the CLI, mapping cancellation to a non-zero exit code.

    Pipelines log their shutdown and re-raise; deciding the process exit status
    is the entry point's job.  Without this a cancelled run exits 0 and any
    caller — notably a GUI launching us as a subprocess — reads it as success.
    """
    configure_console()
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        logging.getLogger(__name__).critical("Interrupted — exiting.")
        sys.exit(_EXIT_INTERRUPTED)


if __name__ == "__main__":
    main()
