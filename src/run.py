# src/run.py
"""
Unified entry point for all NetworkSynth pipelines.

Usage
-----
    python run.py generate       Standard network generation (parallel, quality-checked)
    python run.py from_props     Generate from pre-computed structural attributes
    python run.py mosaic         Mosaic: parallel tiles + stitch
    python run.py scaling        Scaling: multi-root synchronized BFS
    python run.py hybrid         Hybrid: parallel seed tiles + frontier continuation
    python run.py sweep          Hyperparameter sweep (wandb)
    python run.py analyze        Multifractal analysis on existing results
"""
import argparse


def cmd_generate(_args):
    from pipelines.generate import main

    main()


def cmd_from_props(_args):
    from pipelines.generate_from_props import main

    main()


def cmd_mosaic(_args):
    from pipelines.mosaic import main

    main()


def cmd_scaling(_args):
    from pipelines.scaling import main

    main()


def cmd_hybrid(args):
    config_cls = None
    config_name = getattr(args, "config", None)
    if config_name:
        import configs.hybrid_mode as hm

        cls_name = f"HybridConfig{config_name.capitalize()}"
        config_cls = getattr(hm, cls_name, None)
        if config_cls is None:
            available = [n for n in dir(hm) if n.startswith("HybridConfig")]
            raise SystemExit(
                f"Unknown hybrid config '{config_name}'. " f"Available: {available}"
            )

    dataset = getattr(args, "dataset", None)
    if dataset:
        if config_cls is None:
            from configs.hybrid_mode import HybridConfig as _HC

            config_cls = _HC
        from configs.enums import DatasetId

        config_cls.DATASETS = [DatasetId(f"sample_{dataset}")]

    from pipelines.hybrid import main

    main(config_cls=config_cls)


def cmd_sweep(args):
    from pipelines.sweep import main

    main(
        config=args.config,
        nf_range=args.nf_range,
        ef_range=args.ef_range,
    )


def cmd_analyze(_args):
    from pipelines.analyze import main

    main()


COMMANDS = {
    "generate": ("Standard network generation", cmd_generate),
    "from_props": ("Generate from pre-computed attributes", cmd_from_props),
    "mosaic": ("Mosaic: parallel tiles + stitch", cmd_mosaic),
    "scaling": ("Scaling: multi-root synchronized BFS", cmd_scaling),
    "hybrid": ("Hybrid: seed tiles + frontier continuation", cmd_hybrid),
    "sweep": ("Hyperparameter sweep (wandb)", cmd_sweep),
    "analyze": ("Multifractal analysis on existing results", cmd_analyze),
}


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="NetworkSynth — unified pipeline runner",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, (help_text, func) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)

        if name == "hybrid":
            p.add_argument(
                "--config",
                default=None,
                help=(
                    "Hybrid config variant (e.g. 'snapshot'). "
                    "Omit for default. Maps to HybridConfig<Name>."
                ),
            )
            p.add_argument(
                "--dataset",
                choices=["A", "B", "C", "D"],
                default=None,
                help="Run for a single dataset (A/B/C/D). Omit to run all.",
            )

        if name == "sweep":
            p.add_argument(
                "--config",
                choices=["A", "B", "C", "D"],
                default=None,
                help="Run sweep for a single dataset (A/B/C/D). Omit to sweep all.",
            )
            p.add_argument(
                "--nf-range",
                nargs=2,
                type=float,
                metavar=("MIN", "MAX"),
                help="Override node factor range (e.g. --nf-range 2.1 3.0)",
            )
            p.add_argument(
                "--ef-range",
                nargs=2,
                type=float,
                metavar=("MIN", "MAX"),
                help="Override edge factor range (e.g. --ef-range 2.1 3.0)",
            )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
