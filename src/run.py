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


def cmd_hybrid(_args):
    from pipelines.hybrid import main

    main()


def cmd_sweep(args):
    from pipelines.sweep import main

    main(config=args.config)


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

        if name == "sweep":
            p.add_argument(
                "--config",
                choices=["A", "B", "C", "D"],
                default=None,
                help="Run sweep for a single dataset (A/B/C/D). Omit to sweep all.",
            )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
