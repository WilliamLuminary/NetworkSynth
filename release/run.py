# run.py
"""
Unified entry point for the NetworkSynth pipelines.

Usage
-----
    python run.py generate    Generate synthetic networks matching each original
    python run.py hybrid      Build one large network: seed tiles + frontier continuation

The pipelines are config-driven: the only argument is which pipeline to run.
Everything else — datasets, frame sizes, closure factors, how many networks to
generate, the random seed — is set on the config class, so a run is fully
described by its config file. See configs/generate_mode/config_sample.py and
configs/hybrid_mode/config_sample.py.
"""
import fire


class CLI:
    """NetworkSynth — unified pipeline runner."""

    def generate(self):
        """Generate synthetic networks (parallel, quality-checked).

        Configured by configs/generate_mode/config_sample.py.
        """
        from pipelines.generate import main

        main()

    def hybrid(self):
        """Build one large network from parallel seed tiles + frontier continuation.

        Configured by configs/hybrid_mode/config_sample.py.
        """
        from pipelines.hybrid import main

        main()


if __name__ == "__main__":
    fire.Fire(CLI)
